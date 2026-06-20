mod error;
mod history;
mod knowledge;
mod output;
mod plan;
mod probe;
mod util;

use crate::engine::deep_scan::{scan_input, DeepScanConfig};
use crate::engine::lfas::{self, F3Sample, LfasConfig, LfasScheduler};
use crate::engine::pipeline::PipelineConfig;
pub use error::DoomQlockError;
use error::DoomQlockResult;
use history::{
    default_history_path, load_history, save_history, HistoryStore, PlanLookupQuery, RunRecord,
};
use knowledge::{default_knowledge_path, save_knowledge_snapshot};
use output::assess_output_health;
use plan::{ExecutionPlan, PlanOrigin, WorkloadEstimate};
use probe::{HardwareProbe, HardwareSnapshot};
use std::path::{Path, PathBuf};
use util::{
    content_profile_hash, display_disk_mbps, display_duration, display_opt_usize, display_ram,
    now_epoch_secs,
};

/// Number of LFAS arms: one per `QualityProfile` variant (Fast, Balanced, Strict).
const LFAS_ARMS: usize = 3;

#[derive(Debug, Clone)]
pub struct DoomQlock {
    history_path: PathBuf,
    knowledge_path: PathBuf,
    history: HistoryStore,
    /// F.4 Label-Free Adaptive Scheduler. Tracks quality-profile arms
    /// via F.3 counterfactual MI and provides coverage-regret guarantees
    /// (Theorems 1-3 in `docs/F4_lfas.md`).
    lfas: LfasScheduler<LFAS_ARMS>,
}

#[derive(Debug, Clone)]
pub struct PreparedRun {
    pub effective_config: PipelineConfig,
    device_fingerprint: String,
    content_profile_hash: String,
    hardware_snapshot: HardwareSnapshot,
    workload: WorkloadEstimate,
    plan: ExecutionPlan,
}

impl DoomQlock {
    pub fn load_default() -> Self {
        let history_path = default_history_path();
        let knowledge_path = default_knowledge_path(&history_path);
        let history = load_history(&history_path).unwrap_or_else(|error| {
            eprintln!(
                "warning: IBVoid DOOM-QLOCK could not load history at {} ({}); starting with empty cache.",
                history_path.display(),
                error
            );
            HistoryStore::default()
        });

        Self {
            history_path,
            knowledge_path,
            history,
            lfas: LfasScheduler::new(LfasConfig::default()),
        }
    }

    pub fn prepare_run(
        &self,
        input: &Path,
        base_config: &PipelineConfig,
    ) -> DoomQlockResult<PreparedRun> {
        eprintln!("ibvoid-doom-qlock: probing hardware...");
        let hardware = HardwareProbe::probe();
        let deep_scan = scan_input(
            input,
            DeepScanConfig {
                vad_threshold_db: base_config.vad_threshold_db,
                vad_min_silence: base_config.vad_min_silence,
                vad_pad: base_config.vad_pad,
            },
        )
        .ok();
        if let Some(content) = deep_scan.as_ref() {
            eprintln!(
                "ibvoid-doom-qlock: deep-scan speech={:.1}s silence={:.1}s scenes={} avg_difficulty={:.2}",
                content.speech_duration_secs,
                content.silence_duration_secs,
                content.scene_count,
                content.avg_difficulty
            );
        }
        let workload = WorkloadEstimate::probe(input, &base_config.source_lang, deep_scan.as_ref())
            .map_err(|message| DoomQlockError::WorkloadProbe {
                input: input.to_path_buf(),
                message,
            })?;
        let device_fingerprint = hardware.fingerprint();
        let content_profile_hash = deep_scan
            .as_ref()
            .map(content_profile_hash)
            .unwrap_or_else(|| "none".to_string());

        eprintln!(
            "ibvoid-doom-qlock: hardware cpu={} ram={} gpu={} disk_write={}MB/s",
            hardware.cpu_cores,
            display_ram(hardware.total_ram_mb),
            hardware.gpu_summary(),
            display_disk_mbps(hardware.disk_write_mbps)
        );
        eprintln!(
            "ibvoid-doom-qlock: workload kind={} duration={} estimated_cues={} language={} (confidence {:.2}, mode={})",
            workload.input_kind,
            display_duration(workload.duration_secs),
            workload.estimated_cues,
            workload.detected_source_lang,
            workload.detected_confidence,
            workload.detection_mode
        );
        if let (Some(difficulty), Some(speakers), Some(energy)) = (
            workload.avg_difficulty,
            workload.speaker_complexity,
            workload.energy_variance,
        ) {
            eprintln!(
                "ibvoid-doom-qlock: content difficulty={:.2} speaker_complexity={:.2} energy_variance={:.2}",
                difficulty, speakers, energy
            );
        }

        let lookup_query = PlanLookupQuery {
            source_lang: &base_config.source_lang,
            target_lang: &base_config.target_lang,
            profile: base_config.quality_profile,
            input_kind: &workload.input_kind,
            content_profile_hash: &content_profile_hash,
        };
        let learned_exact = self
            .history
            .best_plan_exact(&device_fingerprint, lookup_query);
        let learned_similar = if learned_exact.is_none() {
            self.history
                .best_plan_similar(&hardware, &workload, lookup_query)
        } else {
            None
        };

        let (origin, mut plan) = if let Some(plan) = learned_exact {
            (PlanOrigin::LearnedExact, plan)
        } else if let Some(plan) = learned_similar {
            (PlanOrigin::LearnedSimilar, plan)
        } else {
            (
                PlanOrigin::Heuristic,
                ExecutionPlan::heuristic(base_config, &hardware, &workload),
            )
        };

        plan.validate_and_adjust(base_config, &hardware)
            .map_err(|message| DoomQlockError::PlanValidation {
                input: input.to_path_buf(),
                message,
            })?;
        let effective_config = plan.apply_to_config(base_config);

        // F.4 LFAS: log the scheduler's quality recommendation and coverage floor.
        let lfas_arm = self.lfas.pick_arm();
        let lfas_floor = self.lfas.coverage_floor(0.10);
        let lfas_summary = self.lfas.summary();

        eprintln!(
            "ibvoid-doom-qlock: plan source={} parallel={} workers={} chunk={:.0}s mt_batch={} mt_tokens={} mt_oom_retries={} cpu_fallback={}",
            origin.as_str(),
            plan.parallel,
            plan.workers,
            plan.chunk_duration_secs,
            display_opt_usize(plan.mt_batch_size),
            display_opt_usize(plan.mt_max_batch_tokens),
            display_opt_usize(plan.mt_oom_retries),
            if plan.mt_allow_cpu_fallback {
                "on"
            } else {
                "off"
            }
        );
        eprintln!(
            "ibvoid-doom-qlock: lfas arm={} coverage_floor={:.3} regret={:.1} best_delta_i={:.3}",
            lfas_arm.0, lfas_floor, lfas_summary.estimated_regret, lfas_summary.best_mean_delta_i,
        );

        Ok(PreparedRun {
            effective_config,
            device_fingerprint,
            content_profile_hash,
            hardware_snapshot: HardwareSnapshot::from(&hardware),
            workload,
            plan,
        })
    }

    /// Feed an F.3 audit sample back to the LFAS scheduler.
    ///
    /// Call this after the F.3 audit runs on a processed chunk (typically
    /// every `audit_period`-th chunk — check `lfas_should_audit`). The
    /// quality profile of the prepared run determines which arm receives
    /// the feedback.
    pub fn record_f3_audit(&mut self, prepared: &PreparedRun, sample: F3Sample) {
        let arm = lfas::ArmId(quality_profile_to_arm(
            prepared.effective_config.quality_profile,
        ));
        self.lfas.record(arm, Some(sample));
    }

    /// Whether the LFAS scheduler wants an F.3 audit on this chunk.
    pub fn lfas_should_audit(&self) -> bool {
        self.lfas.should_audit()
    }

    /// Current LFAS coverage floor (Theorem 3). For structured logging.
    pub fn lfas_coverage_floor(&self, alpha: f64) -> f64 {
        self.lfas.coverage_floor(alpha)
    }

    /// Get the LFAS-recommended quality profile.
    ///
    /// Returns `Some(profile)` if LFAS has enough data to make a
    /// recommendation (past the initial exploration phase). Returns
    /// `None` if LFAS is still exploring or has no observations.
    pub fn lfas_recommended_profile(&self) -> Option<crate::engine::transcribe::QualityProfile> {
        use crate::engine::transcribe::QualityProfile;
        let arm = self.lfas.pick_arm();
        let summary = self.lfas.summary();
        // Only override if past exploration phase (each arm pulled at least once).
        if summary.arm_pulls.iter().all(|&n| n > 0) {
            Some(match arm.0 {
                0 => QualityProfile::Fast,
                1 => QualityProfile::Balanced,
                _ => QualityProfile::Strict,
            })
        } else {
            None
        }
    }

    pub fn record_success(&mut self, prepared: &PreparedRun, output: &Path, elapsed_secs: f64) {
        // Advance the LFAS step counter (no F.3 sample — that comes via
        // `record_f3_audit` when the audit actually runs).
        let arm = lfas::ArmId(quality_profile_to_arm(
            prepared.effective_config.quality_profile,
        ));
        self.lfas.record(arm, None);

        let health = assess_output_health(output).ok();
        let record = RunRecord {
            timestamp_epoch_secs: now_epoch_secs(),
            device_fingerprint: prepared.device_fingerprint.clone(),
            content_profile_hash: prepared.content_profile_hash.clone(),
            gpu_backend: prepared.hardware_snapshot.gpu_backend.clone(),
            gpu_vram_mb: prepared.hardware_snapshot.gpu_vram_mb,
            cpu_cores: Some(prepared.hardware_snapshot.cpu_cores),
            total_ram_mb: prepared.hardware_snapshot.total_ram_mb,
            disk_write_mbps: prepared.hardware_snapshot.disk_write_mbps,
            source_lang: prepared.effective_config.source_lang.clone(),
            target_lang: prepared.effective_config.target_lang.clone(),
            quality_profile: prepared
                .effective_config
                .quality_profile
                .as_str()
                .to_string(),
            input_kind: prepared.workload.input_kind.clone(),
            audio_duration_secs: prepared.workload.duration_secs,
            speech_duration_secs: prepared.workload.speech_duration_secs,
            estimated_cues: prepared.workload.estimated_cues,
            avg_difficulty: prepared.workload.avg_difficulty,
            speaker_complexity: prepared.workload.speaker_complexity,
            energy_variance: prepared.workload.energy_variance,
            scene_count: prepared.workload.scene_count,
            plan: prepared.plan.clone(),
            success: true,
            elapsed_secs,
            output_cues: health.map(|h| h.cue_count),
            output_non_empty_ratio: health.map(|h| h.non_empty_ratio),
            output_top_line_ratio: health.map(|h| h.top_line_ratio),
            error: None,
        };
        self.history.push_record(record);
        if let Err(error) = save_history(&self.history_path, &self.history) {
            eprintln!(
                "warning: IBVoid DOOM-QLOCK could not write history at {} ({}).",
                self.history_path.display(),
                error
            );
        }
        if let Err(error) = save_knowledge_snapshot(&self.knowledge_path, &self.history) {
            eprintln!(
                "warning: IBVoid DOOM-QLOCK could not write knowledge snapshot at {} ({}).",
                self.knowledge_path.display(),
                error
            );
        }
    }

    pub fn record_failure(&mut self, prepared: &PreparedRun, elapsed_secs: f64, error: &str) {
        let record = RunRecord {
            timestamp_epoch_secs: now_epoch_secs(),
            device_fingerprint: prepared.device_fingerprint.clone(),
            content_profile_hash: prepared.content_profile_hash.clone(),
            gpu_backend: prepared.hardware_snapshot.gpu_backend.clone(),
            gpu_vram_mb: prepared.hardware_snapshot.gpu_vram_mb,
            cpu_cores: Some(prepared.hardware_snapshot.cpu_cores),
            total_ram_mb: prepared.hardware_snapshot.total_ram_mb,
            disk_write_mbps: prepared.hardware_snapshot.disk_write_mbps,
            source_lang: prepared.effective_config.source_lang.clone(),
            target_lang: prepared.effective_config.target_lang.clone(),
            quality_profile: prepared
                .effective_config
                .quality_profile
                .as_str()
                .to_string(),
            input_kind: prepared.workload.input_kind.clone(),
            audio_duration_secs: prepared.workload.duration_secs,
            speech_duration_secs: prepared.workload.speech_duration_secs,
            estimated_cues: prepared.workload.estimated_cues,
            avg_difficulty: prepared.workload.avg_difficulty,
            speaker_complexity: prepared.workload.speaker_complexity,
            energy_variance: prepared.workload.energy_variance,
            scene_count: prepared.workload.scene_count,
            plan: prepared.plan.clone(),
            success: false,
            elapsed_secs,
            output_cues: None,
            output_non_empty_ratio: None,
            output_top_line_ratio: None,
            error: Some(error.to_string()),
        };
        self.history.push_record(record);
        if let Err(error) = save_history(&self.history_path, &self.history) {
            eprintln!(
                "warning: IBVoid DOOM-QLOCK could not write history at {} ({}).",
                self.history_path.display(),
                error
            );
        }
        if let Err(error) = save_knowledge_snapshot(&self.knowledge_path, &self.history) {
            eprintln!(
                "warning: IBVoid DOOM-QLOCK could not write knowledge snapshot at {} ({}).",
                self.knowledge_path.display(),
                error
            );
        }
    }
}

/// Map a `QualityProfile` to an LFAS arm index.
///
/// The mapping is deterministic and exhaustive — every profile variant
/// maps to exactly one arm in `[0, LFAS_ARMS)`.
fn quality_profile_to_arm(profile: crate::engine::transcribe::QualityProfile) -> usize {
    use crate::engine::transcribe::QualityProfile;
    match profile {
        QualityProfile::Fast => 0,
        QualityProfile::Balanced => 1,
        QualityProfile::Strict => 2,
    }
}

#[cfg(test)]
mod tests;
