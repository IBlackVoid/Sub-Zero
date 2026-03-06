use crate::engine::deep_scan::{scan_input, ContentMap, DeepScanConfig};
use crate::engine::pipeline::PipelineConfig;
use crate::engine::srt::{parse_srt_file, SubtitleCue};
use crate::engine::transcribe::QualityProfile;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const HISTORY_VERSION: u32 = 1;
const HISTORY_MAX_RECORDS: usize = 400;

#[derive(Debug, Clone)]
pub struct DoomQlock {
    history_path: PathBuf,
    knowledge_path: PathBuf,
    history: HistoryStore,
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

#[derive(Debug, Clone)]
struct HardwareSnapshot {
    cpu_cores: usize,
    total_ram_mb: Option<u64>,
    disk_write_mbps: Option<f64>,
    gpu_backend: Option<String>,
    gpu_vram_mb: Option<u64>,
}

impl From<&HardwareProbe> for HardwareSnapshot {
    fn from(value: &HardwareProbe) -> Self {
        Self {
            cpu_cores: value.cpu_cores,
            total_ram_mb: value.total_ram_mb,
            disk_write_mbps: value.disk_write_mbps,
            gpu_backend: value
                .gpu
                .as_ref()
                .map(|gpu| gpu.backend.to_ascii_lowercase()),
            gpu_vram_mb: value.gpu.as_ref().and_then(|gpu| gpu.vram_mb),
        }
    }
}

impl DoomQlock {
    pub fn load_default() -> Result<Self, String> {
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

        Ok(Self {
            history_path,
            knowledge_path,
            history,
        })
    }

    pub fn prepare_run(
        &self,
        input: &Path,
        base_config: &PipelineConfig,
    ) -> Result<PreparedRun, String> {
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
        let workload =
            WorkloadEstimate::probe(input, &base_config.source_lang, deep_scan.as_ref())?;
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

        plan.validate_and_adjust(base_config, &hardware)?;
        let effective_config = plan.apply_to_config(base_config);

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

        Ok(PreparedRun {
            effective_config,
            device_fingerprint,
            content_profile_hash,
            hardware_snapshot: HardwareSnapshot::from(&hardware),
            workload,
            plan,
        })
    }

    pub fn record_success(
        &mut self,
        prepared: &PreparedRun,
        output: &Path,
        elapsed_secs: f64,
    ) -> Result<(), String> {
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
        save_history(&self.history_path, &self.history)?;
        if let Err(error) = save_knowledge_snapshot(&self.knowledge_path, &self.history) {
            eprintln!(
                "warning: IBVoid DOOM-QLOCK could not write knowledge snapshot at {} ({}).",
                self.knowledge_path.display(),
                error
            );
        }
        Ok(())
    }

    pub fn record_failure(
        &mut self,
        prepared: &PreparedRun,
        elapsed_secs: f64,
        error: &str,
    ) -> Result<(), String> {
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
        save_history(&self.history_path, &self.history)?;
        if let Err(error) = save_knowledge_snapshot(&self.knowledge_path, &self.history) {
            eprintln!(
                "warning: IBVoid DOOM-QLOCK could not write knowledge snapshot at {} ({}).",
                self.knowledge_path.display(),
                error
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum PlanOrigin {
    Heuristic,
    LearnedExact,
    LearnedSimilar,
}

impl PlanOrigin {
    fn as_str(self) -> &'static str {
        match self {
            Self::Heuristic => "heuristic",
            Self::LearnedExact => "history-exact",
            Self::LearnedSimilar => "history-similar",
        }
    }
}

#[derive(Debug, Clone)]
struct HardwareProbe {
    cpu_cores: usize,
    total_ram_mb: Option<u64>,
    disk_write_mbps: Option<f64>,
    gpu: Option<GpuProbe>,
}

impl HardwareProbe {
    fn probe() -> Self {
        Self {
            cpu_cores: std::thread::available_parallelism()
                .map(|cores| cores.get())
                .unwrap_or(4),
            total_ram_mb: probe_total_ram_mb(),
            disk_write_mbps: probe_disk_write_mbps(),
            gpu: probe_gpu(),
        }
    }

    fn fingerprint(&self) -> String {
        let gpu_component = self
            .gpu
            .as_ref()
            .map(|gpu| {
                format!(
                    "{}-{}",
                    sanitize_fingerprint_component(&gpu.backend),
                    sanitize_fingerprint_component(&gpu.name)
                )
            })
            .unwrap_or_else(|| "none".to_string());
        let gpu_vram = self.gpu.as_ref().and_then(|gpu| gpu.vram_mb).unwrap_or(0);
        let disk_mbps = self.disk_write_mbps.unwrap_or(0.0).round() as u64;
        format!(
            "cpu{}-ram{}-gpu{}-vram{}-disk{}",
            self.cpu_cores,
            self.total_ram_mb.unwrap_or(0),
            gpu_component,
            gpu_vram,
            disk_mbps
        )
    }

    fn gpu_summary(&self) -> String {
        let Some(gpu) = &self.gpu else {
            return "none".to_string();
        };
        let vram = gpu
            .vram_mb
            .map(|value| format!("{value}MB"))
            .unwrap_or_else(|| "unknown-vram".to_string());
        let cc = gpu
            .compute_capability
            .as_ref()
            .map(|value| format!("cc={value}"))
            .unwrap_or_else(|| "cc=unknown".to_string());
        format!("{}:{} ({}, {})", gpu.backend, gpu.name, vram, cc)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct GpuProbe {
    backend: String,
    name: String,
    vram_mb: Option<u64>,
    compute_capability: Option<String>,
}

#[derive(Debug, Clone)]
struct WorkloadEstimate {
    input_kind: String,
    is_video: bool,
    duration_secs: Option<f64>,
    speech_duration_secs: Option<f64>,
    estimated_cues: usize,
    avg_difficulty: Option<f64>,
    speaker_complexity: Option<f64>,
    energy_variance: Option<f64>,
    scene_count: Option<usize>,
    detected_source_lang: String,
    detected_confidence: f64,
    detection_mode: String,
}

impl WorkloadEstimate {
    fn probe(
        input: &Path,
        source_lang_hint: &str,
        deep_scan: Option<&ContentMap>,
    ) -> Result<Self, String> {
        if is_srt_path(input) {
            let cues = parse_srt_file(input).map_err(|e| format!("{}: {e}", input.display()))?;
            let duration_secs = duration_from_cues(&cues);
            let mut workload = Self {
                input_kind: "srt".to_string(),
                is_video: false,
                duration_secs,
                speech_duration_secs: duration_secs,
                estimated_cues: cues.len(),
                avg_difficulty: None,
                speaker_complexity: None,
                energy_variance: None,
                scene_count: None,
                detected_source_lang: source_lang_hint.to_string(),
                detected_confidence: 0.85,
                detection_mode: "declared".to_string(),
            };
            if let Some(scan) = deep_scan {
                workload.estimated_cues = scan.estimated_cues.max(workload.estimated_cues);
                workload.avg_difficulty = Some(scan.avg_difficulty);
                workload.speaker_complexity = Some(scan.speaker_complexity_score);
                workload.energy_variance = Some(scan.energy_variance_score);
                workload.scene_count = Some(scan.scene_count);
                workload.speech_duration_secs = Some(scan.speech_duration_secs);
            }
            return Ok(workload);
        }

        let ffprobe = probe_media_format(input)?;
        let estimated_cues = ffprobe
            .duration_secs
            .map(|duration| ((duration / 3.0).round() as usize).max(1))
            .unwrap_or(400);
        let mut workload = Self {
            input_kind: ffprobe.format_name.unwrap_or_else(|| "video".to_string()),
            is_video: true,
            duration_secs: ffprobe.duration_secs,
            speech_duration_secs: ffprobe.duration_secs,
            estimated_cues,
            avg_difficulty: None,
            speaker_complexity: None,
            energy_variance: None,
            scene_count: None,
            detected_source_lang: source_lang_hint.to_string(),
            detected_confidence: 0.70,
            detection_mode: "hint".to_string(),
        };
        if let Some(scan) = deep_scan {
            workload.input_kind = scan.input_kind.clone();
            workload.duration_secs = Some(scan.total_duration_secs);
            workload.speech_duration_secs = Some(scan.speech_duration_secs);
            workload.estimated_cues = scan.estimated_cues;
            workload.avg_difficulty = Some(scan.avg_difficulty);
            workload.speaker_complexity = Some(scan.speaker_complexity_score);
            workload.energy_variance = Some(scan.energy_variance_score);
            workload.scene_count = Some(scan.scene_count);
            workload.detection_mode = "deep-scan".to_string();
            workload.detected_confidence = 0.92;
        }
        Ok(workload)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ExecutionPlan {
    parallel: bool,
    workers: usize,
    chunk_duration_secs: f64,
    mt_batch_size: Option<usize>,
    mt_max_batch_tokens: Option<usize>,
    mt_oom_retries: Option<usize>,
    mt_allow_cpu_fallback: bool,
}

impl ExecutionPlan {
    fn from_config(config: &PipelineConfig) -> Self {
        Self {
            parallel: config.parallel,
            workers: config.max_workers,
            chunk_duration_secs: config.chunk_duration_secs,
            mt_batch_size: config.mt_batch_size,
            mt_max_batch_tokens: config.mt_max_batch_tokens,
            mt_oom_retries: config.mt_oom_retries,
            mt_allow_cpu_fallback: config.mt_allow_cpu_fallback,
        }
    }

    fn heuristic(
        base_config: &PipelineConfig,
        hardware: &HardwareProbe,
        workload: &WorkloadEstimate,
    ) -> Self {
        let mut plan = Self::from_config(base_config);

        if base_config.transcribe && workload.is_video {
            plan.parallel = true;
        }

        let max_workers_by_cpu = hardware.cpu_cores.clamp(1, 16);
        let max_workers_by_ram = hardware
            .total_ram_mb
            .map(|ram| (ram / 2_048).clamp(1, 16) as usize)
            .unwrap_or(8);
        let recommended_workers = max_workers_by_cpu.min(max_workers_by_ram).max(1);
        plan.workers = plan.workers.min(recommended_workers).max(1);

        if plan.parallel && workload.is_video {
            plan.chunk_duration_secs =
                duration_to_chunk_secs(workload.duration_secs, workload.avg_difficulty);
        }

        if let Some(difficulty) = workload.avg_difficulty {
            if difficulty >= 0.70 {
                plan.workers = plan.workers.saturating_sub(1).max(1);
                plan.mt_oom_retries = Some(plan.mt_oom_retries.unwrap_or(3).max(3));
            }
        }
        if let Some(speaker_complexity) = workload.speaker_complexity {
            if speaker_complexity >= 0.60 {
                let batch = plan
                    .mt_batch_size
                    .unwrap_or(default_mt_batch_for_profile(base_config.quality_profile));
                let tokens = plan
                    .mt_max_batch_tokens
                    .unwrap_or(default_mt_max_tokens_for_profile(
                        base_config.quality_profile,
                    ));
                plan.mt_batch_size = Some((batch * 3 / 4).max(8));
                plan.mt_max_batch_tokens = Some((tokens * 3 / 4).max(2048));
                plan.mt_oom_retries = Some(plan.mt_oom_retries.unwrap_or(3).max(4));
            }
        }
        if let Some(energy_variance) = workload.energy_variance {
            if energy_variance >= 0.50 {
                plan.chunk_duration_secs = (plan.chunk_duration_secs * 0.85).max(120.0);
            }
        }

        if base_config.mt_batch_size.is_none()
            || base_config.mt_max_batch_tokens.is_none()
            || base_config.mt_oom_retries.is_none()
        {
            let tuning = choose_mt_tuning(hardware, base_config.quality_profile, base_config.gpu);
            if base_config.mt_batch_size.is_none() {
                plan.mt_batch_size = Some(tuning.batch_size);
            }
            if base_config.mt_max_batch_tokens.is_none() {
                plan.mt_max_batch_tokens = Some(tuning.max_batch_tokens);
            }
            if base_config.mt_oom_retries.is_none() {
                plan.mt_oom_retries = Some(tuning.oom_retries);
            }
        }

        plan.mt_allow_cpu_fallback = !base_config.require_gpu && plan.mt_allow_cpu_fallback;
        plan
    }

    fn validate_and_adjust(
        &mut self,
        base_config: &PipelineConfig,
        hardware: &HardwareProbe,
    ) -> Result<(), String> {
        if self.workers == 0 {
            self.workers = 1;
        }
        if self.chunk_duration_secs < 30.0 {
            self.chunk_duration_secs = 30.0;
        }
        if base_config.require_gpu && hardware.gpu.is_none() {
            return Err(
                "IBVoid DOOM-QLOCK: --require-gpu is set, but no CUDA GPU was detected."
                    .to_string(),
            );
        }

        if let Some(gpu) = &hardware.gpu {
            if let Some(vram_mb) = gpu.vram_mb {
                self.shrink_mt_plan_to_vram(vram_mb, base_config.quality_profile);
            }
        }

        Ok(())
    }

    fn shrink_mt_plan_to_vram(&mut self, vram_mb: u64, profile: QualityProfile) {
        let budget_mb = (vram_mb.saturating_mul(85)).max(1) / 100;
        loop {
            let estimate = self.estimate_mt_vram_mb(profile);
            if estimate <= budget_mb {
                break;
            }

            let current_batch = self
                .mt_batch_size
                .unwrap_or(default_mt_batch_for_profile(profile));
            let current_tokens = self
                .mt_max_batch_tokens
                .unwrap_or(default_mt_max_tokens_for_profile(profile));

            if current_batch <= 4 && current_tokens <= 1_024 {
                break;
            }

            self.mt_batch_size = Some((current_batch.saturating_mul(3) / 4).max(4));
            self.mt_max_batch_tokens = Some((current_tokens.saturating_mul(3) / 4).max(1_024));
            let retries = self
                .mt_oom_retries
                .unwrap_or(default_mt_oom_retries(profile));
            self.mt_oom_retries = Some((retries + 1).min(8));
        }
    }

    fn estimate_mt_vram_mb(&self, profile: QualityProfile) -> u64 {
        let batch = self
            .mt_batch_size
            .unwrap_or(default_mt_batch_for_profile(profile)) as u64;
        let tokens = self
            .mt_max_batch_tokens
            .unwrap_or(default_mt_max_tokens_for_profile(profile)) as u64;
        2_048 + (batch * 180) + (tokens / 2)
    }

    fn apply_to_config(&self, base_config: &PipelineConfig) -> PipelineConfig {
        let mut effective = base_config.clone();
        effective.parallel = self.parallel;
        effective.max_workers = self.workers;
        effective.chunk_duration_secs = self.chunk_duration_secs;
        effective.mt_batch_size = self.mt_batch_size;
        effective.mt_max_batch_tokens = self.mt_max_batch_tokens;
        effective.mt_oom_retries = self.mt_oom_retries;
        effective.mt_allow_cpu_fallback = self.mt_allow_cpu_fallback && !base_config.require_gpu;
        effective
    }
}

#[derive(Debug, Clone, Copy)]
struct MtTuning {
    batch_size: usize,
    max_batch_tokens: usize,
    oom_retries: usize,
}

fn choose_mt_tuning(
    hardware: &HardwareProbe,
    profile: QualityProfile,
    prefer_gpu: bool,
) -> MtTuning {
    if !prefer_gpu {
        return MtTuning {
            batch_size: default_mt_batch_for_profile(profile),
            max_batch_tokens: default_mt_max_tokens_for_profile(profile),
            oom_retries: default_mt_oom_retries(profile),
        };
    }

    let mut tuning = match hardware.gpu.as_ref().and_then(|gpu| gpu.vram_mb) {
        Some(vram) if vram <= 4_096 => MtTuning {
            batch_size: 8,
            max_batch_tokens: 2_048,
            oom_retries: 4,
        },
        Some(vram) if vram <= 6_144 => MtTuning {
            batch_size: 12,
            max_batch_tokens: 3_072,
            oom_retries: 4,
        },
        Some(vram) if vram <= 8_192 => MtTuning {
            batch_size: 16,
            max_batch_tokens: 4_096,
            oom_retries: 3,
        },
        Some(vram) if vram <= 12_288 => MtTuning {
            batch_size: 24,
            max_batch_tokens: 6_144,
            oom_retries: 2,
        },
        Some(_) => MtTuning {
            batch_size: 32,
            max_batch_tokens: 8_192,
            oom_retries: 2,
        },
        None => MtTuning {
            batch_size: default_mt_batch_for_profile(profile),
            max_batch_tokens: default_mt_max_tokens_for_profile(profile),
            oom_retries: default_mt_oom_retries(profile),
        },
    };

    match profile {
        QualityProfile::Fast => {}
        QualityProfile::Balanced => {
            tuning.batch_size = tuning.batch_size.min(24);
            tuning.max_batch_tokens = tuning.max_batch_tokens.min(6_144);
            tuning.oom_retries = tuning.oom_retries.max(2);
        }
        QualityProfile::Strict => {
            tuning.batch_size = tuning.batch_size.min(16);
            tuning.max_batch_tokens = tuning.max_batch_tokens.min(4_096);
            tuning.oom_retries = tuning.oom_retries.max(3);
        }
    }

    if prefer_gpu {
        let backend = hardware
            .gpu
            .as_ref()
            .map(|gpu| gpu.backend.to_ascii_lowercase());
        match backend.as_deref() {
            Some("cuda") => {}
            Some("rocm") => {
                tuning.batch_size = ((tuning.batch_size as f64) * 0.85).round() as usize;
                tuning.max_batch_tokens =
                    ((tuning.max_batch_tokens as f64) * 0.85).round() as usize;
                tuning.oom_retries = (tuning.oom_retries + 1).min(8);
            }
            Some("metal") => {
                tuning.batch_size = ((tuning.batch_size as f64) * 0.75).round() as usize;
                tuning.max_batch_tokens =
                    ((tuning.max_batch_tokens as f64) * 0.75).round() as usize;
                tuning.oom_retries = (tuning.oom_retries + 1).min(8);
            }
            Some(_) => {
                tuning.batch_size = ((tuning.batch_size as f64) * 0.80).round() as usize;
                tuning.max_batch_tokens =
                    ((tuning.max_batch_tokens as f64) * 0.80).round() as usize;
                tuning.oom_retries = (tuning.oom_retries + 1).min(8);
            }
            None => {}
        }
    }

    tuning.batch_size = tuning.batch_size.max(4);
    tuning.max_batch_tokens = tuning.max_batch_tokens.max(1024);
    tuning
}

fn duration_to_chunk_secs(duration_secs: Option<f64>, avg_difficulty: Option<f64>) -> f64 {
    let mut chunk: f64 = match duration_secs {
        Some(duration) if duration >= 7_200.0 => 360.0,
        Some(duration) if duration >= 3_600.0 => 300.0,
        Some(duration) if duration >= 1_800.0 => 240.0,
        Some(_) => 180.0,
        None => 300.0,
    };
    if let Some(difficulty) = avg_difficulty {
        if difficulty >= 0.70 {
            chunk = (chunk * 0.75).max(120.0);
        } else if difficulty <= 0.30 {
            chunk = (chunk * 1.15).min(420.0);
        }
    }
    chunk
}

fn default_mt_batch_for_profile(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 32,
        QualityProfile::Balanced => 24,
        QualityProfile::Strict => 16,
    }
}

fn default_mt_max_tokens_for_profile(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 8_192,
        QualityProfile::Balanced => 6_144,
        QualityProfile::Strict => 4_096,
    }
}

fn default_mt_oom_retries(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 1,
        QualityProfile::Balanced => 2,
        QualityProfile::Strict => 3,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunRecord {
    timestamp_epoch_secs: u64,
    device_fingerprint: String,
    #[serde(default = "default_content_profile_hash")]
    content_profile_hash: String,
    #[serde(default)]
    gpu_backend: Option<String>,
    #[serde(default)]
    gpu_vram_mb: Option<u64>,
    #[serde(default)]
    cpu_cores: Option<usize>,
    #[serde(default)]
    total_ram_mb: Option<u64>,
    #[serde(default)]
    disk_write_mbps: Option<f64>,
    source_lang: String,
    target_lang: String,
    quality_profile: String,
    input_kind: String,
    audio_duration_secs: Option<f64>,
    speech_duration_secs: Option<f64>,
    estimated_cues: usize,
    avg_difficulty: Option<f64>,
    speaker_complexity: Option<f64>,
    energy_variance: Option<f64>,
    scene_count: Option<usize>,
    plan: ExecutionPlan,
    success: bool,
    elapsed_secs: f64,
    output_cues: Option<usize>,
    output_non_empty_ratio: Option<f64>,
    output_top_line_ratio: Option<f64>,
    error: Option<String>,
}

fn default_content_profile_hash() -> String {
    "none".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistoryStore {
    version: u32,
    records: Vec<RunRecord>,
}

#[derive(Debug, Clone, Copy)]
struct PlanLookupQuery<'a> {
    source_lang: &'a str,
    target_lang: &'a str,
    profile: QualityProfile,
    input_kind: &'a str,
    content_profile_hash: &'a str,
}

impl Default for HistoryStore {
    fn default() -> Self {
        Self {
            version: HISTORY_VERSION,
            records: Vec::new(),
        }
    }
}

impl HistoryStore {
    fn push_record(&mut self, record: RunRecord) {
        self.records.push(record);
        if self.records.len() > HISTORY_MAX_RECORDS {
            let overflow = self.records.len() - HISTORY_MAX_RECORDS;
            self.records.drain(0..overflow);
        }
    }

    fn best_plan_exact(
        &self,
        device_fingerprint: &str,
        query: PlanLookupQuery<'_>,
    ) -> Option<ExecutionPlan> {
        let expected_profile = query.profile.as_str();
        self.records
            .iter()
            .filter(|record| {
                record.success
                    && record.device_fingerprint == device_fingerprint
                    && record.source_lang.eq_ignore_ascii_case(query.source_lang)
                    && record.target_lang.eq_ignore_ascii_case(query.target_lang)
                    && record.input_kind == query.input_kind
                    && record.content_profile_hash == query.content_profile_hash
                    && record.quality_profile == expected_profile
                    && record_passes_quality_gate(record)
            })
            .min_by(|a, b| a.elapsed_secs.total_cmp(&b.elapsed_secs))
            .map(|record| record.plan.clone())
    }

    fn best_plan_similar(
        &self,
        hardware: &HardwareProbe,
        workload: &WorkloadEstimate,
        query: PlanLookupQuery<'_>,
    ) -> Option<ExecutionPlan> {
        let expected_profile = query.profile.as_str();
        let target_backend = hardware
            .gpu
            .as_ref()
            .map(|gpu| gpu.backend.to_ascii_lowercase());
        let target_vram = hardware.gpu.as_ref().and_then(|gpu| gpu.vram_mb);

        self.records
            .iter()
            .filter(|record| {
                record.success
                    && record.source_lang.eq_ignore_ascii_case(query.source_lang)
                    && record.target_lang.eq_ignore_ascii_case(query.target_lang)
                    && record.input_kind == query.input_kind
                    && record.quality_profile == expected_profile
                    && record_passes_quality_gate(record)
                    && record_matches_hardware(record, target_backend.as_deref(), target_vram)
            })
            .min_by(|a, b| {
                let lhs = similarity_score(a, workload, target_vram, query.content_profile_hash);
                let rhs = similarity_score(b, workload, target_vram, query.content_profile_hash);
                lhs.total_cmp(&rhs)
            })
            .map(|record| record.plan.clone())
    }
}

fn record_passes_quality_gate(record: &RunRecord) -> bool {
    record.output_non_empty_ratio.unwrap_or(1.0) >= 0.80
        && (record.output_cues.unwrap_or(0) < 80
            || record.output_top_line_ratio.unwrap_or(0.0) <= 0.30)
}

fn record_matches_hardware(
    record: &RunRecord,
    target_backend: Option<&str>,
    target_vram: Option<u64>,
) -> bool {
    let record_backend = record_backend(record);
    match target_backend {
        Some(target) => {
            if record_backend.as_deref() != Some(target) {
                return false;
            }
        }
        None => {
            if record_backend
                .as_deref()
                .is_some_and(|value| value != "none")
            {
                return false;
            }
        }
    }

    if let (Some(target), Some(record_vram)) = (target_vram, record_vram_mb(record)) {
        // Reject plans learned on materially larger VRAM footprints.
        if record_vram > (target.saturating_mul(6) / 5) {
            return false;
        }
    }
    true
}

fn similarity_score(
    record: &RunRecord,
    workload: &WorkloadEstimate,
    target_vram: Option<u64>,
    content_profile_hash: &str,
) -> f64 {
    let mut score = elapsed_per_audio_hour(record);
    if !score.is_finite() {
        score = record.elapsed_secs.max(1.0);
    }

    if record.content_profile_hash != content_profile_hash {
        score += 45.0;
    }

    if let (Some(a), Some(b)) = (record.avg_difficulty, workload.avg_difficulty) {
        score += (a - b).abs() * 320.0;
    } else {
        score += 25.0;
    }

    if let (Some(a), Some(b)) = (record.scene_count, workload.scene_count) {
        let max_count = a.max(b).max(1) as f64;
        let diff_ratio = ((a as f64 - b as f64).abs() / max_count).clamp(0.0, 1.0);
        score += diff_ratio * 180.0;
    } else {
        score += 20.0;
    }

    if let (Some(target), Some(record_vram)) = (target_vram, record_vram_mb(record)) {
        if record_vram > target {
            score += ((record_vram - target) as f64 / 256.0) * 8.0;
        } else {
            score -= ((target - record_vram) as f64 / 1024.0).min(20.0);
        }
    }

    score
}

fn elapsed_per_audio_hour(record: &RunRecord) -> f64 {
    if let Some(duration_secs) = record.audio_duration_secs {
        if duration_secs.is_finite() && duration_secs > 0.0 {
            return record.elapsed_secs / (duration_secs / 3600.0);
        }
    }
    record.elapsed_secs
}

fn record_backend(record: &RunRecord) -> Option<String> {
    record
        .gpu_backend
        .as_ref()
        .map(|value| value.to_ascii_lowercase())
        .or_else(|| infer_backend_from_fingerprint(&record.device_fingerprint))
}

fn record_vram_mb(record: &RunRecord) -> Option<u64> {
    record
        .gpu_vram_mb
        .or_else(|| parse_prefixed_u64(&record.device_fingerprint, "-vram"))
}

fn infer_backend_from_fingerprint(device_fingerprint: &str) -> Option<String> {
    let lowered = device_fingerprint.to_ascii_lowercase();
    if lowered.contains("-gpucuda") {
        Some("cuda".to_string())
    } else if lowered.contains("-gpurocm") {
        Some("rocm".to_string())
    } else if lowered.contains("-gpumetal") {
        Some("metal".to_string())
    } else if lowered.contains("-gpunone") {
        Some("none".to_string())
    } else {
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KnowledgeSnapshot {
    version: u32,
    generated_at_epoch_secs: u64,
    devices: HashMap<String, DeviceKnowledge>,
    language_pairs: HashMap<String, LanguagePairKnowledge>,
    content_kinds: HashMap<String, ContentKnowledge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeviceKnowledge {
    runs: usize,
    success_rate: f64,
    avg_elapsed_secs: f64,
    best_elapsed_secs: Option<f64>,
    best_plan: Option<ExecutionPlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LanguagePairKnowledge {
    runs: usize,
    success_rate: f64,
    avg_elapsed_secs: f64,
    avg_duration_secs: f64,
    avg_elapsed_per_audio_hour_secs: Option<f64>,
    best_elapsed_secs: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContentKnowledge {
    runs: usize,
    success_rate: f64,
    avg_elapsed_secs: f64,
    avg_difficulty: Option<f64>,
    avg_scene_count: Option<f64>,
}

#[derive(Debug, Clone)]
struct KnowledgeAccum {
    runs: usize,
    success: usize,
    elapsed_sum: f64,
    best_elapsed: Option<f64>,
    best_plan: Option<ExecutionPlan>,
    duration_sum: f64,
    duration_count: usize,
    difficulty_sum: f64,
    difficulty_count: usize,
    scene_sum: f64,
    scene_count: usize,
}

impl KnowledgeAccum {
    fn observe(&mut self, record: &RunRecord) {
        self.runs += 1;
        if record.success {
            self.success += 1;
        }
        if record.elapsed_secs.is_finite() && record.elapsed_secs >= 0.0 {
            self.elapsed_sum += record.elapsed_secs;
            if record.success {
                let is_better = self
                    .best_elapsed
                    .map(|best| record.elapsed_secs < best)
                    .unwrap_or(true);
                if is_better {
                    self.best_elapsed = Some(record.elapsed_secs);
                    self.best_plan = Some(record.plan.clone());
                }
            }
        }
        if let Some(duration) = record.audio_duration_secs {
            if duration.is_finite() && duration > 0.0 {
                self.duration_sum += duration;
                self.duration_count += 1;
            }
        }
        if let Some(difficulty) = record.avg_difficulty {
            if difficulty.is_finite() {
                self.difficulty_sum += difficulty;
                self.difficulty_count += 1;
            }
        }
        if let Some(scene_count) = record.scene_count {
            self.scene_sum += scene_count as f64;
            self.scene_count += 1;
        }
    }

    fn success_rate(&self) -> f64 {
        if self.runs == 0 {
            0.0
        } else {
            self.success as f64 / self.runs as f64
        }
    }

    fn avg_elapsed_secs(&self) -> f64 {
        if self.runs == 0 {
            0.0
        } else {
            self.elapsed_sum / self.runs as f64
        }
    }
}

fn build_knowledge_snapshot(history: &HistoryStore) -> KnowledgeSnapshot {
    let mut device_accum = HashMap::<String, KnowledgeAccum>::new();
    let mut pair_accum = HashMap::<String, KnowledgeAccum>::new();
    let mut content_accum = HashMap::<String, KnowledgeAccum>::new();

    for record in &history.records {
        device_accum
            .entry(device_knowledge_key(record))
            .or_insert_with(|| KnowledgeAccum {
                runs: 0,
                success: 0,
                elapsed_sum: 0.0,
                best_elapsed: None,
                best_plan: None,
                duration_sum: 0.0,
                duration_count: 0,
                difficulty_sum: 0.0,
                difficulty_count: 0,
                scene_sum: 0.0,
                scene_count: 0,
            })
            .observe(record);

        let pair_key = format!(
            "{}->{}",
            record.source_lang.to_ascii_lowercase(),
            record.target_lang.to_ascii_lowercase()
        );
        pair_accum
            .entry(pair_key)
            .or_insert_with(|| KnowledgeAccum {
                runs: 0,
                success: 0,
                elapsed_sum: 0.0,
                best_elapsed: None,
                best_plan: None,
                duration_sum: 0.0,
                duration_count: 0,
                difficulty_sum: 0.0,
                difficulty_count: 0,
                scene_sum: 0.0,
                scene_count: 0,
            })
            .observe(record);

        content_accum
            .entry(record.input_kind.clone())
            .or_insert_with(|| KnowledgeAccum {
                runs: 0,
                success: 0,
                elapsed_sum: 0.0,
                best_elapsed: None,
                best_plan: None,
                duration_sum: 0.0,
                duration_count: 0,
                difficulty_sum: 0.0,
                difficulty_count: 0,
                scene_sum: 0.0,
                scene_count: 0,
            })
            .observe(record);
    }

    let mut devices = HashMap::<String, DeviceKnowledge>::new();
    for (key, accum) in device_accum {
        devices.insert(
            key,
            DeviceKnowledge {
                runs: accum.runs,
                success_rate: accum.success_rate(),
                avg_elapsed_secs: accum.avg_elapsed_secs(),
                best_elapsed_secs: accum.best_elapsed,
                best_plan: accum.best_plan,
            },
        );
    }

    let mut language_pairs = HashMap::<String, LanguagePairKnowledge>::new();
    for (key, accum) in pair_accum {
        let avg_duration_secs = if accum.duration_count == 0 {
            0.0
        } else {
            accum.duration_sum / accum.duration_count as f64
        };
        let avg_elapsed_per_audio_hour_secs =
            if accum.duration_count == 0 || avg_duration_secs <= 0.0 {
                None
            } else {
                Some((accum.avg_elapsed_secs() / avg_duration_secs) * 3600.0)
            };
        language_pairs.insert(
            key,
            LanguagePairKnowledge {
                runs: accum.runs,
                success_rate: accum.success_rate(),
                avg_elapsed_secs: accum.avg_elapsed_secs(),
                avg_duration_secs,
                avg_elapsed_per_audio_hour_secs,
                best_elapsed_secs: accum.best_elapsed,
            },
        );
    }

    let mut content_kinds = HashMap::<String, ContentKnowledge>::new();
    for (key, accum) in content_accum {
        content_kinds.insert(
            key,
            ContentKnowledge {
                runs: accum.runs,
                success_rate: accum.success_rate(),
                avg_elapsed_secs: accum.avg_elapsed_secs(),
                avg_difficulty: if accum.difficulty_count == 0 {
                    None
                } else {
                    Some(accum.difficulty_sum / accum.difficulty_count as f64)
                },
                avg_scene_count: if accum.scene_count == 0 {
                    None
                } else {
                    Some(accum.scene_sum / accum.scene_count as f64)
                },
            },
        );
    }

    KnowledgeSnapshot {
        version: HISTORY_VERSION,
        generated_at_epoch_secs: now_epoch_secs(),
        devices,
        language_pairs,
        content_kinds,
    }
}

fn save_knowledge_snapshot(path: &Path, history: &HistoryStore) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create knowledge directory {}: {error}",
                parent.display()
            )
        })?;
    }
    let snapshot = build_knowledge_snapshot(history);
    let serialized = serde_json::to_string_pretty(&snapshot)
        .map_err(|error| format!("failed to serialize knowledge snapshot: {error}"))?;
    fs::write(path, serialized).map_err(|error| {
        format!(
            "failed to write knowledge snapshot {}: {error}",
            path.display()
        )
    })
}

#[derive(Debug, Clone, Copy)]
struct OutputHealthSnapshot {
    cue_count: usize,
    non_empty_ratio: f64,
    top_line_ratio: f64,
}

fn assess_output_health(path: &Path) -> Result<OutputHealthSnapshot, String> {
    let cues = parse_srt_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if cues.is_empty() {
        return Ok(OutputHealthSnapshot {
            cue_count: 0,
            non_empty_ratio: 0.0,
            top_line_ratio: 0.0,
        });
    }

    let mut freq = HashMap::<String, usize>::new();
    let mut non_empty = 0usize;
    for cue in &cues {
        let normalized = cue
            .text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        if normalized.is_empty() {
            continue;
        }
        non_empty += 1;
        *freq.entry(normalized).or_insert(0) += 1;
    }

    let top_count = freq.values().copied().max().unwrap_or(0);
    let total = cues.len() as f64;
    Ok(OutputHealthSnapshot {
        cue_count: cues.len(),
        non_empty_ratio: (non_empty as f64) / total,
        top_line_ratio: (top_count as f64) / total,
    })
}

#[derive(Debug, Clone, Default)]
struct MediaProbe {
    format_name: Option<String>,
    duration_secs: Option<f64>,
}

fn probe_media_format(input: &Path) -> Result<MediaProbe, String> {
    let ffprobe = find_in_path(&["ffprobe", "ffprobe.exe"])
        .ok_or_else(|| "ffprobe not found in PATH".to_string())?;

    let output = Command::new(ffprobe)
        .arg("-hide_banner")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration,format_name")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=0")
        .arg(input)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("failed to spawn ffprobe: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffprobe failed with status {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    parse_ffprobe_output(&String::from_utf8_lossy(&output.stdout))
}

fn parse_ffprobe_output(stdout: &str) -> Result<MediaProbe, String> {
    let mut probe = MediaProbe::default();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("format_name=") {
            if !value.is_empty() {
                probe.format_name = Some(value.to_string());
            }
            continue;
        }
        if let Some(value) = trimmed.strip_prefix("duration=") {
            let duration = value
                .parse::<f64>()
                .map_err(|_| format!("invalid ffprobe duration: {value}"))?;
            if duration.is_finite() && duration > 0.0 {
                probe.duration_secs = Some(duration);
            }
        }
    }
    Ok(probe)
}

fn duration_from_cues(cues: &[SubtitleCue]) -> Option<f64> {
    let mut max_end = 0.0f64;
    for cue in cues {
        if let Ok((_, end)) = parse_srt_timing_line(&cue.timing) {
            max_end = max_end.max(end);
        }
    }
    if max_end > 0.0 {
        Some(max_end)
    } else {
        None
    }
}

fn parse_srt_timing_line(line: &str) -> Result<(f64, f64), String> {
    let (start, end) = line
        .split_once("-->")
        .ok_or_else(|| format!("invalid timing line: {line}"))?;
    let start = parse_srt_timestamp_to_seconds(start)?;
    let end = parse_srt_timestamp_to_seconds(end)?;
    if end < start {
        return Err(format!("invalid timing (end < start): {line}"));
    }
    Ok((start, end))
}

fn parse_srt_timestamp_to_seconds(ts: &str) -> Result<f64, String> {
    let ts = ts.trim();
    let (hms, ms) = ts
        .split_once(',')
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?;
    let mut parts = hms.split(':');
    let hours = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let minutes = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let seconds = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let millis = ms
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;

    Ok((hours as f64) * 3600.0
        + (minutes as f64) * 60.0
        + (seconds as f64)
        + (millis as f64) / 1000.0)
}

fn is_srt_path(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|ext| ext.eq_ignore_ascii_case("srt"))
        .unwrap_or(false)
}

fn probe_total_ram_mb() -> Option<u64> {
    if cfg!(target_os = "linux") {
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if let Some(value) = line.strip_prefix("MemTotal:") {
                    let kb = value
                        .split_whitespace()
                        .next()
                        .and_then(|chunk| chunk.parse::<u64>().ok())?;
                    return Some(kb / 1024);
                }
            }
        }
    }

    if cfg!(target_os = "windows") {
        let output = Command::new("wmic")
            .arg("ComputerSystem")
            .arg("get")
            .arg("TotalPhysicalMemory")
            .arg("/value")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if let Some(value) = line.trim().strip_prefix("TotalPhysicalMemory=") {
                    let bytes = value.trim().parse::<u64>().ok()?;
                    return Some(bytes / 1024 / 1024);
                }
            }
        }
    }

    None
}

fn probe_gpu() -> Option<GpuProbe> {
    probe_nvidia_gpu()
        .or_else(probe_rocm_gpu)
        .or_else(probe_metal_gpu)
}

fn probe_nvidia_gpu() -> Option<GpuProbe> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total,compute_cap")
        .arg("--format=csv,noheader,nounits")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().find(|line| !line.trim().is_empty())?;
    let parts: Vec<&str> = first_line.split(',').map(str::trim).collect();
    let name = parts.first()?.to_string();
    let vram_mb = parts.get(1).and_then(|value| value.parse::<u64>().ok());
    let compute_capability = parts.get(2).map(|value| value.to_string());

    Some(GpuProbe {
        backend: "cuda".to_string(),
        name,
        vram_mb,
        compute_capability,
    })
}

fn probe_rocm_gpu() -> Option<GpuProbe> {
    // Try rocm-smi first.
    let rocm_smi = find_in_path(&["rocm-smi", "rocm-smi.exe"])?;
    let output = Command::new(rocm_smi)
        .arg("--showproductname")
        .arg("--showmeminfo")
        .arg("vram")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut name = None::<String>;
    let mut vram_mb = None::<u64>;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if name.is_none()
            && (trimmed.to_ascii_lowercase().contains("card series")
                || trimmed.to_ascii_lowercase().contains("product name"))
        {
            name = trimmed
                .split(':')
                .nth(1)
                .map(str::trim)
                .map(ToString::to_string)
                .filter(|value| !value.is_empty());
        }
        if vram_mb.is_none() && trimmed.to_ascii_lowercase().contains("total") {
            let mb = trimmed
                .split_whitespace()
                .find_map(|part| part.parse::<u64>().ok())
                .map(|kb_or_mb| {
                    if kb_or_mb > 200_000 {
                        kb_or_mb / 1024
                    } else {
                        kb_or_mb
                    }
                });
            vram_mb = mb;
        }
    }
    let name = name.unwrap_or_else(|| "AMD GPU".to_string());
    Some(GpuProbe {
        backend: "rocm".to_string(),
        name,
        vram_mb,
        compute_capability: None,
    })
}

fn probe_metal_gpu() -> Option<GpuProbe> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let output = Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut chip = None::<String>;
    let mut vram_mb = None::<u64>;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if chip.is_none() && trimmed.starts_with("Chipset Model:") {
            chip = trimmed
                .split_once(':')
                .map(|(_, value)| value.trim().to_string())
                .filter(|value| !value.is_empty());
        }
        if vram_mb.is_none() && trimmed.starts_with("VRAM") {
            let parsed = trimmed
                .split_whitespace()
                .find_map(|part| part.parse::<u64>().ok())
                .map(|value| {
                    if trimmed.to_ascii_lowercase().contains("gb") {
                        value * 1024
                    } else {
                        value
                    }
                });
            vram_mb = parsed;
        }
    }
    let name = chip.unwrap_or_else(|| "Apple GPU".to_string());
    Some(GpuProbe {
        backend: "metal".to_string(),
        name,
        vram_mb,
        compute_capability: None,
    })
}

fn probe_disk_write_mbps() -> Option<f64> {
    let root = std::env::var_os("SUB_ZERO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(PathBuf::from))
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .unwrap_or_else(std::env::temp_dir);
    let probe_dir = root.join(".sub-zero");
    if fs::create_dir_all(&probe_dir).is_err() {
        return None;
    }
    let probe_file = probe_dir.join(".disk_probe.bin");
    let payload = vec![0u8; 1_048_576];
    let start = std::time::Instant::now();
    if fs::write(&probe_file, &payload).is_err() {
        return None;
    }
    let elapsed = start.elapsed().as_secs_f64().max(0.000_5);
    let _ = fs::remove_file(&probe_file);
    Some((payload.len() as f64 / 1_048_576.0) / elapsed)
}

fn default_history_path() -> PathBuf {
    if let Some(home) = std::env::var_os("SUB_ZERO_HOME") {
        return PathBuf::from(home).join("history.json");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".sub-zero").join("history.json");
    }
    if let Some(home) = std::env::var_os("USERPROFILE") {
        return PathBuf::from(home).join(".sub-zero").join("history.json");
    }
    PathBuf::from(".sub-zero-history.json")
}

fn default_knowledge_path(history_path: &Path) -> PathBuf {
    history_path.with_file_name("knowledge.json")
}

fn load_history(path: &Path) -> Result<HistoryStore, String> {
    if !path.is_file() {
        return Ok(HistoryStore::default());
    }
    let content = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let mut history: HistoryStore =
        serde_json::from_str(&content).map_err(|error| format!("invalid history JSON: {error}"))?;
    if history.version == 0 {
        history.version = HISTORY_VERSION;
    }
    Ok(history)
}

fn save_history(path: &Path, history: &HistoryStore) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create history directory {}: {error}",
                parent.display()
            )
        })?;
    }
    let serialized = serde_json::to_string_pretty(history)
        .map_err(|error| format!("failed to serialize history: {error}"))?;
    fs::write(path, serialized)
        .map_err(|error| format!("failed to write history {}: {error}", path.display()))
}

fn device_knowledge_key(record: &RunRecord) -> String {
    let cpu = record
        .cpu_cores
        .or_else(|| parse_prefixed_u64(&record.device_fingerprint, "cpu").map(|v| v as usize))
        .unwrap_or(0);
    let ram_mb = record
        .total_ram_mb
        .or_else(|| parse_prefixed_u64(&record.device_fingerprint, "-ram"));
    let backend = record_backend(record).unwrap_or_else(|| "unknown".to_string());
    let vram_mb = record_vram_mb(record).unwrap_or(0);
    let cpu_bucket = bucket_cpu_cores(cpu);
    let ram_bucket = bucket_ram_mb(ram_mb.unwrap_or(0));
    let vram_bucket = bucket_vram_mb(vram_mb);
    format!(
        "cpu{}-ram{}-gpu{}-vram{}",
        cpu_bucket, ram_bucket, backend, vram_bucket
    )
}

fn bucket_cpu_cores(cores: usize) -> usize {
    match cores {
        0 => 0,
        1..=4 => 4,
        5..=8 => 8,
        9..=12 => 12,
        13..=16 => 16,
        17..=24 => 24,
        _ => 32,
    }
}

fn bucket_ram_mb(ram_mb: u64) -> u64 {
    if ram_mb == 0 {
        0
    } else {
        // Round to 2GB buckets for better cross-run grouping.
        ram_mb.div_ceil(2_048) * 2_048
    }
}

fn bucket_vram_mb(vram_mb: u64) -> u64 {
    if vram_mb == 0 {
        0
    } else {
        // Round to nearest 1GB bucket.
        vram_mb.div_ceil(1_024) * 1_024
    }
}

fn parse_prefixed_u64(input: &str, prefix: &str) -> Option<u64> {
    let start = input.find(prefix)? + prefix.len();
    let digits = input[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.parse::<u64>().ok()
    }
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn sanitize_fingerprint_component(input: &str) -> String {
    input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

fn content_profile_hash(content: &ContentMap) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    content.input_kind.hash(&mut hasher);
    ((content.total_duration_secs * 10.0).round() as i64).hash(&mut hasher);
    ((content.speech_duration_secs * 10.0).round() as i64).hash(&mut hasher);
    ((content.avg_difficulty * 100.0).round() as i64).hash(&mut hasher);
    ((content.speaker_complexity_score * 100.0).round() as i64).hash(&mut hasher);
    ((content.energy_variance_score * 100.0).round() as i64).hash(&mut hasher);
    ((content.overlap_risk_score * 100.0).round() as i64).hash(&mut hasher);
    content.scene_count.hash(&mut hasher);
    for scene in content.scenes.iter().take(32) {
        ((scene.duration_secs * 10.0).round() as i64).hash(&mut hasher);
        ((scene.speech_density * 100.0).round() as i64).hash(&mut hasher);
        ((scene.difficulty * 100.0).round() as i64).hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

fn find_in_path(candidates: &[&str]) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        for name in candidates {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn display_ram(total_ram_mb: Option<u64>) -> String {
    total_ram_mb
        .map(|value| format!("{:.1}GB", value as f64 / 1024.0))
        .unwrap_or_else(|| "unknown".to_string())
}

fn display_duration(duration_secs: Option<f64>) -> String {
    duration_secs
        .map(|value| format!("{value:.1}s"))
        .unwrap_or_else(|| "unknown".to_string())
}

fn display_opt_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "auto".to_string())
}

fn display_disk_mbps(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.0}"))
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        assess_output_health, build_knowledge_snapshot, choose_mt_tuning,
        default_mt_batch_for_profile, default_mt_max_tokens_for_profile, duration_to_chunk_secs,
        parse_ffprobe_output, ExecutionPlan, GpuProbe, HardwareProbe, HistoryStore,
        PlanLookupQuery, RunRecord, WorkloadEstimate,
    };
    use crate::engine::pipeline::PipelineConfig;
    use crate::engine::transcribe::QualityProfile;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn sample_config() -> PipelineConfig {
        PipelineConfig {
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            offline: true,
            transcribe: true,
            whisper_bin: None,
            whisper_model: None,
            whisper_args: Vec::new(),
            skip_existing: false,
            vad: true,
            vad_threshold_db: -35.0,
            vad_min_silence: 0.35,
            vad_pad: 0.20,
            verify: false,
            verify_min_speech_overlap: 0.35,
            gpu: true,
            require_gpu: false,
            parallel: false,
            max_workers: 12,
            chunk_duration_secs: 300.0,
            force_phrase_table: false,
            mt_model: None,
            mt_batch_size: None,
            mt_max_batch_tokens: None,
            mt_oom_retries: None,
            mt_allow_cpu_fallback: true,
            auto_repair_sidecar: true,
            quality_profile: QualityProfile::Strict,
        }
    }

    fn temp_case_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after UNIX_EPOCH")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("sub_zero_doom_qlock_{name}_{stamp}"));
        fs::create_dir_all(&path).expect("temp directory should be creatable");
        path
    }

    #[test]
    fn heuristic_plan_scales_mt_for_low_vram_gpu() {
        let base = sample_config();
        let hardware = HardwareProbe {
            cpu_cores: 16,
            total_ram_mb: Some(32_768),
            disk_write_mbps: Some(450.0),
            gpu: Some(GpuProbe {
                backend: "cuda".to_string(),
                name: "RTX 3050".to_string(),
                vram_mb: Some(4_096),
                compute_capability: Some("8.6".to_string()),
            }),
        };
        let workload = WorkloadEstimate {
            input_kind: "matroska,webm".to_string(),
            is_video: true,
            duration_secs: Some(6_000.0),
            speech_duration_secs: Some(5_400.0),
            estimated_cues: 2_000,
            avg_difficulty: Some(0.45),
            speaker_complexity: Some(0.35),
            energy_variance: Some(0.25),
            scene_count: Some(32),
            detected_source_lang: "ja".to_string(),
            detected_confidence: 0.7,
            detection_mode: "hint".to_string(),
        };

        let plan = ExecutionPlan::heuristic(&base, &hardware, &workload);
        assert!(plan.parallel);
        assert!(plan.workers <= 12);
        assert_eq!(plan.chunk_duration_secs, 300.0);
        assert!(plan.mt_batch_size.expect("batch should be set") <= 16);
        assert!(plan.mt_max_batch_tokens.expect("max tokens should be set") <= 4096);
    }

    #[test]
    fn mt_tuning_is_backend_aware() {
        let cuda_hw = HardwareProbe {
            cpu_cores: 8,
            total_ram_mb: Some(16_384),
            disk_write_mbps: Some(800.0),
            gpu: Some(GpuProbe {
                backend: "cuda".to_string(),
                name: "RTX".to_string(),
                vram_mb: Some(8_192),
                compute_capability: Some("8.6".to_string()),
            }),
        };
        let metal_hw = HardwareProbe {
            cpu_cores: 8,
            total_ram_mb: Some(16_384),
            disk_write_mbps: Some(800.0),
            gpu: Some(GpuProbe {
                backend: "metal".to_string(),
                name: "Apple GPU".to_string(),
                vram_mb: Some(8_192),
                compute_capability: None,
            }),
        };

        let cuda = choose_mt_tuning(&cuda_hw, QualityProfile::Strict, true);
        let metal = choose_mt_tuning(&metal_hw, QualityProfile::Strict, true);
        assert!(metal.batch_size < cuda.batch_size);
        assert!(metal.max_batch_tokens < cuda.max_batch_tokens);
        assert!(metal.oom_retries >= cuda.oom_retries);
    }

    #[test]
    fn history_prefers_fastest_successful_plan() {
        let plan_fast = ExecutionPlan {
            parallel: true,
            workers: 8,
            chunk_duration_secs: 240.0,
            mt_batch_size: Some(16),
            mt_max_batch_tokens: Some(4096),
            mt_oom_retries: Some(2),
            mt_allow_cpu_fallback: true,
        };
        let plan_slow = ExecutionPlan {
            parallel: true,
            workers: 4,
            chunk_duration_secs: 300.0,
            mt_batch_size: Some(8),
            mt_max_batch_tokens: Some(2048),
            mt_oom_retries: Some(4),
            mt_allow_cpu_fallback: true,
        };

        let mut history = HistoryStore::default();
        history.push_record(RunRecord {
            timestamp_epoch_secs: 1,
            device_fingerprint: "dev-1".to_string(),
            content_profile_hash: "content-a".to_string(),
            gpu_backend: Some("cuda".to_string()),
            gpu_vram_mb: Some(8_192),
            cpu_cores: Some(8),
            total_ram_mb: Some(16_384),
            disk_write_mbps: Some(400.0),
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            quality_profile: "strict".to_string(),
            input_kind: "srt".to_string(),
            audio_duration_secs: Some(120.0),
            speech_duration_secs: Some(115.0),
            estimated_cues: 40,
            avg_difficulty: Some(0.3),
            speaker_complexity: Some(0.1),
            energy_variance: Some(0.1),
            scene_count: Some(1),
            plan: plan_slow.clone(),
            success: true,
            elapsed_secs: 20.0,
            output_cues: Some(40),
            output_non_empty_ratio: Some(1.0),
            output_top_line_ratio: Some(0.10),
            error: None,
        });
        history.push_record(RunRecord {
            timestamp_epoch_secs: 2,
            device_fingerprint: "dev-1".to_string(),
            content_profile_hash: "content-a".to_string(),
            gpu_backend: Some("cuda".to_string()),
            gpu_vram_mb: Some(8_192),
            cpu_cores: Some(8),
            total_ram_mb: Some(16_384),
            disk_write_mbps: Some(400.0),
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            quality_profile: "strict".to_string(),
            input_kind: "srt".to_string(),
            audio_duration_secs: Some(120.0),
            speech_duration_secs: Some(115.0),
            estimated_cues: 40,
            avg_difficulty: Some(0.3),
            speaker_complexity: Some(0.1),
            energy_variance: Some(0.1),
            scene_count: Some(1),
            plan: plan_fast.clone(),
            success: true,
            elapsed_secs: 8.0,
            output_cues: Some(40),
            output_non_empty_ratio: Some(0.99),
            output_top_line_ratio: Some(0.08),
            error: None,
        });

        let selected = history.best_plan_exact(
            "dev-1",
            PlanLookupQuery {
                source_lang: "ja",
                target_lang: "en",
                profile: QualityProfile::Strict,
                input_kind: "srt",
                content_profile_hash: "content-a",
            },
        );
        assert_eq!(selected.expect("plan should be found"), plan_fast);
    }

    #[test]
    fn history_similar_plan_fallback_uses_compatible_hardware() {
        let plan_gpu = ExecutionPlan {
            parallel: true,
            workers: 6,
            chunk_duration_secs: 240.0,
            mt_batch_size: Some(12),
            mt_max_batch_tokens: Some(3072),
            mt_oom_retries: Some(3),
            mt_allow_cpu_fallback: true,
        };
        let plan_cpu = ExecutionPlan {
            parallel: false,
            workers: 4,
            chunk_duration_secs: 300.0,
            mt_batch_size: Some(8),
            mt_max_batch_tokens: Some(2048),
            mt_oom_retries: Some(4),
            mt_allow_cpu_fallback: true,
        };
        let mut history = HistoryStore::default();
        history.push_record(RunRecord {
            timestamp_epoch_secs: 3,
            device_fingerprint: "cpu8-ram32768-gpucuda-rtx-3060-vram8192-disk420".to_string(),
            content_profile_hash: "content-old".to_string(),
            gpu_backend: Some("cuda".to_string()),
            gpu_vram_mb: Some(8_192),
            cpu_cores: Some(8),
            total_ram_mb: Some(32_768),
            disk_write_mbps: Some(420.0),
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            quality_profile: "strict".to_string(),
            input_kind: "srt".to_string(),
            audio_duration_secs: Some(600.0),
            speech_duration_secs: Some(540.0),
            estimated_cues: 180,
            avg_difficulty: Some(0.45),
            speaker_complexity: Some(0.25),
            energy_variance: Some(0.20),
            scene_count: Some(12),
            plan: plan_gpu.clone(),
            success: true,
            elapsed_secs: 42.0,
            output_cues: Some(180),
            output_non_empty_ratio: Some(0.98),
            output_top_line_ratio: Some(0.05),
            error: None,
        });
        history.push_record(RunRecord {
            timestamp_epoch_secs: 4,
            device_fingerprint: "cpu8-ram32768-gpunone-vram0-disk390".to_string(),
            content_profile_hash: "content-old".to_string(),
            gpu_backend: Some("none".to_string()),
            gpu_vram_mb: Some(0),
            cpu_cores: Some(8),
            total_ram_mb: Some(32_768),
            disk_write_mbps: Some(390.0),
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            quality_profile: "strict".to_string(),
            input_kind: "srt".to_string(),
            audio_duration_secs: Some(600.0),
            speech_duration_secs: Some(540.0),
            estimated_cues: 180,
            avg_difficulty: Some(0.45),
            speaker_complexity: Some(0.25),
            energy_variance: Some(0.20),
            scene_count: Some(12),
            plan: plan_cpu.clone(),
            success: true,
            elapsed_secs: 55.0,
            output_cues: Some(180),
            output_non_empty_ratio: Some(0.98),
            output_top_line_ratio: Some(0.05),
            error: None,
        });

        let hardware = HardwareProbe {
            cpu_cores: 8,
            total_ram_mb: Some(32_768),
            disk_write_mbps: Some(410.0),
            gpu: Some(GpuProbe {
                backend: "cuda".to_string(),
                name: "RTX 3070".to_string(),
                vram_mb: Some(8_192),
                compute_capability: Some("8.6".to_string()),
            }),
        };
        let workload = WorkloadEstimate {
            input_kind: "srt".to_string(),
            is_video: false,
            duration_secs: Some(650.0),
            speech_duration_secs: Some(590.0),
            estimated_cues: 190,
            avg_difficulty: Some(0.48),
            speaker_complexity: Some(0.30),
            energy_variance: Some(0.22),
            scene_count: Some(14),
            detected_source_lang: "ja".to_string(),
            detected_confidence: 0.9,
            detection_mode: "declared".to_string(),
        };

        let selected = history.best_plan_similar(
            &hardware,
            &workload,
            PlanLookupQuery {
                source_lang: "ja",
                target_lang: "en",
                profile: QualityProfile::Strict,
                input_kind: "srt",
                content_profile_hash: "content-new",
            },
        );
        assert_eq!(
            selected.expect("similar fallback should select plan"),
            plan_gpu
        );
    }

    #[test]
    fn parse_ffprobe_output_reads_duration_and_format() {
        let parsed = parse_ffprobe_output("format_name=matroska,webm\nduration=6262.123\n")
            .expect("ffprobe output should parse");
        assert_eq!(parsed.format_name.as_deref(), Some("matroska,webm"));
        let duration = parsed.duration_secs.expect("duration should exist");
        assert!((duration - 6262.123).abs() < 0.001);
    }

    #[test]
    fn duration_to_chunk_secs_scales_with_length() {
        assert_eq!(duration_to_chunk_secs(Some(900.0), None), 180.0);
        assert_eq!(duration_to_chunk_secs(Some(2_500.0), None), 240.0);
        assert_eq!(duration_to_chunk_secs(Some(4_000.0), None), 300.0);
        assert_eq!(duration_to_chunk_secs(Some(9_000.0), None), 360.0);
        assert_eq!(duration_to_chunk_secs(None, None), 300.0);
        assert_eq!(duration_to_chunk_secs(Some(4_000.0), Some(0.8)), 225.0);
        assert_eq!(duration_to_chunk_secs(Some(4_000.0), Some(0.2)), 345.0);
    }

    #[test]
    fn output_health_flags_repetition() {
        let dir = temp_case_dir("output_health");
        let srt = dir.join("sample.en.srt");
        let mut body = String::new();
        for idx in 1..=10usize {
            body.push_str(&format!(
                "{idx}\n00:00:{:02},000 --> 00:00:{:02},500\nrepeat line\n\n",
                idx,
                idx + 1
            ));
        }
        fs::write(&srt, body).expect("test SRT should be writable");
        let health = assess_output_health(&srt).expect("health should parse");
        assert_eq!(health.cue_count, 10);
        assert!(health.non_empty_ratio >= 0.99);
        assert!(health.top_line_ratio >= 0.90);
    }

    #[test]
    fn strict_defaults_are_stable() {
        assert_eq!(default_mt_batch_for_profile(QualityProfile::Strict), 16);
        assert_eq!(
            default_mt_max_tokens_for_profile(QualityProfile::Strict),
            4096
        );
    }

    #[test]
    fn knowledge_snapshot_aggregates_device_and_language_stats() {
        let plan = ExecutionPlan {
            parallel: true,
            workers: 6,
            chunk_duration_secs: 240.0,
            mt_batch_size: Some(12),
            mt_max_batch_tokens: Some(3072),
            mt_oom_retries: Some(3),
            mt_allow_cpu_fallback: true,
        };
        let mut history = HistoryStore::default();
        history.push_record(RunRecord {
            timestamp_epoch_secs: 10,
            device_fingerprint: "gpu-a".to_string(),
            content_profile_hash: "hash-1".to_string(),
            gpu_backend: Some("cuda".to_string()),
            gpu_vram_mb: Some(12_288),
            cpu_cores: Some(8),
            total_ram_mb: Some(32_768),
            disk_write_mbps: Some(450.0),
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            quality_profile: "strict".to_string(),
            input_kind: "matroska,webm".to_string(),
            audio_duration_secs: Some(600.0),
            speech_duration_secs: Some(500.0),
            estimated_cues: 100,
            avg_difficulty: Some(0.4),
            speaker_complexity: Some(0.2),
            energy_variance: Some(0.2),
            scene_count: Some(6),
            plan: plan.clone(),
            success: true,
            elapsed_secs: 40.0,
            output_cues: Some(100),
            output_non_empty_ratio: Some(0.98),
            output_top_line_ratio: Some(0.05),
            error: None,
        });
        history.push_record(RunRecord {
            timestamp_epoch_secs: 11,
            device_fingerprint: "gpu-a".to_string(),
            content_profile_hash: "hash-1".to_string(),
            gpu_backend: Some("cuda".to_string()),
            gpu_vram_mb: Some(12_288),
            cpu_cores: Some(8),
            total_ram_mb: Some(32_768),
            disk_write_mbps: Some(450.0),
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            quality_profile: "strict".to_string(),
            input_kind: "matroska,webm".to_string(),
            audio_duration_secs: Some(1200.0),
            speech_duration_secs: Some(1000.0),
            estimated_cues: 180,
            avg_difficulty: Some(0.5),
            speaker_complexity: Some(0.3),
            energy_variance: Some(0.3),
            scene_count: Some(10),
            plan: plan.clone(),
            success: false,
            elapsed_secs: 85.0,
            output_cues: None,
            output_non_empty_ratio: None,
            output_top_line_ratio: None,
            error: Some("oom".to_string()),
        });

        let snapshot = build_knowledge_snapshot(&history);
        let device = snapshot
            .devices
            .get("cpu8-ram32768-gpucuda-vram12288")
            .expect("device knowledge should exist");
        assert_eq!(device.runs, 2);
        assert!(device.success_rate > 0.4 && device.success_rate < 0.6);

        let pair = snapshot
            .language_pairs
            .get("ja->en")
            .expect("pair knowledge should exist");
        assert_eq!(pair.runs, 2);
        assert!(pair.avg_elapsed_per_audio_hour_secs.is_some());
    }
}
