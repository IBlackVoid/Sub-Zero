use super::plan::{ExecutionPlan, WorkloadEstimate};
use super::probe::HardwareProbe;
use super::util::parse_prefixed_u64;
use crate::engine::transcribe::QualityProfile;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub(super) const HISTORY_VERSION: u32 = 1;
pub(super) const HISTORY_MAX_RECORDS: usize = 400;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct RunRecord {
    pub(super) timestamp_epoch_secs: u64,
    pub(super) device_fingerprint: String,
    #[serde(default = "default_content_profile_hash")]
    pub(super) content_profile_hash: String,
    #[serde(default)]
    pub(super) gpu_backend: Option<String>,
    #[serde(default)]
    pub(super) gpu_vram_mb: Option<u64>,
    #[serde(default)]
    pub(super) cpu_cores: Option<usize>,
    #[serde(default)]
    pub(super) total_ram_mb: Option<u64>,
    #[serde(default)]
    pub(super) disk_write_mbps: Option<f64>,
    pub(super) source_lang: String,
    pub(super) target_lang: String,
    pub(super) quality_profile: String,
    pub(super) input_kind: String,
    pub(super) audio_duration_secs: Option<f64>,
    pub(super) speech_duration_secs: Option<f64>,
    pub(super) estimated_cues: usize,
    pub(super) avg_difficulty: Option<f64>,
    pub(super) speaker_complexity: Option<f64>,
    pub(super) energy_variance: Option<f64>,
    pub(super) scene_count: Option<usize>,
    pub(super) plan: ExecutionPlan,
    pub(super) success: bool,
    pub(super) elapsed_secs: f64,
    pub(super) output_cues: Option<usize>,
    pub(super) output_non_empty_ratio: Option<f64>,
    pub(super) output_top_line_ratio: Option<f64>,
    pub(super) error: Option<String>,
}

pub(super) fn default_content_profile_hash() -> String {
    "none".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct HistoryStore {
    pub(super) version: u32,
    pub(super) records: Vec<RunRecord>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PlanLookupQuery<'a> {
    pub(super) source_lang: &'a str,
    pub(super) target_lang: &'a str,
    pub(super) profile: QualityProfile,
    pub(super) input_kind: &'a str,
    pub(super) content_profile_hash: &'a str,
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
    pub(super) fn push_record(&mut self, record: RunRecord) {
        self.records.push(record);
        if self.records.len() > HISTORY_MAX_RECORDS {
            let overflow = self.records.len() - HISTORY_MAX_RECORDS;
            self.records.drain(0..overflow);
        }
    }

    pub(super) fn best_plan_exact(
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

    pub(super) fn best_plan_similar(
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

pub(super) fn elapsed_per_audio_hour(record: &RunRecord) -> f64 {
    if let Some(duration_secs) = record.audio_duration_secs {
        if duration_secs.is_finite() && duration_secs > 0.0 {
            return record.elapsed_secs / (duration_secs / 3600.0);
        }
    }
    record.elapsed_secs
}

pub(super) fn record_backend(record: &RunRecord) -> Option<String> {
    record
        .gpu_backend
        .as_ref()
        .map(|value| value.to_ascii_lowercase())
        .or_else(|| infer_backend_from_fingerprint(&record.device_fingerprint))
}

pub(super) fn record_vram_mb(record: &RunRecord) -> Option<u64> {
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

pub(super) fn default_history_path() -> PathBuf {
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

pub(super) fn load_history(path: &Path) -> Result<HistoryStore, String> {
    if !path.is_file() {
        return Ok(HistoryStore::default());
    }
    let content = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let mut history: HistoryStore =
        serde_json::from_str(&content).map_err(|error| format!("invalid history JSON: {error}"))?;
    if history.version == 0 {
        history.version = HISTORY_VERSION;
    }
    Ok(history)
}

pub(super) fn save_history(path: &Path, history: &HistoryStore) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create history directory {}: {error}",
                parent.display()
            )
        })?;
    }
    let serialized = serde_json::to_string_pretty(history)
        .map_err(|error| format!("failed to serialize history: {error}"))?;
    std::fs::write(path, serialized)
        .map_err(|error| format!("failed to write history {}: {error}", path.display()))
}
