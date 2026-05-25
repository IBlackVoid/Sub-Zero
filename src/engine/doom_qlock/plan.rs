use super::probe::{probe_media_format, HardwareProbe};
use super::util::{duration_from_cues, is_srt_path};
use crate::engine::deep_scan::ContentMap;
use crate::engine::pipeline::PipelineConfig;
use crate::engine::srt::parse_srt_file;
use crate::engine::transcribe::QualityProfile;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub(super) enum PlanOrigin {
    Heuristic,
    LearnedExact,
    LearnedSimilar,
}

impl PlanOrigin {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Heuristic => "heuristic",
            Self::LearnedExact => "history-exact",
            Self::LearnedSimilar => "history-similar",
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct WorkloadEstimate {
    pub(super) input_kind: String,
    pub(super) is_video: bool,
    pub(super) duration_secs: Option<f64>,
    pub(super) speech_duration_secs: Option<f64>,
    pub(super) estimated_cues: usize,
    pub(super) avg_difficulty: Option<f64>,
    pub(super) speaker_complexity: Option<f64>,
    pub(super) energy_variance: Option<f64>,
    pub(super) scene_count: Option<usize>,
    pub(super) detected_source_lang: String,
    pub(super) detected_confidence: f64,
    pub(super) detection_mode: String,
}

impl WorkloadEstimate {
    pub(super) fn probe(
        input: &Path,
        source_lang_hint: &str,
        deep_scan: Option<&ContentMap>,
    ) -> Result<Self, String> {
        if is_srt_path(input) {
            let cues = parse_srt_file(input).map_err(|error| error.to_string())?;
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
pub(super) struct ExecutionPlan {
    pub(super) parallel: bool,
    pub(super) workers: usize,
    pub(super) chunk_duration_secs: f64,
    pub(super) mt_batch_size: Option<usize>,
    pub(super) mt_max_batch_tokens: Option<usize>,
    pub(super) mt_oom_retries: Option<usize>,
    pub(super) mt_allow_cpu_fallback: bool,
}

impl ExecutionPlan {
    pub(super) fn from_config(config: &PipelineConfig) -> Self {
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

    pub(super) fn heuristic(
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
                let profile = base_config.quality_profile;
                let batch = plan
                    .mt_batch_size
                    .unwrap_or_else(|| default_mt_batch_for_profile(profile));
                let tokens = plan
                    .mt_max_batch_tokens
                    .unwrap_or_else(|| default_mt_max_tokens_for_profile(profile));
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

    pub(super) fn validate_and_adjust(
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
                .unwrap_or_else(|| default_mt_batch_for_profile(profile));
            let current_tokens = self
                .mt_max_batch_tokens
                .unwrap_or_else(|| default_mt_max_tokens_for_profile(profile));

            if current_batch <= 4 && current_tokens <= 1_024 {
                break;
            }

            self.mt_batch_size = Some((current_batch.saturating_mul(3) / 4).max(4));
            self.mt_max_batch_tokens = Some((current_tokens.saturating_mul(3) / 4).max(1_024));
            let retries = self
                .mt_oom_retries
                .unwrap_or_else(|| default_mt_oom_retries(profile));
            self.mt_oom_retries = Some((retries + 1).min(8));
        }
    }

    fn estimate_mt_vram_mb(&self, profile: QualityProfile) -> u64 {
        let batch = self
            .mt_batch_size
            .unwrap_or_else(|| default_mt_batch_for_profile(profile)) as u64;
        let tokens =
            self.mt_max_batch_tokens
                .unwrap_or_else(|| default_mt_max_tokens_for_profile(profile)) as u64;
        2_048 + (batch * 180) + (tokens / 2)
    }

    pub(super) fn apply_to_config(&self, base_config: &PipelineConfig) -> PipelineConfig {
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
pub(super) struct MtTuning {
    pub(super) batch_size: usize,
    pub(super) max_batch_tokens: usize,
    pub(super) oom_retries: usize,
}

pub(super) fn choose_mt_tuning(
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

pub(super) fn duration_to_chunk_secs(
    duration_secs: Option<f64>,
    avg_difficulty: Option<f64>,
) -> f64 {
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

pub(super) fn default_mt_batch_for_profile(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 32,
        QualityProfile::Balanced => 24,
        QualityProfile::Strict => 16,
    }
}

pub(super) fn default_mt_max_tokens_for_profile(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 8_192,
        QualityProfile::Balanced => 6_144,
        QualityProfile::Strict => 4_096,
    }
}

pub(super) fn default_mt_oom_retries(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 1,
        QualityProfile::Balanced => 2,
        QualityProfile::Strict => 3,
    }
}
