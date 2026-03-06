// Parallel Transcription Worker Pool — Theorem 2 of SRT².
//
// Spawns N whisper processes in parallel, one per audio chunk.
// Each worker produces a partial SRT which the stitcher later merges.

use crate::engine::chunker::AudioChunk;
use crate::engine::srt::{parse_srt, SubtitleCue};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Result from a single chunk transcription.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ChunkTranscription {
    /// Which chunk this came from.
    pub chunk_index: usize,
    /// Absolute start offset of the chunk in the original audio (seconds).
    pub offset_secs: f64,
    /// How much overlap exists at the start of this chunk.
    pub overlap_before: f64,
    /// How much overlap exists at the end of this chunk.
    pub overlap_after: f64,
    /// The raw cues parsed from whisper output (timestamps relative to chunk).
    pub cues: Vec<SubtitleCue>,
}

use crate::engine::transcribe::{QualityProfile, TranscribeConfig, Transcriber};

/// Transcribe all chunks in parallel and return chunk-level transcriptions.
pub fn parallel_transcribe(
    chunks: &[AudioChunk],
    config: &TranscribeConfig,
    max_workers: usize,
    checkpoint_path: Option<PathBuf>,
) -> Result<Vec<ChunkTranscription>, String> {
    let monitor = if let Some(path) = checkpoint_path {
        Some(Arc::new(ChunkRunMonitor::new(path, chunks, config)?))
    } else {
        None
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_workers)
        .build()
        .map_err(|e| format!("failed to build thread pool: {e}"))?;

    let monitor_for_workers = monitor.clone();
    let results: Vec<Result<ChunkTranscription, String>> = pool.install(|| {
        chunks
            .par_iter()
            .map_init(
                || {
                    (
                        Transcriber::new(config.clone()).and_then(|opt| {
                            opt.ok_or_else(|| "transcriber disabled explicitly".to_string())
                        }),
                        monitor_for_workers.clone(),
                    )
                },
                |state, chunk| match &state.0 {
                    Ok(transcriber) => {
                        transcribe_single_chunk(chunk, transcriber, config, state.1.as_deref())
                    }
                    Err(error) => Err(format!("failed to initialize worker transcriber: {error}")),
                },
            )
            .collect()
    });

    // Collect results, fail on any error.
    let mut transcriptions = Vec::with_capacity(results.len());
    for result in results {
        transcriptions.push(result?);
    }
    transcriptions.sort_by_key(|t| t.chunk_index);
    Ok(transcriptions)
}

/// Transcribe a single chunk using the generic Transcriber (supports Cpp and Python).
fn transcribe_single_chunk(
    chunk: &AudioChunk,
    transcriber: &Transcriber,
    config: &TranscribeConfig,
    monitor: Option<&ChunkRunMonitor>,
) -> Result<ChunkTranscription, String> {
    let policy = monitor.map(|m| m.policy_snapshot()).unwrap_or_default();
    let srt_path = chunk.wav_path.with_extension("srt");
    let max_attempts = config.quality_profile.max_chunk_retries() + 1 + policy.extra_retry_budget;
    let mut last_issue = String::from("unknown quality gate failure");
    let predicted_secs = estimate_chunk_predicted_secs(chunk, config);
    let profile_timeout_floor = match config.quality_profile {
        QualityProfile::Fast => 120.0,
        QualityProfile::Balanced => 180.0,
        QualityProfile::Strict => 300.0,
    };
    let timeout_secs = (predicted_secs * 10.0 * policy.timeout_scale)
        .clamp(120.0, 3600.0)
        .max(profile_timeout_floor);

    for attempt in 0..max_attempts {
        let mut from_cache = false;
        let chunk_start = Instant::now();
        let is_strict_pass = policy.strict_first_pass || attempt > 0;
        let cues_result = if attempt == 0 && srt_path.is_file() {
            let srt_content = std::fs::read_to_string(&srt_path)
                .map_err(|e| format!("chunk {} read cached SRT: {e}", chunk.index));
            match srt_content {
                Ok(srt_content) => match parse_srt(&srt_content) {
                    Ok(parsed) => {
                        from_cache = true;
                        Ok(parsed)
                    }
                    Err(_) => {
                        if let Err(error) = std::fs::remove_file(&srt_path) {
                            Err(format!(
                                "chunk {} remove invalid cached SRT: {error}",
                                chunk.index
                            ))
                        } else if is_strict_pass {
                            let retry_config = strict_retry_config(config, attempt.max(1));
                            let retry_transcriber = Transcriber::new(retry_config)?
                                .ok_or_else(|| "transcriber disabled explicitly".to_string())?;
                            transcribe_chunk_once(chunk, &retry_transcriber, timeout_secs)
                        } else {
                            transcribe_chunk_once(chunk, transcriber, timeout_secs)
                        }
                    }
                },
                Err(error) => Err(error),
            }
        } else if is_strict_pass {
            let retry_config = strict_retry_config(config, attempt.max(1));
            let retry_transcriber = Transcriber::new(retry_config)?
                .ok_or_else(|| "transcriber disabled explicitly".to_string())?;
            transcribe_chunk_once(chunk, &retry_transcriber, timeout_secs)
        } else {
            transcribe_chunk_once(chunk, transcriber, timeout_secs)
        };

        let cues = match cues_result {
            Ok(cues) => cues,
            Err(error) => {
                last_issue = error;
                if attempt + 1 < max_attempts {
                    eprintln!(
                        "warning: chunk {} attempt {}/{} failed ({}); retrying...",
                        chunk.index,
                        attempt + 1,
                        max_attempts,
                        last_issue
                    );
                    if srt_path.is_file() {
                        let _ = std::fs::remove_file(&srt_path);
                    }
                    continue;
                }
                break;
            }
        };
        let actual_secs = if from_cache {
            0.0
        } else {
            chunk_start.elapsed().as_secs_f64()
        };

        let health = assess_chunk_health(&cues)?;
        if !health.is_pathological(config.quality_profile) {
            if let Some(monitor) = monitor {
                monitor.record_success(
                    chunk.index,
                    predicted_secs,
                    actual_secs,
                    health.score(config.quality_profile),
                    attempt + 1,
                    from_cache,
                );
            }
            return Ok(ChunkTranscription {
                chunk_index: chunk.index,
                offset_secs: chunk.start_sec,
                overlap_before: chunk.overlap_before,
                overlap_after: chunk.overlap_after,
                cues,
            });
        }

        last_issue = format!(
            "chunk {} failed quality gate on attempt {}/{} ({})",
            chunk.index,
            attempt + 1,
            max_attempts,
            health.summary()
        );
        if attempt + 1 < max_attempts {
            eprintln!("warning: {last_issue}; retrying with safer settings...");
            if srt_path.is_file() {
                std::fs::remove_file(&srt_path).map_err(|e| {
                    format!(
                        "chunk {} remove pathological SRT before retry: {e}",
                        chunk.index
                    )
                })?;
            }
        }
    }

    let timeout_failure = last_issue.to_ascii_lowercase().contains("timed out");
    if let Some(monitor) = monitor {
        monitor.record_failure(chunk.index, predicted_secs, &last_issue);
    }
    if timeout_failure && config.quality_profile != QualityProfile::Strict {
        eprintln!(
            "warning: chunk {} timed out after retries; continuing with empty output because profile={}",
            chunk.index,
            config.quality_profile.as_str()
        );
        return Ok(ChunkTranscription {
            chunk_index: chunk.index,
            offset_secs: chunk.start_sec,
            overlap_before: chunk.overlap_before,
            overlap_after: chunk.overlap_after,
            cues: Vec::new(),
        });
    }
    Err(last_issue)
}

fn transcribe_chunk_once(
    chunk: &AudioChunk,
    transcriber: &Transcriber,
    timeout_secs: f64,
) -> Result<Vec<SubtitleCue>, String> {
    let srt_path = transcriber
        .transcribe_wav_to_srt_with_timeout(&chunk.wav_path, timeout_secs)
        .map_err(|e| format!("chunk {} transcription failed: {e}", chunk.index))?;
    let srt_content = std::fs::read_to_string(&srt_path)
        .map_err(|e| format!("chunk {} read SRT: {e}", chunk.index))?;
    parse_srt(&srt_content).map_err(|e| format!("chunk {} parse SRT: {e}", chunk.index))
}

fn strict_retry_config(base: &TranscribeConfig, attempt: usize) -> TranscribeConfig {
    let mut retry = base.clone();
    retry.quality_profile = QualityProfile::Strict;
    retry.vad = false;
    if attempt >= 2 && !has_temperature_override(&retry.whisper_args) {
        retry.whisper_args.push("--temperature".to_string());
        retry.whisper_args.push("0".to_string());
    }
    retry
}

fn has_temperature_override(args: &[String]) -> bool {
    args.iter()
        .any(|arg| arg == "--temperature" || arg.starts_with("--temperature="))
}

#[derive(Debug, Clone, Copy)]
struct ChunkHealth {
    cue_count: usize,
    top_line_ratio: f64,
    overlap_ratio: f64,
    non_empty_ratio: f64,
}

impl ChunkHealth {
    fn is_pathological(&self, profile: QualityProfile) -> bool {
        let thresholds = ChunkHealthThresholds::for_profile(profile);
        if self.cue_count < thresholds.min_cues {
            return false;
        }
        self.top_line_ratio >= thresholds.max_top_line_ratio
            || self.overlap_ratio >= thresholds.max_overlap_ratio
            || self.non_empty_ratio < thresholds.min_non_empty_ratio
    }

    fn summary(&self) -> String {
        format!(
            "cues={} top_line_ratio={:.2}% overlap_ratio={:.2}% non_empty_ratio={:.2}%",
            self.cue_count,
            self.top_line_ratio * 100.0,
            self.overlap_ratio * 100.0,
            self.non_empty_ratio * 100.0
        )
    }

    fn score(&self, profile: QualityProfile) -> f64 {
        let thresholds = ChunkHealthThresholds::for_profile(profile);
        if self.cue_count < thresholds.min_cues {
            return 1.0;
        }
        let top_component =
            (1.0 - (self.top_line_ratio / thresholds.max_top_line_ratio)).clamp(0.0, 1.0);
        let overlap_component =
            (1.0 - (self.overlap_ratio / thresholds.max_overlap_ratio)).clamp(0.0, 1.0);
        let non_empty_component =
            (self.non_empty_ratio / thresholds.min_non_empty_ratio).clamp(0.0, 1.0);
        (top_component + overlap_component + non_empty_component) / 3.0
    }
}

#[derive(Debug, Clone, Copy)]
struct ChunkHealthThresholds {
    min_cues: usize,
    max_top_line_ratio: f64,
    max_overlap_ratio: f64,
    min_non_empty_ratio: f64,
}

impl ChunkHealthThresholds {
    fn for_profile(profile: QualityProfile) -> Self {
        match profile {
            QualityProfile::Fast => Self {
                min_cues: 120,
                max_top_line_ratio: 0.75,
                max_overlap_ratio: 0.60,
                min_non_empty_ratio: 0.60,
            },
            QualityProfile::Balanced => Self {
                min_cues: 60,
                max_top_line_ratio: 0.60,
                max_overlap_ratio: 0.45,
                min_non_empty_ratio: 0.70,
            },
            QualityProfile::Strict => Self {
                min_cues: 30,
                max_top_line_ratio: 0.45,
                max_overlap_ratio: 0.35,
                min_non_empty_ratio: 0.80,
            },
        }
    }
}

fn assess_chunk_health(cues: &[SubtitleCue]) -> Result<ChunkHealth, String> {
    if cues.is_empty() {
        return Ok(ChunkHealth {
            cue_count: 0,
            top_line_ratio: 0.0,
            overlap_ratio: 0.0,
            non_empty_ratio: 0.0,
        });
    }

    let mut freq = std::collections::HashMap::<String, usize>::new();
    let mut non_empty = 0usize;
    for cue in cues {
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
    let top_line_ratio = (top_count as f64) / (cues.len() as f64);
    let non_empty_ratio = (non_empty as f64) / (cues.len() as f64);

    let mut overlaps = 0usize;
    let mut parsed = 0usize;
    let mut prev_end = 0.0f64;
    for cue in cues {
        let (start, end) = parse_srt_timing_line(&cue.timing)?;
        if parsed > 0 && start < prev_end {
            overlaps += 1;
        }
        parsed += 1;
        prev_end = prev_end.max(end);
    }
    let overlap_ratio = if parsed > 1 {
        (overlaps as f64) / ((parsed - 1) as f64)
    } else {
        0.0
    };

    Ok(ChunkHealth {
        cue_count: cues.len(),
        top_line_ratio,
        overlap_ratio,
        non_empty_ratio,
    })
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
    let h = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let m = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let s = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let ms = ms
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    Ok((h as f64) * 3600.0 + (m as f64) * 60.0 + (s as f64) + (ms as f64) / 1000.0)
}

fn estimate_chunk_predicted_secs(chunk: &AudioChunk, config: &TranscribeConfig) -> f64 {
    let chunk_duration = (chunk.end_sec - chunk.start_sec).max(1.0);
    let realtime_factor = if config.gpu {
        match config.quality_profile {
            QualityProfile::Fast => 22.0,
            QualityProfile::Balanced => 16.0,
            QualityProfile::Strict => 10.0,
        }
    } else {
        match config.quality_profile {
            QualityProfile::Fast => 1.4,
            QualityProfile::Balanced => 1.0,
            QualityProfile::Strict => 0.8,
        }
    };
    (chunk_duration / realtime_factor).max(8.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkCheckpointFile {
    version: u32,
    total_chunks: usize,
    completed: Vec<ChunkCheckpointRecord>,
    failed: Vec<ChunkFailureRecord>,
}

impl Default for ChunkCheckpointFile {
    fn default() -> Self {
        Self {
            version: 1,
            total_chunks: 0,
            completed: Vec::new(),
            failed: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkCheckpointRecord {
    chunk_index: usize,
    predicted_secs: f64,
    actual_secs: f64,
    quality_score: f64,
    attempts: usize,
    from_cache: bool,
    completed_epoch_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkFailureRecord {
    chunk_index: usize,
    predicted_secs: f64,
    reason: String,
    failed_epoch_secs: u64,
}

struct ChunkRunMonitor {
    path: PathBuf,
    total_chunks: usize,
    total_predicted_secs: f64,
    quality_floor: f64,
    state: Mutex<ChunkCheckpointFile>,
    policy: Mutex<AdaptivePolicy>,
}

#[derive(Debug, Clone, Copy)]
struct AdaptivePolicy {
    timeout_scale: f64,
    extra_retry_budget: usize,
    strict_first_pass: bool,
}

impl Default for AdaptivePolicy {
    fn default() -> Self {
        Self {
            timeout_scale: 1.0,
            extra_retry_budget: 0,
            strict_first_pass: false,
        }
    }
}

impl ChunkRunMonitor {
    fn new(
        path: PathBuf,
        chunks: &[AudioChunk],
        config: &TranscribeConfig,
    ) -> Result<Self, String> {
        let mut state = if path.is_file() {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| format!("checkpoint read {}: {e}", path.display()))?;
            serde_json::from_str::<ChunkCheckpointFile>(&content)
                .map_err(|e| format!("checkpoint parse {}: {e}", path.display()))?
        } else {
            ChunkCheckpointFile::default()
        };
        state.total_chunks = chunks.len();

        if !state.completed.is_empty() {
            eprintln!(
                "ibvoid-doom-qlock: found checkpoint with {}/{} completed chunks (resume cache active)",
                state.completed.len(),
                chunks.len()
            );
        }

        let total_predicted_secs = chunks
            .iter()
            .map(|chunk| estimate_chunk_predicted_secs(chunk, config))
            .sum::<f64>();
        let quality_floor = match config.quality_profile {
            QualityProfile::Fast => 0.55,
            QualityProfile::Balanced => 0.65,
            QualityProfile::Strict => 0.75,
        };

        Ok(Self {
            path,
            total_chunks: chunks.len(),
            total_predicted_secs,
            quality_floor,
            state: Mutex::new(state),
            policy: Mutex::new(AdaptivePolicy::default()),
        })
    }

    fn policy_snapshot(&self) -> AdaptivePolicy {
        self.policy.lock().map(|policy| *policy).unwrap_or_default()
    }

    fn record_success(
        &self,
        chunk_index: usize,
        predicted_secs: f64,
        actual_secs: f64,
        quality_score: f64,
        attempts: usize,
        from_cache: bool,
    ) {
        let mut state = match self.state.lock() {
            Ok(lock) => lock,
            Err(_) => return,
        };

        state
            .completed
            .retain(|entry| entry.chunk_index != chunk_index);
        state.completed.push(ChunkCheckpointRecord {
            chunk_index,
            predicted_secs,
            actual_secs,
            quality_score,
            attempts,
            from_cache,
            completed_epoch_secs: now_epoch_secs(),
        });
        state.completed.sort_by_key(|entry| entry.chunk_index);
        state
            .failed
            .retain(|entry| entry.chunk_index != chunk_index);

        let completed = state.completed.len();
        let actual_sum = state
            .completed
            .iter()
            .map(|entry| entry.actual_secs)
            .sum::<f64>();
        let predicted_sum = state
            .completed
            .iter()
            .map(|entry| entry.predicted_secs.max(0.001))
            .sum::<f64>();
        let remaining_predicted = (self.total_predicted_secs - predicted_sum).max(0.0);
        let eta_secs = if predicted_sum > 0.0 {
            let ratio = (actual_sum / predicted_sum).max(0.05);
            remaining_predicted * ratio
        } else {
            0.0
        };

        let slow_trigger = !from_cache && actual_secs > predicted_secs * 2.0;
        let low_quality_trigger = quality_score < self.quality_floor;

        if slow_trigger {
            eprintln!(
                "ibvoid-doom-qlock: monitor trigger=slow chunk={} actual={:.1}s predicted={:.1}s",
                chunk_index, actual_secs, predicted_secs
            );
        }
        if low_quality_trigger {
            eprintln!(
                "ibvoid-doom-qlock: monitor trigger=quality chunk={} score={:.2} floor={:.2}",
                chunk_index, quality_score, self.quality_floor
            );
        }

        eprintln!(
            "ibvoid-doom-qlock: chunk {}/{} | eta={} | q={:.2}",
            completed,
            self.total_chunks,
            display_eta(eta_secs),
            quality_score
        );

        if let Err(error) = save_checkpoint_file(&self.path, &state) {
            eprintln!(
                "warning: checkpoint write failed ({}): {}",
                self.path.display(),
                error
            );
        }

        drop(state);
        self.adapt_policy(slow_trigger, low_quality_trigger, false);
    }

    fn record_failure(&self, chunk_index: usize, predicted_secs: f64, reason: &str) {
        let mut state = match self.state.lock() {
            Ok(lock) => lock,
            Err(_) => return,
        };
        state
            .failed
            .retain(|entry| entry.chunk_index != chunk_index);
        state.failed.push(ChunkFailureRecord {
            chunk_index,
            predicted_secs,
            reason: reason.to_string(),
            failed_epoch_secs: now_epoch_secs(),
        });
        if let Err(error) = save_checkpoint_file(&self.path, &state) {
            eprintln!(
                "warning: checkpoint write failed ({}): {}",
                self.path.display(),
                error
            );
        }

        let is_timeout = reason.to_ascii_lowercase().contains("timed out");
        drop(state);
        self.adapt_policy(false, true, is_timeout);
    }

    fn adapt_policy(&self, slow: bool, low_quality: bool, timeout: bool) {
        let mut policy = match self.policy.lock() {
            Ok(lock) => lock,
            Err(_) => return,
        };

        if slow {
            policy.timeout_scale = (policy.timeout_scale * 1.20).clamp(1.0, 3.0);
            policy.extra_retry_budget = (policy.extra_retry_budget + 1).min(2);
        } else {
            policy.timeout_scale = (policy.timeout_scale * 0.98).clamp(1.0, 3.0);
        }

        if low_quality {
            policy.strict_first_pass = true;
            policy.extra_retry_budget = policy.extra_retry_budget.max(1);
        } else {
            policy.strict_first_pass = false;
        }

        if timeout {
            policy.timeout_scale = (policy.timeout_scale * 1.35).clamp(1.0, 4.0);
            policy.extra_retry_budget = (policy.extra_retry_budget + 1).min(3);
        }
    }
}

fn save_checkpoint_file(path: &Path, state: &ChunkCheckpointFile) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let temp_path = path.with_extension("json.tmp");
    let serialized = serde_json::to_string_pretty(state)
        .map_err(|e| format!("serialize checkpoint {}: {e}", path.display()))?;
    std::fs::write(&temp_path, serialized)
        .map_err(|e| format!("write checkpoint temp {}: {e}", temp_path.display()))?;
    std::fs::rename(&temp_path, path).map_err(|e| {
        format!(
            "move checkpoint {} -> {}: {e}",
            temp_path.display(),
            path.display()
        )
    })
}

fn display_eta(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let mins = total / 60;
    let secs = total % 60;
    format!("{mins}m{secs:02}s")
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::transcribe::QualityProfile;

    #[test]
    fn chunk_transcription_sorts_by_index() {
        let mut ts = [
            ChunkTranscription {
                chunk_index: 2,
                offset_secs: 600.0,
                overlap_before: 2.0,
                overlap_after: 2.0,
                cues: vec![],
            },
            ChunkTranscription {
                chunk_index: 0,
                offset_secs: 0.0,
                overlap_before: 0.0,
                overlap_after: 2.0,
                cues: vec![],
            },
            ChunkTranscription {
                chunk_index: 1,
                offset_secs: 300.0,
                overlap_before: 2.0,
                overlap_after: 2.0,
                cues: vec![],
            },
        ];
        ts.sort_by_key(|t| t.chunk_index);
        assert_eq!(ts[0].chunk_index, 0);
        assert_eq!(ts[1].chunk_index, 1);
        assert_eq!(ts[2].chunk_index, 2);
    }

    #[test]
    fn strict_retry_config_forces_safer_settings() {
        let base = TranscribeConfig {
            enabled: true,
            whisper_bin: None,
            whisper_model: None,
            source_lang: "ja".to_string(),
            target_lang: "ja".to_string(),
            whisper_args: vec![],
            vad: true,
            vad_threshold_db: -35.0,
            vad_min_silence: 0.35,
            vad_pad: 0.2,
            gpu: false,
            require_gpu: false,
            quality_profile: QualityProfile::Balanced,
        };

        let retry = strict_retry_config(&base, 2);
        assert_eq!(retry.quality_profile, QualityProfile::Strict);
        assert!(!retry.vad);
        assert!(retry.whisper_args.contains(&"--temperature".to_string()));
    }

    #[test]
    fn chunk_health_flags_repetition_in_strict_mode() {
        let mut cues = Vec::<SubtitleCue>::new();
        for i in 0..80usize {
            cues.push(SubtitleCue {
                index: i + 1,
                timing: format!("00:00:{:02},000 --> 00:00:{:02},900", i % 60, i % 60),
                text: "おかげです".to_string(),
            });
        }
        let health = assess_chunk_health(&cues).expect("health should compute");
        assert!(health.is_pathological(QualityProfile::Strict));
    }

    #[test]
    fn estimate_chunk_predicted_secs_gpu_is_faster_than_cpu() {
        let chunk = AudioChunk {
            index: 0,
            start_sec: 0.0,
            end_sec: 300.0,
            wav_path: std::path::PathBuf::from("/tmp/chunk.wav"),
            overlap_before: 0.0,
            overlap_after: 0.0,
        };
        let mut cfg = TranscribeConfig {
            enabled: true,
            whisper_bin: None,
            whisper_model: None,
            source_lang: "ja".to_string(),
            target_lang: "ja".to_string(),
            whisper_args: vec![],
            vad: true,
            vad_threshold_db: -35.0,
            vad_min_silence: 0.35,
            vad_pad: 0.2,
            gpu: false,
            require_gpu: false,
            quality_profile: QualityProfile::Balanced,
        };
        let cpu = estimate_chunk_predicted_secs(&chunk, &cfg);
        cfg.gpu = true;
        let gpu = estimate_chunk_predicted_secs(&chunk, &cfg);
        assert!(gpu < cpu);
    }

    #[test]
    fn display_eta_is_stable() {
        assert_eq!(display_eta(0.0), "0m00s");
        assert_eq!(display_eta(65.0), "1m05s");
    }

    #[test]
    fn chunk_health_score_is_bounded() {
        let health = ChunkHealth {
            cue_count: 100,
            top_line_ratio: 0.1,
            overlap_ratio: 0.05,
            non_empty_ratio: 0.95,
        };
        let score = health.score(QualityProfile::Strict);
        assert!((0.0..=1.0).contains(&score));
    }
}
