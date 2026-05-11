use crate::engine::deep_scan::ContentMap;
use crate::engine::srt::SubtitleCue;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn parse_srt_timing_line(line: &str) -> Result<(f64, f64), String> {
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

pub(super) fn parse_srt_timestamp_to_seconds(ts: &str) -> Result<f64, String> {
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

pub(super) fn is_srt_path(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|ext| ext.eq_ignore_ascii_case("srt"))
        .unwrap_or(false)
}

pub(super) fn duration_from_cues(cues: &[SubtitleCue]) -> Option<f64> {
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

pub(super) fn bucket_cpu_cores(cores: usize) -> usize {
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

pub(super) fn bucket_ram_mb(ram_mb: u64) -> u64 {
    if ram_mb == 0 {
        0
    } else {
        ram_mb.div_ceil(2_048) * 2_048
    }
}

pub(super) fn bucket_vram_mb(vram_mb: u64) -> u64 {
    if vram_mb == 0 {
        0
    } else {
        vram_mb.div_ceil(1_024) * 1_024
    }
}

pub(super) fn parse_prefixed_u64(input: &str, prefix: &str) -> Option<u64> {
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

pub(super) fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(super) fn sanitize_fingerprint_component(input: &str) -> String {
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

pub(super) fn content_profile_hash(content: &ContentMap) -> String {
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

pub(super) fn find_in_path(candidates: &[&str]) -> Option<PathBuf> {
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

pub(super) fn display_ram(total_ram_mb: Option<u64>) -> String {
    total_ram_mb
        .map(|value| format!("{:.1}GB", value as f64 / 1024.0))
        .unwrap_or_else(|| "unknown".to_string())
}

pub(super) fn display_duration(duration_secs: Option<f64>) -> String {
    duration_secs
        .map(|value| format!("{value:.1}s"))
        .unwrap_or_else(|| "unknown".to_string())
}

pub(super) fn display_opt_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "auto".to_string())
}

pub(super) fn display_disk_mbps(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.0}"))
        .unwrap_or_else(|| "unknown".to_string())
}
