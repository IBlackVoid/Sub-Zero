use crate::engine::srt::SubtitleCue;

#[derive(Debug, Clone, Copy)]
pub(super) struct Interval {
    pub(super) start: f64,
    pub(super) end: f64,
}

pub(super) fn interval_overlap_seconds(a: Interval, b: Interval) -> f64 {
    let start = a.start.max(b.start);
    let end = a.end.min(b.end);
    if end <= start {
        0.0
    } else {
        end - start
    }
}

pub(super) fn scene_time_span(scene: &[SubtitleCue]) -> Option<(f64, f64)> {
    let mut min_start = f64::INFINITY;
    let mut max_end = 0.0f64;
    for cue in scene {
        let (start, end) = parse_srt_timing_line(&cue.timing).ok()?;
        min_start = min_start.min(start);
        max_end = max_end.max(end);
    }
    if min_start.is_finite() && max_end > min_start {
        Some((min_start, max_end))
    } else {
        None
    }
}

pub(super) fn shift_cues_by_offset(
    cues: &[SubtitleCue],
    offset: f64,
) -> Result<Vec<SubtitleCue>, String> {
    let mut shifted = Vec::<SubtitleCue>::with_capacity(cues.len());
    for cue in cues {
        let (start, end) = parse_srt_timing_line(&cue.timing)?;
        shifted.push(SubtitleCue {
            index: cue.index,
            timing: format_srt_timing_line(start + offset, end + offset),
            text: cue.text.clone(),
        });
    }
    Ok(shifted)
}

pub(super) fn format_srt_timing_line(start: f64, end: f64) -> String {
    format!(
        "{} --> {}",
        format_srt_timestamp(start),
        format_srt_timestamp(end)
    )
}

pub(super) fn format_srt_timestamp(seconds: f64) -> String {
    let clamped = seconds.max(0.0);
    let total_ms = (clamped * 1000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let secs = (total_ms % 60_000) / 1000;
    let millis = total_ms % 1000;
    format!("{hours:02}:{minutes:02}:{secs:02},{millis:03}")
}

pub(super) fn parse_srt_timestamp_to_seconds(ts: &str) -> Result<f64, String> {
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
