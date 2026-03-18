// Chunk Stitcher — merges N partial SRTs from parallel transcription into one
// coherent timeline.
//
// Core responsibilities:
// 1. Offset each chunk's timestamps by its absolute position in the source audio.
// 2. Detect and deduplicate cues in overlap regions using Levenshtein similarity.
// 3. Re-index the final cue sequence.

use crate::engine::parallel::ChunkTranscription;
use crate::engine::srt::SubtitleCue;

/// Similarity threshold (0.0–1.0) for considering two cues as duplicates.
const DEDUP_SIMILARITY_THRESHOLD: f64 = 0.6;

/// Merge chunk transcriptions into a single cue sequence with correct global
/// timestamps and no duplicates at chunk boundaries.
pub fn stitch_chunks(chunks: &[ChunkTranscription]) -> Result<Vec<SubtitleCue>, String> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    if chunks.len() == 1 {
        return Ok(reindex(&chunks[0].cues));
    }

    let mut all_cues: Vec<TimedCue> = Vec::new();

    for chunk in chunks {
        for cue in &chunk.cues {
            let (start, end) = parse_timing_pair(&cue.timing)?;
            // Shift timestamps to absolute position.
            let abs_start = start + chunk.offset_secs;
            let abs_end = end + chunk.offset_secs;

            all_cues.push(TimedCue {
                abs_start,
                abs_end,
                text: cue.text.clone(),
                chunk_index: chunk.chunk_index,
            });
        }
    }

    // Sort by absolute start time.
    all_cues.sort_by(|a, b| {
        a.abs_start
            .partial_cmp(&b.abs_start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate overlapping cues from adjacent chunks.
    let deduped = deduplicate_overlaps(&all_cues);

    // Build final cue list with fresh indices and proper timestamps.
    let final_cues: Vec<SubtitleCue> = deduped
        .into_iter()
        .enumerate()
        .map(|(i, tc)| SubtitleCue {
            index: i + 1,
            timing: format_srt_timing(tc.abs_start, tc.abs_end),
            text: tc.text,
        })
        .collect();

    Ok(final_cues)
}

// ── internal types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TimedCue {
    abs_start: f64,
    abs_end: f64,
    text: String,
    chunk_index: usize,
}

// ── deduplication ────────────────────────────────────────────────────────────

/// Walk through sorted cues and remove near-duplicates (same text,
/// overlapping time) that arise from chunk overlap regions.
fn deduplicate_overlaps(cues: &[TimedCue]) -> Vec<TimedCue> {
    if cues.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<TimedCue> = Vec::with_capacity(cues.len());
    result.push(cues[0].clone());

    for cue in cues.iter().skip(1) {
        let last = result.last().unwrap();
        // If cues come from different chunks and are very close in time with
        // similar text, they're overlap duplicates.
        if last.chunk_index != cue.chunk_index {
            let time_gap = (cue.abs_start - last.abs_start).abs();
            if time_gap < 3.0 {
                let sim = normalized_similarity(&last.text, &cue.text);
                if sim >= DEDUP_SIMILARITY_THRESHOLD {
                    // Keep the one from the earlier chunk (it had more context
                    // before the boundary).
                    continue;
                }
            }
        }
        // Also skip exact text duplicates within a 2-second window even from
        // the same chunk (whisper sometimes stutters).
        if (cue.abs_start - last.abs_start).abs() < 2.0 && cue.text == last.text {
            continue;
        }
        result.push(cue.clone());
    }

    result
}

/// Normalized Levenshtein similarity in [0, 1].
fn normalized_similarity(a: &str, b: &str) -> f64 {
    strsim::normalized_levenshtein(a, b)
}

// ── timestamp helpers ────────────────────────────────────────────────────────

/// Parse "HH:MM:SS,mmm --> HH:MM:SS,mmm" into (start_secs, end_secs).
fn parse_timing_pair(timing: &str) -> Result<(f64, f64), String> {
    let (start_str, end_str) = timing
        .split_once("-->")
        .ok_or_else(|| format!("invalid timing line: {timing}"))?;
    let start = parse_ts(start_str.trim())?;
    let end = parse_ts(end_str.trim())?;
    Ok((start, end))
}

fn parse_ts(ts: &str) -> Result<f64, String> {
    let (hms, ms) = ts
        .split_once(',')
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?;
    let parts: Vec<&str> = hms.split(':').collect();
    if parts.len() != 3 {
        return Err(format!("invalid timestamp: {ts}"));
    }
    let h: f64 = parts[0].parse().map_err(|_| format!("bad hour: {ts}"))?;
    let m: f64 = parts[1].parse().map_err(|_| format!("bad minute: {ts}"))?;
    let s: f64 = parts[2].parse().map_err(|_| format!("bad second: {ts}"))?;
    let ms: f64 = ms.parse().map_err(|_| format!("bad ms: {ts}"))?;
    Ok(h * 3600.0 + m * 60.0 + s + ms / 1000.0)
}

fn format_srt_timing(start: f64, end: f64) -> String {
    format!("{} --> {}", format_ts(start), format_ts(end))
}

fn format_ts(secs: f64) -> String {
    let total_ms = (secs * 1000.0).round() as u64;
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

fn reindex(cues: &[SubtitleCue]) -> Vec<SubtitleCue> {
    cues.iter()
        .enumerate()
        .map(|(i, c)| SubtitleCue {
            index: i + 1,
            timing: c.timing.clone(),
            text: c.text.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_ts_roundtrip() {
        let ts = "01:23:45,678";
        let secs = parse_ts(ts).unwrap();
        assert_eq!(format_ts(secs), ts);
    }

    #[test]
    fn dedup_removes_overlap_duplicates() {
        let cues = vec![
            TimedCue {
                abs_start: 298.0,
                abs_end: 301.0,
                text: "Hello everyone".to_string(),
                chunk_index: 0,
            },
            TimedCue {
                abs_start: 298.5,
                abs_end: 301.5,
                text: "Hello everyone".to_string(),
                chunk_index: 1,
            },
            TimedCue {
                abs_start: 305.0,
                abs_end: 308.0,
                text: "Goodbye".to_string(),
                chunk_index: 1,
            },
        ];
        let result = deduplicate_overlaps(&cues);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "Hello everyone");
        assert_eq!(result[1].text, "Goodbye");
    }

    #[test]
    fn stitch_single_chunk() {
        let chunk = ChunkTranscription {
            chunk_index: 0,
            offset_secs: 0.0,
            overlap_before: 0.0,
            overlap_after: 0.0,
            cues: vec![
                SubtitleCue {
                    index: 1,
                    timing: "00:00:01,000 --> 00:00:03,000".to_string(),
                    text: "Hello".to_string(),
                },
                SubtitleCue {
                    index: 2,
                    timing: "00:00:04,000 --> 00:00:06,000".to_string(),
                    text: "World".to_string(),
                },
            ],
        };
        let result = stitch_chunks(&[chunk]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].index, 1);
        assert_eq!(result[1].index, 2);
    }

    #[test]
    fn stitch_offsets_timestamps() {
        let chunks = vec![
            ChunkTranscription {
                chunk_index: 0,
                offset_secs: 0.0,
                overlap_before: 0.0,
                overlap_after: 0.0,
                cues: vec![SubtitleCue {
                    index: 1,
                    timing: "00:00:01,000 --> 00:00:02,000".to_string(),
                    text: "First".to_string(),
                }],
            },
            ChunkTranscription {
                chunk_index: 1,
                offset_secs: 300.0,
                overlap_before: 2.0,
                overlap_after: 0.0,
                cues: vec![SubtitleCue {
                    index: 1,
                    timing: "00:00:05,000 --> 00:00:07,000".to_string(),
                    text: "Second".to_string(),
                }],
            },
        ];
        let result = stitch_chunks(&chunks).unwrap();
        assert_eq!(result.len(), 2);
        // Second chunk's cue at 5s + 300s offset = 305s = 00:05:05,000
        assert!(result[1].timing.starts_with("00:05:05"));
    }
}
