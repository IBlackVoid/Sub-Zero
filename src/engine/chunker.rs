// Silence-Boundary Optimal Decomposition (SBOD) — Theorem 1 of SRT².
//
// Splits audio at natural silence gaps so chunk boundaries never bisect a word
// or sentence.  Each chunk carries enough overlap for the stitcher to detect
// and merge duplicate cues near edges.

use crate::engine::transcribe::{detect_speech_intervals_from_wav, VadInterval};
use std::path::{Path, PathBuf};
use std::process::Command;

/// A contiguous slice of the original audio, ready for independent whisper.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AudioChunk {
    /// Zero-based chunk index.
    pub index: usize,
    /// Absolute start time in the original audio (seconds).
    pub start_sec: f64,
    /// Absolute end time in the original audio (seconds).
    pub end_sec: f64,
    /// Path to the extracted WAV file for this chunk.
    pub wav_path: PathBuf,
    /// Seconds of audio from the *previous* chunk that are duplicated at the
    /// start of this one (used by the stitcher for overlap dedup).
    pub overlap_before: f64,
    /// Seconds of audio from the *next* chunk that are duplicated at the end.
    pub overlap_after: f64,
}

#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Target duration per chunk in seconds (default 300 = 5 min).
    pub target_chunk_secs: f64,
    /// Minimum silence gap duration to consider as a candidate boundary (secs).
    pub min_silence_gap: f64,
    /// Overlap window in seconds added to each side of a chunk boundary.
    pub overlap_secs: f64,
    /// VAD energy threshold in dB.
    pub vad_threshold_db: f64,
    /// Minimum silence duration for VAD.
    pub vad_min_silence: f64,
    /// Padding around detected speech intervals.
    pub vad_pad: f64,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            target_chunk_secs: 300.0,
            min_silence_gap: 0.4,
            overlap_secs: 2.0,
            vad_threshold_db: -35.0,
            vad_min_silence: 0.35,
            vad_pad: 0.20,
        }
    }
}

/// Runs SBOD: detect silence gaps → pick optimal boundaries → export WAV chunks.
pub fn chunk_audio(
    wav_path: &Path,
    out_dir: &Path,
    duration_secs: f64,
    config: &ChunkerConfig,
) -> Result<Vec<AudioChunk>, String> {
    // Short audio — no point chunking.
    if duration_secs <= config.target_chunk_secs * 1.3 {
        return Ok(vec![AudioChunk {
            index: 0,
            start_sec: 0.0,
            end_sec: duration_secs,
            wav_path: wav_path.to_path_buf(),
            overlap_before: 0.0,
            overlap_after: 0.0,
        }]);
    }

    // 1. VAD — find all speech intervals (inversion gives silence gaps).
    let speech = detect_speech_intervals_from_wav(
        wav_path,
        config.vad_threshold_db,
        config.vad_min_silence,
        config.vad_pad,
        None,
    )?;

    let silence_gaps = invert_to_silence(duration_secs, &speech, config.min_silence_gap);

    // 2. Pick boundaries — greedy walk: advance by ~target_chunk_secs, snap to
    //    the nearest silence gap centroid.
    let boundaries = pick_boundaries(duration_secs, &silence_gaps, config.target_chunk_secs);

    // 3. Export each chunk with overlap padding.
    let mut chunks = Vec::with_capacity(boundaries.len() + 1);
    let mut prev_end = 0.0f64;

    for (i, &boundary) in boundaries.iter().enumerate() {
        let is_first = i == 0;
        let overlap_before = if is_first { 0.0 } else { config.overlap_secs };
        let chunk_start = (prev_end - overlap_before).max(0.0);
        let chunk_end = (boundary + config.overlap_secs).min(duration_secs);
        let overlap_after = chunk_end - boundary;

        let wav = export_chunk(wav_path, out_dir, i, chunk_start, chunk_end)?;
        chunks.push(AudioChunk {
            index: i,
            start_sec: chunk_start,
            end_sec: chunk_end,
            wav_path: wav,
            overlap_before: if is_first {
                0.0
            } else {
                prev_end - chunk_start
            },
            overlap_after,
        });
        prev_end = boundary;
    }

    // Last chunk: from last boundary to end.
    let last_idx = boundaries.len();
    let overlap_before = config.overlap_secs;
    let chunk_start = (prev_end - overlap_before).max(0.0);
    let wav = export_chunk(wav_path, out_dir, last_idx, chunk_start, duration_secs)?;
    chunks.push(AudioChunk {
        index: last_idx,
        start_sec: chunk_start,
        end_sec: duration_secs,
        wav_path: wav,
        overlap_before: prev_end - chunk_start,
        overlap_after: 0.0,
    });

    Ok(chunks)
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Invert speech intervals to silence gaps, keeping only those ≥ min_dur.
fn invert_to_silence(duration: f64, speech: &[VadInterval], min_dur: f64) -> Vec<SilenceGap> {
    let mut gaps = Vec::new();
    let mut cursor = 0.0f64;
    for s in speech {
        if s.start > cursor {
            let dur = s.start - cursor;
            if dur >= min_dur {
                gaps.push(SilenceGap {
                    start: cursor,
                    end: s.start,
                    centroid: (cursor + s.start) / 2.0,
                });
            }
        }
        cursor = cursor.max(s.end);
    }
    if duration > cursor {
        let dur = duration - cursor;
        if dur >= min_dur {
            gaps.push(SilenceGap {
                start: cursor,
                end: duration,
                centroid: (cursor + duration) / 2.0,
            });
        }
    }
    gaps
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct SilenceGap {
    start: f64,
    end: f64,
    centroid: f64,
}

/// Greedy boundary selection: walk through the audio at `target_step` intervals
/// and snap each boundary to the nearest silence-gap centroid.
fn pick_boundaries(duration: f64, gaps: &[SilenceGap], target_step: f64) -> Vec<f64> {
    if gaps.is_empty() {
        // No silence gaps found — fall back to uniform splitting.
        let n = (duration / target_step).ceil() as usize;
        if n <= 1 {
            return Vec::new();
        }
        return (1..n).map(|i| (i as f64) * target_step).collect();
    }

    let mut boundaries = Vec::new();
    let mut cursor = target_step;

    while cursor < duration - target_step * 0.3 {
        // Find the silence gap whose centroid is closest to `cursor`.
        let best = gaps
            .iter()
            .filter(|g| {
                // Don't reuse a gap that's behind the last boundary.
                let min_pos = boundaries.last().copied().unwrap_or(0.0) + target_step * 0.3;
                g.centroid >= min_pos && g.centroid < duration - target_step * 0.2
            })
            .min_by(|a, b| {
                let da = (a.centroid - cursor).abs();
                let db = (b.centroid - cursor).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

        match best {
            Some(gap) => {
                boundaries.push(gap.centroid);
                cursor = gap.centroid + target_step;
            }
            None => break,
        }
    }

    boundaries
}

/// Extract a time-slice of a WAV file using ffmpeg.
fn export_chunk(
    source_wav: &Path,
    out_dir: &Path,
    index: usize,
    start: f64,
    end: f64,
) -> Result<PathBuf, String> {
    let out_path = out_dir.join(format!("chunk_{index:04}.wav"));
    let duration = end - start;

    let ffmpeg = find_ffmpeg()?;
    let status = Command::new(&ffmpeg)
        .arg("-hide_banner")
        .arg("-nostdin")
        .arg("-v")
        .arg("error")
        .arg("-y")
        .arg("-ss")
        .arg(format!("{start:.3}"))
        .arg("-t")
        .arg(format!("{duration:.3}"))
        .arg("-i")
        .arg(source_wav)
        .arg("-c")
        .arg("copy")
        .arg(&out_path)
        .status()
        .map_err(|e| format!("failed to spawn ffmpeg for chunk export: {e}"))?;

    if !status.success() {
        return Err(format!("ffmpeg chunk export failed with status: {status}"));
    }
    Ok(out_path)
}

fn find_ffmpeg() -> Result<PathBuf, String> {
    let path_var = std::env::var_os("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&path_var) {
        for name in &["ffmpeg", "ffmpeg.exe"] {
            let p = dir.join(name);
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    Err("ffmpeg not found in PATH".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_audio_returns_single_chunk() {
        // 4 min audio with 5 min target → no splitting.
        let cfg = ChunkerConfig {
            target_chunk_secs: 300.0,
            ..Default::default()
        };
        let tmp = std::env::temp_dir().join("sub_zero_chunker_test_stub.wav");
        // We only test the early-return path; no actual file needed.
        let chunks = chunk_audio(&tmp, &std::env::temp_dir(), 240.0, &cfg).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_sec, 0.0);
        assert_eq!(chunks[0].end_sec, 240.0);
    }

    #[test]
    fn pick_boundaries_uniform_fallback() {
        let bounds = pick_boundaries(900.0, &[], 300.0);
        assert_eq!(bounds.len(), 2);
        assert!((bounds[0] - 300.0).abs() < 0.01);
        assert!((bounds[1] - 600.0).abs() < 0.01);
    }

    #[test]
    fn pick_boundaries_snaps_to_gaps() {
        let gaps = vec![
            SilenceGap {
                start: 290.0,
                end: 295.0,
                centroid: 292.5,
            },
            SilenceGap {
                start: 580.0,
                end: 590.0,
                centroid: 585.0,
            },
        ];
        let bounds = pick_boundaries(900.0, &gaps, 300.0);
        assert_eq!(bounds.len(), 2);
        assert!((bounds[0] - 292.5).abs() < 0.01);
        assert!((bounds[1] - 585.0).abs() < 0.01);
    }

    #[test]
    fn invert_to_silence_finds_gaps() {
        let speech = vec![
            VadInterval {
                start: 0.0,
                end: 100.0,
            },
            VadInterval {
                start: 102.0,
                end: 200.0,
            },
        ];
        let gaps = invert_to_silence(200.0, &speech, 1.0);
        assert_eq!(gaps.len(), 1);
        assert!((gaps[0].start - 100.0).abs() < 0.01);
        assert!((gaps[0].end - 102.0).abs() < 0.01);
    }
}
