use crate::engine::srt::SubtitleCue;
use crate::engine::transcribe::{detect_speech_intervals_from_wav, VadInterval};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::audio::extract_audio_to_wav;
use super::paths::checkpoint_dir_for;
use super::time::parse_srt_timing_line;

#[derive(Debug, Default, Clone)]
pub(super) struct AudioDiarizationStats {
    pub(super) audio_available: bool,
    pub(super) speakers: usize,
    pub(super) used_segments: usize,
    pub(super) assigned_cues: usize,
    pub(super) unique_speakers: usize,
    pub(super) sidecar_file: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct AudioSource {
    wav_path: PathBuf,
    checkpoint_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct DiarizationPayload {
    speakers: usize,
    segments: Vec<DiarizationSegment>,
}

#[derive(Debug, Deserialize)]
struct DiarizationSegment {
    start: f64,
    end: f64,
    speaker: usize,
}

pub(super) fn audio_diarize_speakers_for_cues(
    input: &Path,
    audio_hint: Option<&Path>,
    cues: &[SubtitleCue],
    vad_threshold_db: f64,
    vad_min_silence: f64,
    vad_pad: f64,
    max_speakers: usize,
) -> Result<(Vec<Option<String>>, AudioDiarizationStats), String> {
    let mut stats = AudioDiarizationStats::default();

    let Some(audio) = resolve_audio_source(input, audio_hint)? else {
        return Ok((vec![None; cues.len()], stats));
    };
    stats.audio_available = true;

    let speech = detect_speech_intervals_from_wav(
        &audio.wav_path,
        vad_threshold_db,
        vad_min_silence,
        vad_pad,
        None,
    )?;

    if speech.is_empty() {
        return Ok((vec![None; cues.len()], stats));
    }

    let (payload_path, sidecar_path) = run_diarization_worker(&audio, &speech, max_speakers)?;

    let raw = std::fs::read_to_string(&payload_path)
        .map_err(|e| format!("{}: {e}", payload_path.display()))?;
    let payload: DiarizationPayload = serde_json::from_str(&raw)
        .map_err(|e| format!("invalid diarization JSON {}: {e}", payload_path.display()))?;

    stats.speakers = payload.speakers;
    stats.used_segments = payload.segments.len();
    stats.sidecar_file = Some(sidecar_path);

    let speakers = assign_segments_to_cues(cues, &payload.segments);
    stats.assigned_cues = speakers.iter().filter(|s| s.is_some()).count();
    stats.unique_speakers = speakers
        .iter()
        .filter_map(|s| s.as_ref())
        .collect::<HashSet<_>>()
        .len();

    Ok((speakers, stats))
}

fn resolve_audio_source(
    input: &Path,
    audio_hint: Option<&Path>,
) -> Result<Option<AudioSource>, String> {
    if cues_need_no_audio(input, audio_hint) {
        return Ok(None);
    }

    let checkpoint_dir = checkpoint_dir_for(input)?;
    let work_dir = checkpoint_dir.join("work");
    std::fs::create_dir_all(&work_dir).map_err(|e| format!("{}: {e}", work_dir.display()))?;

    // Prefer a directly-supplied WAV from an earlier transcription step.
    if let Some(hint) = audio_hint {
        if hint.is_file() && is_wav_path(hint) {
            return Ok(Some(AudioSource {
                wav_path: hint.to_path_buf(),
                checkpoint_dir,
            }));
        }
    }

    // Fallback: if the input is a video, extract a checkpoint-local WAV.
    if !input.is_file() || is_wav_path(input) {
        return Ok(None);
    }

    let wav_path = work_dir.join("speaker_diarize_audio.wav");
    if !wav_path.is_file() {
        extract_audio_to_wav(input, &wav_path)?;
    }

    Ok(Some(AudioSource {
        wav_path,
        checkpoint_dir,
    }))
}

fn cues_need_no_audio(input: &Path, audio_hint: Option<&Path>) -> bool {
    if audio_hint.is_some() {
        return false;
    }
    // Pure subtitle input: we have no audio to diarize against.
    input
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|ext| ext.eq_ignore_ascii_case("srt"))
        .unwrap_or(false)
}

fn is_wav_path(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|ext| ext.eq_ignore_ascii_case("wav"))
        .unwrap_or(false)
}

fn run_diarization_worker(
    audio: &AudioSource,
    speech: &[VadInterval],
    max_speakers: usize,
) -> Result<(PathBuf, PathBuf), String> {
    let python = find_python_with_numpy().ok_or_else(|| {
        "python with numpy not found (required for --speaker-diarize)".to_string()
    })?;

    let script = PathBuf::from("scripts").join("diarize_speakers.py");
    if !script.is_file() {
        return Err(format!(
            "diarization script not found: {}",
            script.display()
        ));
    }

    let segments_path = audio.checkpoint_dir.join("speaker_vad_segments.json");
    let segments_json = speech
        .iter()
        .map(|s| serde_json::json!({ "start": s.start, "end": s.end }))
        .collect::<Vec<_>>();
    let serialized = serde_json::to_string(&segments_json)
        .map_err(|e| format!("failed to serialize VAD segments: {e}"))?;
    std::fs::write(&segments_path, serialized)
        .map_err(|e| format!("{}: {e}", segments_path.display()))?;

    let out_path = audio.checkpoint_dir.join("speaker_diarization.json");

    let output = Command::new(&python)
        .arg(&script)
        .arg("--wav")
        .arg(&audio.wav_path)
        .arg("--segments-json")
        .arg(&segments_path)
        .arg("--max-speakers")
        .arg(max_speakers.to_string())
        .arg("--out-json")
        .arg(&out_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("failed to spawn diarization worker: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "diarization worker failed with status {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    Ok((out_path.clone(), out_path))
}

fn find_python_with_numpy() -> Option<String> {
    let cwd = std::env::current_dir().ok()?;

    let candidates = [
        cwd.join(".venv").join("Scripts").join("python.exe"),
        cwd.join(".venv").join("bin").join("python3"),
        cwd.join(".venv").join("bin").join("python"),
        PathBuf::from("python3"),
        PathBuf::from("python"),
    ];

    for candidate in candidates {
        let as_str = candidate.to_string_lossy().to_string();
        let status = Command::new(&as_str)
            .arg("-c")
            .arg("import numpy")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if matches!(status, Ok(s) if s.success()) {
            return Some(as_str);
        }
    }

    None
}

fn assign_segments_to_cues(
    cues: &[SubtitleCue],
    segments: &[DiarizationSegment],
) -> Vec<Option<String>> {
    if cues.is_empty() {
        return Vec::new();
    }
    if segments.is_empty() {
        return vec![None; cues.len()];
    }

    let mut out = vec![None::<String>; cues.len()];
    let mut seg_idx = 0usize;

    for (cue_idx, cue) in cues.iter().enumerate() {
        let Ok((cue_start, cue_end)) = parse_srt_timing_line(&cue.timing) else {
            continue;
        };
        if !(cue_start.is_finite() && cue_end.is_finite()) || cue_end <= cue_start {
            continue;
        }

        while seg_idx < segments.len() && segments[seg_idx].end <= cue_start {
            seg_idx += 1;
        }

        let mut best = None::<(usize, f64)>;
        let mut scan = seg_idx;
        while scan < segments.len() {
            let s = &segments[scan];
            if s.start >= cue_end {
                break;
            }
            let overlap = (cue_end.min(s.end) - cue_start.max(s.start)).max(0.0);
            if overlap > 0.0 {
                match best {
                    None => best = Some((s.speaker, overlap)),
                    Some((_, best_overlap)) if overlap > best_overlap => {
                        best = Some((s.speaker, overlap));
                    }
                    _ => {}
                }
            }
            scan += 1;
        }

        if let Some((speaker, _)) = best {
            out[cue_idx] = Some(format!("spk{speaker}"));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::assign_segments_to_cues;
    use super::DiarizationSegment;
    use crate::engine::srt::SubtitleCue;

    #[test]
    fn assign_segments_to_cues_picks_max_overlap() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,200 --> 00:00:00,800".to_string(),
                text: "a".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,200 --> 00:00:01,400".to_string(),
                text: "b".to_string(),
            },
        ];
        let segments = vec![
            DiarizationSegment {
                start: 0.0,
                end: 1.0,
                speaker: 0,
            },
            DiarizationSegment {
                start: 1.0,
                end: 2.0,
                speaker: 1,
            },
        ];
        let assigned = assign_segments_to_cues(&cues, &segments);
        assert_eq!(assigned[0].as_deref(), Some("spk0"));
        assert_eq!(assigned[1].as_deref(), Some("spk1"));
    }
}
