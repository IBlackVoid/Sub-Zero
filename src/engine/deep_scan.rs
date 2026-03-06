use crate::engine::srt::parse_srt_file;
use crate::engine::transcribe::detect_speech_intervals_from_wav;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentMap {
    pub input_kind: String,
    pub total_duration_secs: f64,
    pub speech_duration_secs: f64,
    pub silence_duration_secs: f64,
    pub estimated_cues: usize,
    pub avg_difficulty: f64,
    pub speaker_complexity_score: f64,
    pub energy_variance_score: f64,
    pub overlap_risk_score: f64,
    pub scene_count: usize,
    pub boundaries: Vec<f64>,
    pub scenes: Vec<SceneProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneProfile {
    pub index: usize,
    pub start_secs: f64,
    pub end_secs: f64,
    pub duration_secs: f64,
    pub speech_density: f64,
    pub difficulty: f64,
    pub scene_type: String,
    pub speaker_estimate: usize,
    pub energy_level: String,
    pub emotion_hint: String,
}

#[derive(Debug, Clone, Copy)]
pub struct DeepScanConfig {
    pub vad_threshold_db: f64,
    pub vad_min_silence: f64,
    pub vad_pad: f64,
}

pub fn scan_input(input: &Path, config: DeepScanConfig) -> Result<ContentMap, String> {
    if is_srt_path(input) {
        return scan_srt(input);
    }
    scan_media(input, config)
}

fn scan_srt(input: &Path) -> Result<ContentMap, String> {
    let cues = parse_srt_file(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let total_duration_secs = srt_total_duration_secs(&cues).unwrap_or(0.0);
    let estimated_cues = cues.len();
    let avg_cue_secs = if estimated_cues > 0 {
        (total_duration_secs / estimated_cues as f64).max(0.001)
    } else {
        0.0
    };
    let avg_difficulty = if avg_cue_secs >= 4.0 {
        0.20
    } else if avg_cue_secs >= 2.5 {
        0.35
    } else {
        0.55
    };
    Ok(ContentMap {
        input_kind: "srt".to_string(),
        total_duration_secs,
        speech_duration_secs: total_duration_secs,
        silence_duration_secs: 0.0,
        estimated_cues,
        avg_difficulty,
        speaker_complexity_score: 0.1,
        energy_variance_score: 0.1,
        overlap_risk_score: 0.0,
        scene_count: 1,
        boundaries: Vec::new(),
        scenes: vec![SceneProfile {
            index: 0,
            start_secs: 0.0,
            end_secs: total_duration_secs.max(0.1),
            duration_secs: total_duration_secs.max(0.1),
            speech_density: 1.0,
            difficulty: avg_difficulty,
            scene_type: "subtitle-source".to_string(),
            speaker_estimate: 1,
            energy_level: "medium".to_string(),
            emotion_hint: "neutral".to_string(),
        }],
    })
}

fn scan_media(input: &Path, config: DeepScanConfig) -> Result<ContentMap, String> {
    let probe = probe_media_format(input)?;
    let total_duration_secs = probe.duration_secs.ok_or_else(|| {
        format!(
            "ffprobe could not determine duration for {}",
            input.display()
        )
    })?;

    let temp_dir = create_temp_scan_dir(input)?;
    let wav = temp_dir.join("deep_scan_audio.wav");
    extract_audio_to_wav(input, &wav)?;
    let speech = detect_speech_intervals_from_wav(
        &wav,
        config.vad_threshold_db,
        config.vad_min_silence,
        config.vad_pad,
        Some(total_duration_secs),
    )?;

    let speech_duration_secs = speech
        .iter()
        .map(|interval| (interval.end - interval.start).max(0.0))
        .sum::<f64>()
        .min(total_duration_secs);
    let silence_duration_secs = (total_duration_secs - speech_duration_secs).max(0.0);

    let silence_gaps = invert_to_silence(total_duration_secs, &speech);
    let boundaries = silence_gaps
        .iter()
        .filter(|gap| (gap.end - gap.start) >= 1.5)
        .map(|gap| (gap.start + gap.end) * 0.5)
        .collect::<Vec<_>>();

    let scenes = build_scene_profiles(total_duration_secs, &speech, &boundaries);
    let avg_difficulty = if scenes.is_empty() {
        0.5
    } else {
        scenes.iter().map(|scene| scene.difficulty).sum::<f64>() / scenes.len() as f64
    };
    let speaker_complexity_score = estimate_speaker_complexity(&scenes);
    let energy_variance_score = estimate_energy_variance(&scenes);
    let overlap_risk_score = estimate_overlap_risk(&scenes);
    let estimated_cues = estimate_cues(speech_duration_secs, avg_difficulty);
    let _ = std::fs::remove_file(&wav);
    let _ = std::fs::remove_dir_all(&temp_dir);

    Ok(ContentMap {
        input_kind: probe.format_name.unwrap_or_else(|| "video".to_string()),
        total_duration_secs,
        speech_duration_secs,
        silence_duration_secs,
        estimated_cues,
        avg_difficulty,
        speaker_complexity_score,
        energy_variance_score,
        overlap_risk_score,
        scene_count: scenes.len(),
        boundaries,
        scenes,
    })
}

fn estimate_cues(speech_duration_secs: f64, avg_difficulty: f64) -> usize {
    let base = (speech_duration_secs / 2.8).round();
    let scaled = base * (1.0 + (avg_difficulty - 0.35) * 0.35);
    scaled.max(1.0) as usize
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

fn build_scene_profiles(
    total_duration_secs: f64,
    speech_intervals: &[crate::engine::transcribe::VadInterval],
    boundaries: &[f64],
) -> Vec<SceneProfile> {
    let mut sorted_boundaries = boundaries
        .iter()
        .copied()
        .filter(|boundary| {
            boundary.is_finite() && *boundary > 0.0 && *boundary < total_duration_secs
        })
        .collect::<Vec<_>>();
    sorted_boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_boundaries.dedup_by(|a, b| (*a - *b).abs() < 0.05);

    let mut scenes = Vec::<SceneProfile>::new();
    let mut cursor = 0.0f64;
    let mut index = 0usize;
    for boundary in sorted_boundaries
        .into_iter()
        .chain(std::iter::once(total_duration_secs))
    {
        if boundary <= cursor {
            continue;
        }
        let duration_secs = boundary - cursor;
        let speech_secs = speech_intervals
            .iter()
            .map(|interval| overlap_seconds(cursor, boundary, interval.start, interval.end))
            .sum::<f64>();
        let speech_density = (speech_secs / duration_secs.max(0.001)).clamp(0.0, 1.0);
        let difficulty = estimate_scene_difficulty(duration_secs, speech_density);
        let scene_type = classify_scene_type(duration_secs, speech_density);

        scenes.push(SceneProfile {
            index,
            start_secs: cursor,
            end_secs: boundary,
            duration_secs,
            speech_density,
            difficulty,
            scene_type,
            speaker_estimate: estimate_speaker_count(duration_secs, speech_density, difficulty),
            energy_level: classify_energy_level(speech_density, difficulty),
            emotion_hint: classify_emotion_hint(speech_density, difficulty),
        });
        cursor = boundary;
        index += 1;
    }

    if scenes.is_empty() {
        scenes.push(SceneProfile {
            index: 0,
            start_secs: 0.0,
            end_secs: total_duration_secs.max(0.1),
            duration_secs: total_duration_secs.max(0.1),
            speech_density: 0.0,
            difficulty: 0.5,
            scene_type: "unknown".to_string(),
            speaker_estimate: 1,
            energy_level: "low".to_string(),
            emotion_hint: "unknown".to_string(),
        });
    }

    scenes
}

fn estimate_speaker_count(duration_secs: f64, speech_density: f64, difficulty: f64) -> usize {
    let mut score = 1usize;
    if speech_density >= 0.85 {
        score += 1;
    }
    if duration_secs <= 45.0 && speech_density >= 0.75 {
        score += 1;
    }
    if difficulty >= 0.70 {
        score += 1;
    }
    score.clamp(1, 4)
}

fn classify_energy_level(speech_density: f64, difficulty: f64) -> String {
    if speech_density >= 0.90 || difficulty >= 0.80 {
        "high".to_string()
    } else if speech_density >= 0.55 || difficulty >= 0.45 {
        "medium".to_string()
    } else {
        "low".to_string()
    }
}

fn classify_emotion_hint(speech_density: f64, difficulty: f64) -> String {
    if difficulty >= 0.80 {
        "intense".to_string()
    } else if speech_density <= 0.25 {
        "calm".to_string()
    } else if speech_density >= 0.85 {
        "urgent".to_string()
    } else {
        "neutral".to_string()
    }
}

fn estimate_speaker_complexity(scenes: &[SceneProfile]) -> f64 {
    if scenes.is_empty() {
        return 0.0;
    }
    let avg_speakers = scenes
        .iter()
        .map(|scene| scene.speaker_estimate as f64)
        .sum::<f64>()
        / scenes.len() as f64;
    ((avg_speakers - 1.0) / 3.0).clamp(0.0, 1.0)
}

fn estimate_energy_variance(scenes: &[SceneProfile]) -> f64 {
    if scenes.len() < 2 {
        return 0.0;
    }
    let mut changes = 0usize;
    for pair in scenes.windows(2) {
        if pair[0].energy_level != pair[1].energy_level {
            changes += 1;
        }
    }
    (changes as f64 / (scenes.len() - 1) as f64).clamp(0.0, 1.0)
}

fn estimate_overlap_risk(scenes: &[SceneProfile]) -> f64 {
    if scenes.is_empty() {
        return 0.0;
    }
    let risky = scenes
        .iter()
        .filter(|scene| scene.speaker_estimate >= 3 && scene.speech_density >= 0.75)
        .count();
    (risky as f64 / scenes.len() as f64).clamp(0.0, 1.0)
}

fn estimate_scene_difficulty(duration_secs: f64, speech_density: f64) -> f64 {
    let pacing_factor: f64 = if duration_secs < 25.0 {
        0.25
    } else if duration_secs < 60.0 {
        0.15
    } else if duration_secs > 600.0 {
        0.18
    } else {
        0.05
    };
    let density_factor: f64 = if speech_density >= 0.92 {
        0.35
    } else if speech_density >= 0.75 {
        0.22
    } else if speech_density <= 0.20 {
        0.28
    } else {
        0.12
    };
    (0.18 + pacing_factor + density_factor).clamp(0.10, 0.95)
}

fn classify_scene_type(duration_secs: f64, speech_density: f64) -> String {
    if speech_density <= 0.20 {
        return "ambient".to_string();
    }
    if speech_density >= 0.90 && duration_secs < 60.0 {
        return "rapid-dialogue".to_string();
    }
    if speech_density >= 0.90 {
        return "dialogue-dense".to_string();
    }
    if duration_secs >= 420.0 {
        return "long-scene".to_string();
    }
    "dialogue".to_string()
}

fn overlap_seconds(a_start: f64, a_end: f64, b_start: f64, b_end: f64) -> f64 {
    let start = a_start.max(b_start);
    let end = a_end.min(b_end);
    if end <= start {
        0.0
    } else {
        end - start
    }
}

fn invert_to_silence(
    duration_secs: f64,
    speech: &[crate::engine::transcribe::VadInterval],
) -> Vec<crate::engine::transcribe::VadInterval> {
    let mut result = Vec::<crate::engine::transcribe::VadInterval>::new();
    let mut cursor = 0.0f64;
    for interval in speech {
        if interval.start > cursor {
            result.push(crate::engine::transcribe::VadInterval {
                start: cursor,
                end: interval.start,
            });
        }
        cursor = cursor.max(interval.end);
    }
    if cursor < duration_secs {
        result.push(crate::engine::transcribe::VadInterval {
            start: cursor,
            end: duration_secs,
        });
    }
    result
}

fn srt_total_duration_secs(cues: &[crate::engine::srt::SubtitleCue]) -> Option<f64> {
    let mut max_end = 0.0f64;
    for cue in cues {
        let (_, end) = parse_srt_timing_line(&cue.timing).ok()?;
        max_end = max_end.max(end);
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

fn extract_audio_to_wav(video: &Path, wav_out: &Path) -> Result<(), String> {
    let ffmpeg = find_in_path(&["ffmpeg", "ffmpeg.exe"])
        .ok_or_else(|| "ffmpeg not found in PATH".to_string())?;

    let status = Command::new(ffmpeg)
        .arg("-hide_banner")
        .arg("-nostdin")
        .arg("-v")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(video)
        .arg("-vn")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-f")
        .arg("wav")
        .arg(wav_out)
        .status()
        .map_err(|e| format!("failed to spawn ffmpeg: {e}"))?;

    if !status.success() {
        return Err(format!("ffmpeg failed with status: {status}"));
    }
    Ok(())
}

fn create_temp_scan_dir(input: &Path) -> Result<PathBuf, String> {
    let stem = input
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("input");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("sub_zero_deep_scan_{stem}_{stamp}"));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn is_srt_path(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|ext| ext.eq_ignore_ascii_case("srt"))
        .unwrap_or(false)
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

#[cfg(test)]
mod tests {
    use super::{
        build_scene_profiles, classify_scene_type, estimate_cues, estimate_scene_difficulty,
        parse_ffprobe_output,
    };
    use crate::engine::transcribe::VadInterval;

    #[test]
    fn scene_builder_creates_monotonic_scenes() {
        let speech = vec![
            VadInterval {
                start: 0.0,
                end: 9.0,
            },
            VadInterval {
                start: 12.0,
                end: 20.0,
            },
        ];
        let boundaries = vec![10.0];
        let scenes = build_scene_profiles(20.0, &speech, &boundaries);
        assert_eq!(scenes.len(), 2);
        assert!(scenes[0].end_secs <= scenes[1].start_secs);
        assert!(scenes[0].duration_secs > 0.0);
        assert!(scenes[1].duration_secs > 0.0);
    }

    #[test]
    fn parse_ffprobe_output_reads_fields() {
        let probe =
            parse_ffprobe_output("format_name=matroska,webm\nduration=123.4").expect("parse ok");
        assert_eq!(probe.format_name.as_deref(), Some("matroska,webm"));
        assert_eq!(probe.duration_secs, Some(123.4));
    }

    #[test]
    fn cue_estimator_scales_with_difficulty() {
        let easy = estimate_cues(300.0, 0.2);
        let hard = estimate_cues(300.0, 0.9);
        assert!(hard > easy);
    }

    #[test]
    fn difficulty_and_scene_type_are_stable() {
        let diff = estimate_scene_difficulty(30.0, 0.95);
        assert!(diff >= 0.5);
        assert_eq!(classify_scene_type(30.0, 0.95), "rapid-dialogue");
        assert_eq!(classify_scene_type(120.0, 0.1), "ambient");
    }
}
