use super::find_in_path;
use crate::engine::media;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(super) fn create_temp_rescue_dir(scene_start: usize) -> Result<PathBuf, String> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("voidex_source_scene_rescue_{scene_start}_{stamp}"));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(dir)
}

pub(super) fn extract_audio_segment_to_wav(
    input: &Path,
    wav_out: &Path,
    start: f64,
    end: f64,
) -> Result<(), String> {
    if end <= start + 0.05 {
        return Err(format!(
            "invalid rescue segment: start={start:.3} end={end:.3}"
        ));
    }
    let ffmpeg = find_in_path(&["ffmpeg", "ffmpeg.exe"])
        .ok_or_else(|| "ffmpeg not found in PATH (required for source rescue)".to_string())?;
    let status = Command::new(ffmpeg)
        .arg("-hide_banner")
        .arg("-nostdin")
        .arg("-v")
        .arg("error")
        .arg("-y")
        .arg("-ss")
        .arg(format!("{start:.3}"))
        .arg("-to")
        .arg(format!("{end:.3}"))
        .arg("-i")
        .arg(input)
        .arg("-vn")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-f")
        .arg("wav")
        .arg(wav_out)
        .status()
        .map_err(|e| format!("failed to spawn ffmpeg for source rescue: {e}"))?;
    if !status.success() {
        return Err(format!("ffmpeg source rescue failed with status: {status}"));
    }
    Ok(())
}

pub(super) fn extract_audio_to_wav(video: &Path, wav_out: &Path) -> Result<(), String> {
    media::extract_audio_to_wav(video, wav_out, None, None)
}

pub(super) fn extract_audio_to_wav_with_selection(
    video: &Path,
    wav_out: &Path,
    audio_stream_index: Option<usize>,
    audio_lang: Option<&str>,
) -> Result<(), String> {
    media::extract_audio_to_wav(video, wav_out, audio_stream_index, audio_lang)
}
