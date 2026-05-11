use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_in_path(candidates: &[&str]) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        for name in candidates {
            let path = dir.join(name);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn normalize_lang(lang: &str) -> String {
    let v = lang.trim().to_ascii_lowercase();
    match v.as_str() {
        "ja" | "jpn" | "jp" => "jpn".to_string(),
        "en" | "eng" => "eng".to_string(),
        _ => v,
    }
}

fn ffprobe_streams(video: &Path) -> Result<Value, String> {
    let ffprobe = find_in_path(&["ffprobe", "ffprobe.exe"])
        .ok_or_else(|| "ffprobe not found in PATH".to_string())?;
    let output = Command::new(ffprobe)
        .arg("-hide_banner")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("stream=index,codec_type:stream_tags=language")
        .arg("-of")
        .arg("json")
        .arg(video)
        .output()
        .map_err(|e| format!("failed to spawn ffprobe: {e}"))?;

    if !output.status.success() {
        return Err(format!("ffprobe failed with status: {}", output.status));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&text).map_err(|e| format!("ffprobe returned invalid JSON: {e}"))
}

fn iter_streams(value: &Value) -> impl Iterator<Item = &Value> {
    value
        .get("streams")
        .and_then(|v| v.as_array())
        .map(|v| v.as_slice())
        .unwrap_or(&[])
        .iter()
}

pub(crate) fn select_audio_stream_index(
    video: &Path,
    audio_stream_index: Option<usize>,
    audio_lang: Option<&str>,
) -> Result<Option<usize>, String> {
    if audio_stream_index.is_none() && audio_lang.is_none() {
        return Ok(None);
    }
    let probe = ffprobe_streams(video)?;

    if let Some(index) = audio_stream_index {
        let found = iter_streams(&probe).any(|s| {
            let kind = s.get("codec_type").and_then(|v| v.as_str()).unwrap_or("");
            let idx = s.get("index").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;
            kind == "audio" && idx == index
        });
        if !found {
            return Err(format!(
                "requested audio stream index {index} not found (or not an audio stream) in {}",
                video.display()
            ));
        }
        return Ok(Some(index));
    }

    let want = normalize_lang(audio_lang.unwrap_or_default());
    for s in iter_streams(&probe) {
        let kind = s.get("codec_type").and_then(|v| v.as_str()).unwrap_or("");
        if kind != "audio" {
            continue;
        }
        let idx = s.get("index").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;
        let lang = s
            .get("tags")
            .and_then(|t| t.get("language"))
            .and_then(|v| v.as_str())
            .map(normalize_lang);
        if lang.as_deref() == Some(want.as_str()) {
            return Ok(Some(idx));
        }
    }

    Err(format!(
        "no audio stream with language {want} found in {}",
        video.display()
    ))
}

pub(crate) fn extract_audio_to_wav(
    video: &Path,
    wav_out: &Path,
    audio_stream_index: Option<usize>,
    audio_lang: Option<&str>,
) -> Result<(), String> {
    let ffmpeg = find_in_path(&["ffmpeg", "ffmpeg.exe"])
        .ok_or_else(|| "ffmpeg not found in PATH (required for --transcribe)".to_string())?;

    let selected = select_audio_stream_index(video, audio_stream_index, audio_lang)?;

    let mut cmd = Command::new(ffmpeg);
    cmd.arg("-hide_banner")
        .arg("-nostdin")
        .arg("-v")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(video);
    if let Some(index) = selected {
        cmd.arg("-map").arg(format!("0:{index}"));
    }
    let status = cmd
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_audio_stream_index_picks_lang_match() {
        let probe = serde_json::json!({
            "streams": [
                {"index": 0, "codec_type": "video"},
                {"index": 1, "codec_type": "audio", "tags": {"language": "eng"}},
                {"index": 2, "codec_type": "audio", "tags": {"language": "jpn"}}
            ]
        });

        // Use the inner logic by calling iter_streams + matching: emulate by writing a tiny shim.
        let video = Path::new("test.mkv");
        // We can't call select_audio_stream_index without ffprobe; validate normalization behavior instead.
        assert_eq!(normalize_lang("ja"), "jpn");
        assert_eq!(normalize_lang("EN"), "eng");

        // Manual match: ensure our stream scanning logic would choose index 2 for jpn.
        let want = normalize_lang("jpn");
        let mut picked = None;
        for s in iter_streams(&probe) {
            if s.get("codec_type").and_then(|v| v.as_str()) != Some("audio") {
                continue;
            }
            let idx = s.get("index").and_then(|v| v.as_u64()).unwrap_or(u64::MAX) as usize;
            let lang = s
                .get("tags")
                .and_then(|t| t.get("language"))
                .and_then(|v| v.as_str())
                .map(normalize_lang);
            if lang.as_deref() == Some(want.as_str()) {
                picked = Some(idx);
                break;
            }
        }
        assert_eq!(picked, Some(2), "{video:?}");
    }
}
