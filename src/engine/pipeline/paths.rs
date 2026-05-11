use std::ffi::OsString;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};

pub(super) fn output_path_for_target_lang(
    input: &Path,
    target_lang: &str,
) -> Result<PathBuf, String> {
    let stem = input
        .file_stem()
        .ok_or_else(|| format!("invalid input filename: {}", input.display()))?;

    let mut file_name = OsString::from(stem);
    file_name.push(format!(".{target_lang}.srt"));
    Ok(input.with_file_name(file_name))
}

pub(super) fn metadata_sidecar_path(input: &Path) -> Result<PathBuf, String> {
    let stem = input
        .file_stem()
        .ok_or_else(|| format!("invalid input filename: {}", input.display()))?;
    let mut file_name = OsString::from(stem);
    file_name.push(".sub-zero.json");
    Ok(input.with_file_name(file_name))
}

pub(super) fn trace_sidecar_path(input: &Path) -> Result<PathBuf, String> {
    let stem = input
        .file_stem()
        .ok_or_else(|| format!("invalid input filename: {}", input.display()))?;
    let mut file_name = OsString::from(stem);
    file_name.push(".sub-zero.trace.json");
    Ok(input.with_file_name(file_name))
}

pub(super) fn is_srt_path(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|ext| ext.eq_ignore_ascii_case("srt"))
        .unwrap_or(false)
}

pub(super) fn is_video_sidecar_source(input: &Path, source_srt: &Path) -> bool {
    !is_srt_path(input) && source_srt == input.with_extension("srt")
}

pub(super) fn looks_like_simulated_placeholder_srt(path: &Path) -> bool {
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = vec![0u8; 4096];
    let Ok(bytes_read) = file.read(&mut buf) else {
        return false;
    };
    buf.truncate(bytes_read);
    let Ok(prefix) = std::str::from_utf8(&buf) else {
        return false;
    };
    let lower = prefix.to_ascii_lowercase();
    lower.contains("(simulated) subtitle #") || lower.contains("ai processing...")
}

pub(super) fn checkpoint_dir_for(video: &Path) -> Result<PathBuf, String> {
    let video_key = video.canonicalize().unwrap_or_else(|_| video.to_path_buf());
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    video_key.hash(&mut hasher);
    let run_hash = format!("{:016x}", hasher.finish());

    let base = if let Some(home) = std::env::var_os("SUB_ZERO_HOME") {
        PathBuf::from(home)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".sub-zero")
    } else if let Some(home) = std::env::var_os("USERPROFILE") {
        PathBuf::from(home).join(".sub-zero")
    } else {
        std::env::temp_dir().join(".sub-zero")
    };

    let dir = base.join("checkpoints").join(run_hash);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(dir)
}
