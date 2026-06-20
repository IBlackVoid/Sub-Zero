use std::path::{Path, PathBuf};

use crate::engine::EngineConfig;

/// One-time, best-effort migration of the legacy `~/.sub-zero` state
/// directory to the rebranded `~/.voidex` location.
///
/// The TUI is a separate crate from the engine library and stores its own
/// preferences, recents, and secret envelopes under the same home dir, so
/// it carries its own copy of this migration (kept in lock-step with
/// `voidex::migrate_legacy_home` in the engine crate). Resolution
/// precedence — `VOIDEX_HOME` override, else `HOME`, else `USERPROFILE`,
/// else temp — matches the engine. Idempotent; any I/O error is ignored so
/// a failed migration never blocks the UI.
pub fn migrate_legacy_home() {
    if std::env::var_os("VOIDEX_HOME").is_some() {
        return;
    }
    let parent = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);

    let legacy = parent.join(".sub-zero");
    let current = parent.join(".voidex");
    if legacy.is_dir() && !current.exists() {
        let _ = std::fs::rename(&legacy, &current);
    }
}

pub fn canonical_output_path(input: &Path, config: &EngineConfig) -> Option<PathBuf> {
    // VoiDex canonical output naming: <stem>.<target>.srt next to input.
    let parent = input.parent()?;
    let stem = input.file_stem()?.to_str()?;
    let lang = if config.target_lang.is_empty() {
        "en"
    } else {
        &config.target_lang
    };
    Some(parent.join(format!("{stem}.{lang}.srt")))
}

pub fn read_srt_preview(path: &Path, max_lines: usize) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .take(max_lines)
        .map(|s| s.to_string())
        .collect()
}

/// Pull the last `max_cues` cues from an SRT file by parsing only the
/// tail of the file. Cheap to call on every tick because the read is
/// capped at the last 32 KiB.
pub fn tail_srt_cues(path: &Path, max_cues: usize) -> Vec<(u32, String)> {
    let Ok(mut file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    use std::io::{Read, Seek, SeekFrom};
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let tail_window: u64 = 32 * 1024;
    let start = len.saturating_sub(tail_window);
    if file.seek(SeekFrom::Start(start)).is_err() {
        return Vec::new();
    }
    let mut buf = String::new();
    if file.read_to_string(&mut buf).is_err() {
        // SRT files can contain non-UTF-8 bytes during mid-write; fall
        // back to lossy decode rather than freezing the UI preview.
        let mut bytes = Vec::new();
        if std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .is_err()
        {
            return Vec::new();
        }
        buf = String::from_utf8_lossy(&bytes).into_owned();
    }

    let mut cues: Vec<(u32, String)> = Vec::new();
    for block in buf.split("\n\n") {
        let lines: Vec<&str> = block.lines().collect();
        if lines.len() < 3 {
            continue;
        }
        let idx: u32 = match lines[0].trim().parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let text = lines[2..].join(" ").trim().to_string();
        if !text.is_empty() {
            cues.push((idx, text));
        }
    }
    if cues.len() > max_cues {
        cues.split_off(cues.len() - max_cues)
    } else {
        cues
    }
}

pub fn open_directory(dir: &Path) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("explorer").arg(dir).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(dir).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
    }
}

pub fn sanitize_path_input(raw: &str) -> String {
    let trimmed = raw.trim();
    let stripped = trimmed
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| {
            trimmed
                .strip_prefix('\'')
                .and_then(|s| s.strip_suffix('\''))
        })
        .unwrap_or(trimmed);
    stripped.to_string()
}

/// Where `:save` / `:snap` are allowed to write. Defaults to the
/// process's current working directory; overridden by the
/// `VOIDEX_TUI_WRITE_ROOT` environment variable when it points at an
/// existing directory.
///
/// The policy is intentionally per-process and non-persistent — it
/// reflects "where the user launched me from" rather than a stored
/// preference. Operators who want a hard write boundary should launch
/// the TUI with the env var explicitly set.
pub fn tui_write_root() -> PathBuf {
    if let Some(env_root) = std::env::var_os("VOIDEX_TUI_WRITE_ROOT") {
        let p = PathBuf::from(env_root);
        if p.is_dir() {
            return p;
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Resolve a user-supplied `:save` / `:snap` target against the TUI
/// write root, rejecting traversal and out-of-root writes.
///
/// Policy:
/// - `..` anywhere in the path → rejected.
/// - Absolute paths must be inside `root` (lexical `starts_with`).
/// - Relative paths are joined onto `root`.
///
/// This does *not* canonicalise — the destination directory may not yet
/// exist. The check is lexical, which is sufficient for an
/// already-loopback, single-user TUI: it stops accidental clobbers of
/// `/etc/passwd` and the obvious traversal payloads in remote-event
/// strings, without paying the syscall for resolving every component.
pub fn resolve_write_target(target: &str, root: &Path) -> Result<PathBuf, String> {
    let trimmed = target.trim();
    if trimmed.is_empty() {
        return Err("path required".to_string());
    }
    let target_path = Path::new(trimmed);
    for comp in target_path.components() {
        if matches!(comp, std::path::Component::ParentDir) {
            return Err(format!(
                "path '{}' contains '..' which is not allowed",
                target_path.display()
            ));
        }
    }
    let candidate = if target_path.is_absolute() {
        target_path.to_path_buf()
    } else {
        root.join(target_path)
    };
    if !candidate.starts_with(root) {
        return Err(format!(
            "path '{}' is outside the write root '{}'; set VOIDEX_TUI_WRITE_ROOT to override",
            candidate.display(),
            root.display()
        ));
    }
    Ok(candidate)
}

pub fn complete_path(input: &str) -> Option<String> {
    let p = Path::new(input);
    if p.is_dir() && !input.ends_with(['/', '\\']) {
        let mut out = input.to_string();
        out.push(std::path::MAIN_SEPARATOR);
        return Some(out);
    }
    let (dir, prefix) = match p.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => {
            let base = p
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            (parent.to_path_buf(), base)
        }
        _ => (PathBuf::from("."), input.to_string()),
    };
    let entries = std::fs::read_dir(&dir).ok()?;
    let mut matches: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with(&prefix) {
            matches.push(name);
        }
    }
    if matches.is_empty() {
        return None;
    }
    let common = longest_common_prefix(&matches);
    if common.len() <= prefix.len() {
        return None;
    }
    let mut completed = if dir.as_os_str() == std::ffi::OsStr::new(".") {
        String::new()
    } else {
        let mut s = dir.to_string_lossy().to_string();
        if !s.ends_with(['/', '\\']) {
            s.push(std::path::MAIN_SEPARATOR);
        }
        s
    };
    completed.push_str(&common);
    if matches.len() == 1 {
        let full = dir.join(&common);
        if full.is_dir() {
            completed.push(std::path::MAIN_SEPARATOR);
        }
    }
    Some(completed)
}

fn longest_common_prefix(strs: &[String]) -> String {
    if strs.is_empty() {
        return String::new();
    }
    let mut prefix = strs[0].clone();
    for s in strs.iter().skip(1) {
        let new_len = prefix
            .chars()
            .zip(s.chars())
            .take_while(|(a, b)| a == b)
            .count();
        prefix.truncate(
            prefix
                .char_indices()
                .nth(new_len)
                .map(|(i, _)| i)
                .unwrap_or(prefix.len()),
        );
    }
    prefix
}

#[cfg(test)]
mod tests {
    use super::resolve_write_target;
    use std::path::PathBuf;

    fn root() -> PathBuf {
        // Use the OS temp dir as a stable root that exists on every CI box.
        std::env::temp_dir()
    }

    #[test]
    fn relative_paths_land_under_root() {
        let r = root();
        let resolved = resolve_write_target("snap.txt", &r).unwrap();
        assert_eq!(resolved, r.join("snap.txt"));
    }

    #[test]
    fn parent_dir_traversal_is_rejected() {
        let r = root();
        for bad in ["..", "../foo", "sub/../../etc/passwd", "..\\foo"] {
            assert!(
                resolve_write_target(bad, &r).is_err(),
                "should reject: {bad}"
            );
        }
    }

    #[test]
    fn absolute_paths_outside_root_are_rejected() {
        let r = root();
        // Pick an absolute path that's almost certainly outside r.
        #[cfg(windows)]
        let outside = "C:\\Windows\\System32\\evil.txt";
        #[cfg(not(windows))]
        let outside = "/etc/evil.txt";
        // If temp dir somehow contains it, the test would falsely pass —
        // assert the precondition.
        assert!(
            !PathBuf::from(outside).starts_with(&r),
            "test invariant: outside path must not be inside root"
        );
        assert!(resolve_write_target(outside, &r).is_err());
    }

    #[test]
    fn absolute_paths_inside_root_are_allowed() {
        let r = root();
        let inside = r.join("nested").join("output.srt");
        let resolved = resolve_write_target(inside.to_str().unwrap(), &r).unwrap();
        assert!(resolved.starts_with(&r));
    }

    #[test]
    fn empty_target_is_rejected() {
        let r = root();
        assert!(resolve_write_target("", &r).is_err());
        assert!(resolve_write_target("   ", &r).is_err());
    }
}
