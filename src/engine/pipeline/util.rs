use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn find_in_path(candidates: &[&str]) -> Option<PathBuf> {
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

pub(super) fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

// O(T²·n) naive scan; fine for cue-level token counts.
pub(super) fn has_repeated_ngram(tokens: &[String], n: usize) -> bool {
    if n < 2 || tokens.len() < n * 2 {
        return false;
    }
    for i in 0..=tokens.len() - n {
        for j in (i + n)..=tokens.len() - n {
            if tokens[i..i + n] == tokens[j..j + n] {
                return true;
            }
        }
    }
    false
}
