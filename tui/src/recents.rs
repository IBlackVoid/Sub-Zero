//! Persistent recent-files list for the Picker screen.
//!
//! Stored at `~/.sub-zero/tui_recents.json`. Atomic writes via temp +
//! rename so a crash mid-write never corrupts the file.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

const MAX_RECENTS: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recent {
    pub path: PathBuf,
    pub last_used_epoch: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Recents {
    #[serde(default)]
    pub entries: Vec<Recent>,
}

impl Recents {
    pub fn load() -> Self {
        match Self::path() {
            Some(p) if p.is_file() => std::fs::read_to_string(&p)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default(),
            _ => Self::default(),
        }
    }

    pub fn record(&mut self, path: &Path) {
        if !path.exists() {
            return;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let normalised = path.to_path_buf();
        self.entries.retain(|e| e.path != normalised);
        self.entries.insert(
            0,
            Recent {
                path: normalised,
                last_used_epoch: now,
            },
        );
        if self.entries.len() > MAX_RECENTS {
            self.entries.truncate(MAX_RECENTS);
        }
        let _ = self.save();
    }

    fn save(&self) -> std::io::Result<()> {
        let Some(p) = Self::path() else {
            return Ok(());
        };
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = p.with_extension("json.tmp");
        let s = serde_json::to_string_pretty(self).unwrap_or_default();
        std::fs::write(&tmp, s)?;
        std::fs::rename(&tmp, &p)?;
        Ok(())
    }

    fn path() -> Option<PathBuf> {
        if let Some(home) = std::env::var_os("SUB_ZERO_HOME") {
            return Some(PathBuf::from(home).join("tui_recents.json"));
        }
        let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
        Some(
            PathBuf::from(home)
                .join(".sub-zero")
                .join("tui_recents.json"),
        )
    }
}
