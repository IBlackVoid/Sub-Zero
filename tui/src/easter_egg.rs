use crate::secret;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Theme {
    #[default]
    Default,
    MrRobot,
    SynthwavePurple,
    AmberCrt,
    KeimaBlue,
}

impl Theme {
    pub fn label(&self) -> &'static str {
        match self {
            Theme::Default => "default",
            Theme::MrRobot => "MR.ROBOT",
            Theme::SynthwavePurple => "synthwave",
            Theme::AmberCrt => "amber CRT",
            Theme::KeimaBlue => "Keima blue",
        }
    }

    pub fn ident(&self) -> &'static str {
        match self {
            Theme::Default => "Default",
            Theme::MrRobot => "MrRobot",
            Theme::SynthwavePurple => "SynthwavePurple",
            Theme::AmberCrt => "AmberCrt",
            Theme::KeimaBlue => "KeimaBlue",
        }
    }

    pub fn from_ident(s: &str) -> Self {
        match s {
            "MrRobot" => Theme::MrRobot,
            "SynthwavePurple" => Theme::SynthwavePurple,
            "AmberCrt" => Theme::AmberCrt,
            "KeimaBlue" => Theme::KeimaBlue,
            _ => Theme::Default,
        }
    }

    pub fn next(self) -> Self {
        match self {
            Theme::Default => Theme::MrRobot,
            Theme::MrRobot => Theme::SynthwavePurple,
            Theme::SynthwavePurple => Theme::AmberCrt,
            Theme::AmberCrt => Theme::KeimaBlue,
            Theme::KeimaBlue => Theme::Default,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HiddenItem {
    pub label: String,
    pub kind: String,
    pub jsonl_index: Option<u32>,
    pub wav_index: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HiddenManifest {
    #[serde(default)]
    pub items: Vec<HiddenItem>,
}

#[derive(Debug)]
pub struct UnlockedSlot {
    pub dir: PathBuf,
    pub manifest: HiddenManifest,
    key: secret::SecretKey,
}

impl UnlockedSlot {
    pub fn decrypt_index(&self, index: u32) -> std::io::Result<Vec<u8>> {
        let path = self.dir.join(format!("{index}.bin"));
        secret::decrypt_asset_file(&path, &self.key)
    }
}

pub fn try_unlock(dir: &Path, phrase: &str) -> Option<UnlockedSlot> {
    let manifest_path = dir.join("manifest.bin");
    if !manifest_path.is_file() {
        return None;
    }
    let (bytes, key) = secret::decrypt_manifest_file(&manifest_path, phrase).ok()?;
    let manifest: HiddenManifest = serde_json::from_slice(&bytes).ok()?;
    Some(UnlockedSlot {
        dir: dir.to_path_buf(),
        manifest,
        key,
    })
}

#[derive(Debug, Default)]
pub struct EasterEgg {
    pub panel: Option<UnlockedSlot>,
    pub solo: Option<UnlockedSlot>,
    pub theme: Theme,
    pub boss: bool,
    pub panel_cursor: usize,
    pub panel_open: bool,
}

impl EasterEgg {
    pub fn new() -> Self {
        Self {
            theme: Theme::Default,
            ..Default::default()
        }
    }

    pub fn cycle_theme(&mut self) {
        self.theme = self.theme.next();
    }

    pub fn toggle_boss(&mut self) {
        self.boss = !self.boss;
    }

    #[cfg(test)]
    fn toggle_panel(&mut self) {
        if self.panel.is_some() {
            self.panel_open = !self.panel_open;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theme_cycle_returns_to_default() {
        let mut t = Theme::Default;
        for _ in 0..5 {
            t = t.next();
        }
        assert_eq!(t, Theme::Default);
    }

    #[test]
    fn each_theme_has_a_label() {
        let labels = [
            Theme::Default.label(),
            Theme::MrRobot.label(),
            Theme::SynthwavePurple.label(),
            Theme::AmberCrt.label(),
            Theme::KeimaBlue.label(),
        ];
        for l in labels {
            assert!(!l.is_empty());
        }
        let mut sorted = labels.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 5);
    }

    #[test]
    fn boss_toggles() {
        let mut e = EasterEgg::new();
        assert!(!e.boss);
        e.toggle_boss();
        assert!(e.boss);
        e.toggle_boss();
        assert!(!e.boss);
    }

    #[test]
    fn panel_toggle_requires_unlock() {
        let mut e = EasterEgg::new();
        e.toggle_panel();
        assert!(!e.panel_open);
    }
}
