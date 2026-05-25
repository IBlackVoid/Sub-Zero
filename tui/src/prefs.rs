use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prefs {
    #[serde(default = "default_theme_name")]
    pub theme: String,
    #[serde(default)]
    pub auto_theme_by_hour: bool,
}

fn default_theme_name() -> String {
    "Default".to_string()
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            theme: default_theme_name(),
            auto_theme_by_hour: false,
        }
    }
}

impl Prefs {
    pub fn path() -> Option<PathBuf> {
        if let Some(home) = std::env::var_os("SUB_ZERO_HOME") {
            return Some(PathBuf::from(home).join("tui.json"));
        }
        let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
        Some(PathBuf::from(home).join(".sub-zero").join("tui.json"))
    }

    pub fn load() -> Self {
        match Self::path() {
            Some(p) if p.is_file() => std::fs::read_to_string(&p)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default(),
            _ => Self::default(),
        }
    }

    pub fn save(&self) -> std::io::Result<()> {
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
}

pub fn local_hour_24() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let offset = tz_offset_seconds();
    let local = secs + offset;
    let day_seconds = local.rem_euclid(86_400);
    (day_seconds / 3600) as u32
}

fn tz_offset_seconds() -> i64 {
    if let Ok(s) = std::env::var("SUB_ZERO_TZ_OFFSET_HOURS") {
        if let Ok(h) = s.trim().parse::<i64>() {
            return h * 3600;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sensible() {
        let p = Prefs::default();
        assert_eq!(p.theme, "Default");
        assert!(!p.auto_theme_by_hour);
    }

    #[test]
    fn round_trip_serializes_cleanly() {
        let p = Prefs {
            theme: "MrRobot".to_string(),
            auto_theme_by_hour: true,
        };
        let s = serde_json::to_string(&p).unwrap();
        let p2: Prefs = serde_json::from_str(&s).unwrap();
        assert_eq!(p2.theme, "MrRobot");
        assert!(p2.auto_theme_by_hour);
    }

    #[test]
    fn unknown_theme_falls_back_at_runtime() {
        let s = r#"{"theme":"DoesNotExist"}"#;
        let p: Prefs = serde_json::from_str(s).unwrap();
        assert_eq!(p.theme, "DoesNotExist");
    }

    #[test]
    fn local_hour_in_valid_range() {
        let h = local_hour_24();
        assert!(h < 24);
    }
}
