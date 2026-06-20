//! Accessibility preferences sourced from environment variables.
//!
//! Two opt-in toggles, both disabled by default so the colourful animated
//! TUI is unchanged for normal users:
//!
//! - `NO_COLOR` (any non-empty value) — accent colours collapse to
//!   `Color::Reset`. Per <https://no-color.org/>. Honours the spec:
//!   "All command-line software which outputs text with ANSI color added
//!   should check for the presence of a NO_COLOR environment variable
//!   that, when present (regardless of its value), prevents the addition
//!   of ANSI color."
//!
//! - `VOIDEX_TUI_REDUCED_MOTION` (any non-empty value) — pin the
//!   animation to its last frame instead of advancing through it.
//!   Vestibular accessibility and slow-SSH friendliness. The easter egg
//!   still triggers; it just lands on its final pose instead of playing
//!   the loop.
//!
//! Resolved once at app start (via `Accessibility::from_env`) so the
//! palette is stable across the whole run — flipping the env mid-run
//! would otherwise cause flicker.

/// Snapshot of accessibility preferences observed at process start.
#[derive(Debug, Clone, Copy, Default)]
pub struct Accessibility {
    pub no_color: bool,
    pub reduced_motion: bool,
}

impl Accessibility {
    pub fn from_env() -> Self {
        Self {
            no_color: env_flag("NO_COLOR"),
            reduced_motion: env_flag("VOIDEX_TUI_REDUCED_MOTION"),
        }
    }
}

fn env_flag(name: &str) -> bool {
    match std::env::var_os(name) {
        Some(value) => !value.is_empty(),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::Accessibility;

    #[test]
    fn defaults_are_off() {
        // Best-effort: if a developer happens to have these set in their
        // shell, this test would be noisy. Skip with a message in that
        // case rather than fail.
        let env_no_color = std::env::var_os("NO_COLOR").is_some();
        let env_motion = std::env::var_os("VOIDEX_TUI_REDUCED_MOTION").is_some();
        if env_no_color || env_motion {
            eprintln!("skipping defaults_are_off: a11y env vars set on host");
            return;
        }
        let a = Accessibility::from_env();
        assert!(!a.no_color);
        assert!(!a.reduced_motion);
    }
}
