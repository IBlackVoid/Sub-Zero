//! voidex-tui — entry point.

mod accessibility;
mod animation;
mod app;
mod audio;
mod easter_egg;
mod engine;
mod paths;
mod prefs;
mod recents;
mod runner_viz;
mod secret;
mod slots;
mod splash;
mod theme;
mod voice_view;
mod waveform;

use std::process::ExitCode;

fn main() -> ExitCode {
    // Rebrand carry-over: move any legacy `~/.sub-zero` state to `~/.voidex`
    // before prefs/recents/secrets are read. Best-effort and idempotent.
    paths::migrate_legacy_home();

    match app::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("voidex-tui: {err}");
            ExitCode::FAILURE
        }
    }
}
