//! sub-zero-tui — entry point.

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
    match app::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("sub-zero-tui: {err}");
            ExitCode::FAILURE
        }
    }
}
