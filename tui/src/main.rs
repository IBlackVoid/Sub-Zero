//! sub-zero-tui — entry point.

mod animation;
mod app;
mod audio;
mod digests;
mod easter_egg;
mod engine;
mod prefs;
mod recents;
mod secret;
mod runner_viz;
mod splash;
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
