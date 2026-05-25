//! Audio playback for video animations.
//!
//! Spawns a `rodio` output stream + sink the first time the user
//! switches to an animation that has a sibling WAV. Switching to an
//! animation without audio (a GIF or static image) stops playback
//! cleanly. Volume and mute are decoupled — muting preserves the
//! current volume so unmute restores it.
//!
//! All audio failures are non-fatal. If the audio device is missing,
//! the WAV is unreadable, or rodio cannot be initialised, the TUI
//! continues to render visuals and disables the volume controls.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

pub struct AudioPlayer {
    /// `_stream` must outlive `sink` — its destruction tears down the
    /// audio device. We hold it as `_` to suppress the dead-code lint.
    _stream: rodio::OutputStream,
    handle: rodio::OutputStreamHandle,
    sink: Option<rodio::Sink>,
    /// Track which file we're currently playing so duplicate switches
    /// to the same animation don't restart it.
    current_path: Option<PathBuf>,
    volume: f32,
    muted: bool,
}

impl AudioPlayer {
    /// Try to construct a player. Returns `None` when no audio device
    /// is available — caller treats absence as "audio just isn't a
    /// thing on this machine" and disables volume keybinds.
    pub fn new() -> Option<Self> {
        let (stream, handle) = rodio::OutputStream::try_default().ok()?;
        Some(Self {
            _stream: stream,
            handle,
            sink: None,
            current_path: None,
            volume: 0.6,
            muted: false,
        })
    }

    /// Switch to playing `path`. If we're already playing it, this is
    /// a no-op. Stops any prior track. Caller owns the decision of
    /// when to call this — typically right after a state switch.
    pub fn play(&mut self, path: &Path) {
        if self.current_path.as_deref() == Some(path) && self.is_playing() {
            return;
        }
        self.stop();
        let Ok(file) = File::open(path) else {
            return;
        };
        let Ok(decoder) = rodio::Decoder::new(BufReader::new(file)) else {
            return;
        };
        let Ok(sink) = rodio::Sink::try_new(&self.handle) else {
            return;
        };
        sink.set_volume(if self.muted { 0.0 } else { self.volume });
        sink.append(rodio::source::Source::repeat_infinite(decoder));
        self.sink = Some(sink);
        self.current_path = Some(path.to_path_buf());
    }

    pub fn stop(&mut self) {
        if let Some(sink) = self.sink.take() {
            sink.stop();
        }
        self.current_path = None;
    }

    pub fn is_playing(&self) -> bool {
        self.sink.as_ref().map(|s| !s.empty()).unwrap_or(false)
    }

    pub fn volume(&self) -> f32 {
        self.volume
    }

    pub fn is_muted(&self) -> bool {
        self.muted
    }

    pub fn set_volume(&mut self, v: f32) {
        self.volume = v.clamp(0.0, 1.0);
        if let Some(s) = self.sink.as_ref() {
            s.set_volume(if self.muted { 0.0 } else { self.volume });
        }
    }

    pub fn nudge_volume(&mut self, delta: f32) {
        self.set_volume(self.volume + delta);
    }

    pub fn toggle_mute(&mut self) {
        self.muted = !self.muted;
        if let Some(s) = self.sink.as_ref() {
            s.set_volume(if self.muted { 0.0 } else { self.volume });
        }
    }
}
