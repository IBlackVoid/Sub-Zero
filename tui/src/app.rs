//! voidex-tui — screen state machine.
//!
//! Screens: Greeting → Picker → Running → Result → (back to Greeting).
//! Layout/visual grammar locked in `docs/TUI_DESIGN.md`. Every screen
//! reuses the same chrome: title bar at top, hint bar at bottom, the
//! braille animation as the centre of gravity wherever it appears.

use std::collections::HashMap;
use std::io::{self};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
    Terminal,
};

use crate::animation::{Animation, AnimationError};
use crate::audio::AudioPlayer;
use crate::easter_egg::{self, EasterEgg};
use crate::engine::{EngineConfig, EngineRunner};
use crate::paths::{
    canonical_output_path, read_srt_preview, resolve_write_target, sanitize_path_input,
    tail_srt_cues, tui_write_root,
};
use crate::recents::Recents;
use crate::slots;
use crate::theme::{brighten, lerp_color, palette, theme_accent};

mod input;

const DEFAULT_ASSET_DIR: &str = "assets/ascii";
const STATES: &[&str] = &["idle", "running", "victory", "warning", "error", "complete"];
/// Animation shown on first launch / when returning to Greeting.
/// User asked for the second slot (running = anime-haruhi.gif).
const GREETING_DEFAULT_STATE: &str = "running";

const LANGS: &[&str] = &["ja", "en", "ko", "zh", "es", "fr", "de", "ru", "ar", "auto"];
const PROFILES: &[&str] = &["fast", "balanced", "strict"];

/// Top-level screen the app is currently showing.
enum Screen {
    Greeting,
    Picker(PickerState),
    Configure(ConfigureState),
    Running(Box<RunningState>),
    Result(ResultState),
}

struct PickerState {
    input: String,
    error: Option<String>,
    selected_recent: Option<usize>, // None = input field is active
}

#[derive(Clone)]
struct ConfigureState {
    input_path: PathBuf,
    config: EngineConfig,
    cursor: ConfigField,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConfigField {
    SourceLang,
    TargetLang,
    Profile,
    Gpu,
    Workers,
    Start,
}

impl ConfigField {
    fn next(self) -> Self {
        match self {
            Self::SourceLang => Self::TargetLang,
            Self::TargetLang => Self::Profile,
            Self::Profile => Self::Gpu,
            Self::Gpu => Self::Workers,
            Self::Workers => Self::Start,
            Self::Start => Self::SourceLang,
        }
    }
    fn prev(self) -> Self {
        match self {
            Self::SourceLang => Self::Start,
            Self::TargetLang => Self::SourceLang,
            Self::Profile => Self::TargetLang,
            Self::Gpu => Self::Profile,
            Self::Workers => Self::Gpu,
            Self::Start => Self::Workers,
        }
    }
}

struct RunningState {
    runner: EngineRunner,
    config: EngineConfig,
}

struct ResultState {
    input: PathBuf,
    config: EngineConfig,
    success: bool,
    duration: Duration,
    output_files: Vec<PathBuf>,
    last_message: Option<String>,
    chunk_total: u32,
    quality: Option<f64>,
    srt_preview: Vec<String>,
}

/// Cross-screen state — the animation pack is loaded once at startup
/// and reused everywhere.
struct App {
    pack: HashMap<String, Animation>,
    current_state_name: String,
    frame_idx: usize,
    last_advance: Instant,
    /// Wall-clock time the current animation started looping. Used to
    /// drive frame selection from the audio playhead when the active
    /// animation has a sound track — keeps audio and visuals in sync.
    loop_started: Instant,
    screen: Screen,
    recents: Recents,
    last_config: EngineConfig,
    audio: Option<AudioPlayer>,
    /// Hidden-content state. Always present; methods inside check
    /// `panel`/`solo` for "is this slot unlocked yet?".
    easter: EasterEgg,
    /// `Some(buffer)` when the vim-style ex-line is open at the bottom
    /// of the screen accepting a secret phrase. `None` otherwise.
    exline: Option<String>,
    /// Scratch directory holding decrypted assets for unlocked slots.
    /// One subdir per slot; created lazily on first unlock and
    /// removed on Drop so plaintext never lingers across runs.
    scratch_dir: Option<PathBuf>,
    /// True once the user has explicitly chosen an animation (via Tab,
    /// 1-7, panel pick, or solo unlock). While set, the engine's
    /// state-machine-driven `switch_anim` on the Running screen is
    /// suppressed so the user's pick actually persists. Without this,
    /// the engine would slam the animation back to "running" /
    /// "victory" / etc. every poll tick.
    anim_override: bool,
    /// Persisted preferences (theme, auto-by-hour). Loaded at startup,
    /// rewritten on theme change.
    prefs: crate::prefs::Prefs,
    /// Splash logo gates the first frames at startup. `None` once the
    /// timer has expired and the App has switched to normal rendering.
    splash_until: Option<Instant>,
    /// Wall-clock of the last theme change. Used to drive the
    /// brief border ripple animation.
    theme_changed_at: Option<Instant>,
    /// Wall-clock of the last panel open. Used to drive the slide-in
    /// width animation.
    panel_opened_at: Option<Instant>,
    /// `:help` overlay visibility. Any key dismisses it.
    help_overlay: bool,
    /// Index → last-seen timestamp for cues in the preview pane.
    /// New cues get a brief accent flash; existing ones render
    /// normally.
    new_cue_arrivals: std::collections::HashMap<u32, Instant>,
    /// Loaded envelope for the currently playing audio file (if any).
    /// Refreshed by `sync_audio_to_animation` so the waveform display
    /// always reflects the active track.
    audio_envelope: Option<crate::waveform::Envelope>,
    /// Cached cues from the tail of the engine's output SRT. Refreshed
    /// once per tick on the Running screen so the renderer can stay
    /// immutable.
    cue_preview_cache: Vec<(u32, String)>,
    /// Cached per-speaker voice priors loaded from
    /// `~/.voidex/voice_priors.json`. Reloaded periodically so the
    /// signature display picks up new speakers as the engine learns
    /// them.
    voice_priors_cache: Vec<crate::voice_view::SpeakerSignature>,
    /// Last time `voice_priors_cache` was refreshed from disk. We
    /// throttle reads to once every couple of seconds; the priors
    /// file is small but we still don't want to thrash on every tick.
    voice_priors_refreshed_at: Option<Instant>,
    /// Active Running-screen visualizer mode plus its internal state.
    /// Cycled by the `g` key while a translation is in progress.
    viz: crate::runner_viz::Viz,
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(p) = self.scratch_dir.as_ref() {
            let _ = std::fs::remove_dir_all(p);
        }
    }
}

impl App {
    fn switch_anim(&mut self, name: &str) {
        if !self.pack.contains_key(name) {
            return;
        }
        if name != self.current_state_name {
            self.current_state_name = name.to_string();
            self.frame_idx = 0;
            self.last_advance = Instant::now();
            self.loop_started = Instant::now();
            self.sync_audio_to_animation();
        }
    }

    /// Manual animation pick. Same as `switch_anim` but locks the
    /// override flag so the engine's Running-screen poll loop stops
    /// fighting the user for control of the animation slot. Use this
    /// for Tab cycle, 1-7 preview, panel pick, and solo unlock.
    fn force_anim(&mut self, name: &str) {
        self.switch_anim(name);
        self.anim_override = true;
    }

    /// Drop the override so the engine state machine resumes driving
    /// the animation. Wired to `Esc` on the Greeting screen so users
    /// have an obvious "back to auto" path.
    fn release_anim_override(&mut self) {
        self.anim_override = false;
    }

    /// Start, switch, or stop audio so it matches the active
    /// animation's audio sidecar (if any). Also loads the
    /// pre-computed amplitude envelope for the waveform display.
    fn sync_audio_to_animation(&mut self) {
        let Some(audio) = self.audio.as_mut() else {
            self.audio_envelope = None;
            return;
        };
        let Some(anim) = self.pack.get(&self.current_state_name) else {
            return;
        };
        match &anim.audio_path {
            Some(p) => {
                audio.play(p);
                self.audio_envelope = crate::waveform::Envelope::for_wav(p);
            }
            None => {
                audio.stop();
                self.audio_envelope = None;
            }
        }
    }

    fn current_anim_has_audio(&self) -> bool {
        self.pack
            .get(&self.current_state_name)
            .and_then(|a| a.audio_path.as_ref())
            .is_some()
    }
    fn cycle_anim(&mut self, delta: i32) {
        let cur = STATES
            .iter()
            .position(|s| *s == self.current_state_name)
            .unwrap_or(0);
        let n = STATES.len() as i32;
        let next = ((cur as i32 + delta).rem_euclid(n)) as usize;
        let target = STATES[next].to_string();
        // Tab/BackTab cycle is a manual pick — lock the override.
        self.force_anim(&target);
    }
    fn current_anim(&self) -> &Animation {
        self.pack
            .get(&self.current_state_name)
            .or_else(|| self.pack.values().next())
            .expect("at least one animation must be loaded")
    }
    fn advance_frame_if_due(&mut self) {
        let anim = self.current_anim();
        if anim.frames.is_empty() {
            return;
        }
        // Reduced-motion: pin to the last frame and stop advancing. The
        // easter-egg trigger still fires; it just lands on its final
        // pose instead of playing the loop. See `accessibility.rs`.
        if crate::accessibility::Accessibility::from_env().reduced_motion {
            let last = anim.frames.len() - 1;
            if self.frame_idx != last {
                self.frame_idx = last;
                self.last_advance = Instant::now();
            }
            return;
        }
        // Drive frame selection by wall-clock elapsed since the loop
        // started. This naturally syncs the visuals to the audio
        // sink — both run on real time, neither blocks the other —
        // and recovers gracefully if a redraw skipped a tick.
        let total_ms = anim.total_duration_ms.max(1);
        let elapsed_ms = (self.loop_started.elapsed().as_millis() as u64) % total_ms;
        let mut acc = 0u64;
        let mut idx = 0usize;
        for (i, frame) in anim.frames.iter().enumerate() {
            let d = if frame.delay_ms == 0 {
                80
            } else {
                frame.delay_ms as u64
            };
            acc += d;
            if acc > elapsed_ms {
                idx = i;
                break;
            }
            idx = i;
        }
        if idx != self.frame_idx {
            self.frame_idx = idx;
            self.last_advance = Instant::now();
        }
    }

    /// Lazily create the per-process scratch directory used to hold
    /// decrypted hidden-content blobs. The directory is removed on
    /// App drop so plaintext never persists across runs.
    fn ensure_scratch_dir(&mut self) -> std::io::Result<PathBuf> {
        if let Some(p) = self.scratch_dir.as_ref() {
            return Ok(p.clone());
        }
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("voidex-tui-{stamp}"));
        std::fs::create_dir_all(&path)?;
        self.scratch_dir = Some(path.clone());
        Ok(path)
    }

    /// Decrypt every asset belonging to an unlocked slot into a fresh
    /// scratch subdirectory, then load each video item as an
    /// `Animation` and inject it into the pack under a stable name
    /// (`secret_<slot_tag>_<index>`). Returns the list of injected
    /// keys so the panel can list them.
    fn materialise_slot(
        &mut self,
        slot: &easter_egg::UnlockedSlot,
        slot_tag: &str,
    ) -> std::io::Result<Vec<String>> {
        let root = self.ensure_scratch_dir()?;
        let slot_dir = root.join(slot_tag);
        std::fs::create_dir_all(&slot_dir)?;

        let mut injected = Vec::new();
        for (i, item) in slot.manifest.items.iter().enumerate() {
            let stem = format!("secret_{slot_tag}_{i}");
            let jsonl_path = slot_dir.join(format!("{stem}.jsonl"));
            let wav_path = slot_dir.join(format!("{stem}.wav"));

            if let Some(idx) = item.jsonl_index {
                let bytes = slot.decrypt_index(idx)?;
                std::fs::write(&jsonl_path, &bytes)?;
            }
            if let Some(idx) = item.wav_index {
                let bytes = slot.decrypt_index(idx)?;
                std::fs::write(&wav_path, &bytes)?;
            }
            if jsonl_path.is_file() {
                match Animation::load(&jsonl_path) {
                    Ok(anim) => {
                        self.pack.insert(stem.clone(), anim);
                        injected.push(stem);
                    }
                    Err(e) => eprintln!("voidex-tui: skipping secret entry: {e}"),
                }
            }
        }
        Ok(injected)
    }

    /// Try each encrypted slot with the given phrase. On any successful
    /// authenticated decrypt, the slot is stitched into the app's
    /// animation pack. Returns the human-facing message to display in
    /// the ex-line area after Enter.
    fn try_unlock_phrase(&mut self, phrase: &str) -> String {
        // Public ex-line commands (not secrets): help, snap. Recognised
        // before slot unlock attempts so an asset phrase can never gate
        // ordinary commands.
        if phrase == "help" {
            self.help_overlay = true;
            return "showing help".to_string();
        }
        if let Some(rest) = phrase.strip_prefix("snap ") {
            let name = rest.trim();
            if name.is_empty() {
                return "snap: name required".to_string();
            }
            return self.snapshot_frame(name);
        }
        if phrase == "snap" {
            return self.snapshot_frame("voidex-snap.txt");
        }
        if let Some(rest) = phrase.strip_prefix("save ") {
            return self.save_output_as(rest.trim());
        }
        if let Some(slot) = easter_egg::try_unlock(Path::new(slots::DIR_1), phrase) {
            if self.easter.panel.is_none() {
                let tag = "a".to_string();
                match self.materialise_slot(&slot, &tag) {
                    Ok(_keys) => {
                        self.easter.panel = Some(slot);
                        self.easter.panel_open = true;
                        self.panel_opened_at = Some(Instant::now());
                        return "panel unlocked".to_string();
                    }
                    Err(_) => return "unlock failed: asset error".to_string(),
                }
            }
            self.easter.panel_open = true;
            self.panel_opened_at = Some(Instant::now());
            return "panel already unlocked".to_string();
        }
        if let Some(slot) = easter_egg::try_unlock(Path::new(slots::DIR_2), phrase) {
            if self.easter.solo.is_none() {
                let tag = "b".to_string();
                match self.materialise_slot(&slot, &tag) {
                    Ok(keys) => {
                        self.easter.solo = Some(slot);
                        if let Some(first) = keys.first() {
                            let name = first.clone();
                            self.force_anim(&name);
                        }
                        return "solo unlocked".to_string();
                    }
                    Err(_) => return "unlock failed: asset error".to_string(),
                }
            }
            // Already unlocked — just switch back to the solo animation.
            if let Some(_slot) = self.easter.solo.as_ref() {
                let first = self
                    .pack
                    .keys()
                    .find(|k| k.starts_with("secret_b_"))
                    .cloned();
                if let Some(name) = first {
                    self.force_anim(&name);
                }
            }
            return "solo refreshed".to_string();
        }
        // Vim-style: "Not an editor command".
        "E: not an editor command".to_string()
    }

    /// Refresh the voice-priors cache from disk if more than 2s has
    /// elapsed since the last load. Cheap enough to call every tick.
    fn maybe_refresh_voice_priors(&mut self) {
        const REFRESH_INTERVAL: Duration = Duration::from_secs(2);
        let due = match self.voice_priors_refreshed_at {
            None => true,
            Some(t) => t.elapsed() >= REFRESH_INTERVAL,
        };
        if !due {
            return;
        }
        self.voice_priors_cache = crate::voice_view::load();
        self.voice_priors_refreshed_at = Some(Instant::now());
    }

    /// Refresh the cue preview cache from the engine's predicted
    /// output SRT and tag any newly-arrived cue indices with the
    /// current timestamp. Called once per tick on the Running screen
    /// so `render_cue_preview` can stay immutable and the cue-arrival
    /// pulse animation has a per-cue timestamp to read.
    fn refresh_cue_preview(&mut self) {
        let path_opt = match &self.screen {
            Screen::Running(rs) => canonical_output_path(&rs.runner.input_path, &rs.config),
            _ => None,
        };
        let Some(path) = path_opt else { return };
        if !path.is_file() {
            return;
        }
        let fresh = tail_srt_cues(&path, 24);
        let known: std::collections::HashSet<u32> =
            self.cue_preview_cache.iter().map(|(i, _)| *i).collect();
        let now = Instant::now();
        for (idx, _) in &fresh {
            if !known.contains(idx) {
                self.new_cue_arrivals.insert(*idx, now);
            }
        }
        // Bound the arrivals map so a long run doesn't leak; we only
        // need indices that are still visible in the preview window.
        let keep: std::collections::HashSet<u32> = fresh.iter().map(|(i, _)| *i).collect();
        self.new_cue_arrivals.retain(|k, _| keep.contains(k));
        self.cue_preview_cache = fresh;
    }

    /// Capture the currently displayed animation frame to a plain text
    /// file at the given path (or under the current working dir if the
    /// path has no directory component). Output is raw braille glyphs,
    /// one line per row — paste-able anywhere that understands UTF-8.
    fn save_output_as(&mut self, target: &str) -> String {
        let cleaned = sanitize_path_input(target);
        let src = match &self.screen {
            Screen::Result(rs) => rs.output_files.first().cloned(),
            _ => None,
        };
        let Some(src) = src else {
            return "save: no output to copy".to_string();
        };
        let root = tui_write_root();
        let dst = match resolve_write_target(&cleaned, &root) {
            Ok(p) => p,
            Err(e) => return format!("save: {e}"),
        };
        match std::fs::copy(&src, &dst) {
            Ok(_) => format!("saved → {}", dst.display()),
            Err(e) => format!("save failed: {e}"),
        }
    }

    fn snapshot_frame(&mut self, path_hint: &str) -> String {
        let anim = self.current_anim();
        if anim.frames.is_empty() {
            return "snap: no frame to capture".to_string();
        }
        let frame = &anim.frames[self.frame_idx.min(anim.frames.len() - 1)];
        let mut out = String::with_capacity((frame.width as usize + 1) * frame.height as usize);
        for row in 0..frame.height as usize {
            for col in 0..frame.width as usize {
                let i = row * frame.width as usize + col;
                if let Some(cell) = frame.cells.get(i) {
                    out.push(cell.ch);
                } else {
                    out.push(' ');
                }
            }
            out.push('\n');
        }
        let root = tui_write_root();
        let path = match resolve_write_target(path_hint, &root) {
            Ok(p) => p,
            Err(e) => return format!("snap: {e}"),
        };
        match std::fs::write(&path, out) {
            Ok(()) => format!("snap → {}", path.display()),
            Err(e) => format!("snap failed: {e}"),
        }
    }
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let pack = load_pack(Path::new(DEFAULT_ASSET_DIR))?;
    if pack.is_empty() {
        return Err(
            "no animations loaded; run scripts/tui/braille_convert.py --batch first".into(),
        );
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let mut app = App {
        pack,
        current_state_name: GREETING_DEFAULT_STATE.to_string(),
        frame_idx: 0,
        last_advance: Instant::now(),
        loop_started: Instant::now(),
        screen: Screen::Greeting,
        recents: Recents::load(),
        last_config: EngineConfig::default(),
        audio: AudioPlayer::new(),
        easter: EasterEgg::new(),
        exline: None,
        scratch_dir: None,
        anim_override: false,
        prefs: crate::prefs::Prefs::load(),
        splash_until: Some(Instant::now() + crate::splash::SPLASH_DURATION),
        theme_changed_at: None,
        panel_opened_at: None,
        help_overlay: false,
        new_cue_arrivals: std::collections::HashMap::new(),
        audio_envelope: None,
        cue_preview_cache: Vec::new(),
        voice_priors_cache: Vec::new(),
        voice_priors_refreshed_at: None,
        viz: crate::runner_viz::Viz::new(),
    };
    // Apply persisted theme + the optional auto-by-hour override.
    let saved = crate::easter_egg::Theme::from_ident(&app.prefs.theme);
    app.easter.theme = if app.prefs.auto_theme_by_hour {
        let h = crate::prefs::local_hour_24();
        if (8..19).contains(&h) {
            crate::easter_egg::Theme::KeimaBlue
        } else {
            crate::easter_egg::Theme::MrRobot
        }
    } else {
        saved
    };
    // Start audio for the greeting animation if it has any.
    app.sync_audio_to_animation();

    let result = main_loop(&mut terminal, &mut app);

    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn load_pack(asset_dir: &Path) -> Result<HashMap<String, Animation>, Box<dyn std::error::Error>> {
    let mut pack: HashMap<String, Animation> = HashMap::new();
    for state in STATES {
        let path = asset_dir.join(format!("{state}.jsonl"));
        match Animation::load(&path) {
            Ok(a) => {
                pack.insert((*state).to_string(), a);
            }
            Err(AnimationError::Io(e)) if e.kind() == io::ErrorKind::NotFound => {
                eprintln!("note: missing {} (skipped)", path.display());
            }
            Err(e) => eprintln!("warning: {} → {}", path.display(), e),
        }
    }
    Ok(pack)
}

fn main_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        // Per-frame engine poll if Running.
        if let Screen::Running(rs) = &mut app.screen {
            let finished = rs.runner.poll();
            // Drive animation from engine state — but only when the
            // user hasn't manually overridden it (panel pick, solo
            // unlock, Tab cycle, etc.). Without this gate the engine
            // would stomp the user's choice every poll tick.
            if !app.anim_override {
                let target = rs.runner.state.effective_anim_state();
                app.switch_anim(target);
            }
            if finished {
                let runner = match std::mem::replace(&mut app.screen, Screen::Greeting) {
                    Screen::Running(rs) => rs.runner,
                    _ => unreachable!(),
                };
                let success = runner.state.success.unwrap_or(false);
                let duration = runner
                    .state
                    .started_at
                    .map(|t| t.elapsed())
                    .unwrap_or_default();

                // If success and we have an output, fall back to the
                // canonical output path heuristic when the engine
                // didn't emit an explicit `output` field.
                let mut outputs = runner.state.output_files.clone();
                if outputs.is_empty() && success {
                    if let Some(canonical) =
                        canonical_output_path(&runner.input_path, &runner.config)
                    {
                        if canonical.is_file() {
                            outputs.push(canonical);
                        }
                    }
                }
                let srt_preview = outputs
                    .first()
                    .map(|p| read_srt_preview(p, 12))
                    .unwrap_or_default();

                let cfg = runner.config.clone();
                app.last_config = cfg.clone();
                app.screen = Screen::Result(ResultState {
                    input: runner.input_path.clone(),
                    config: cfg,
                    success,
                    duration,
                    output_files: outputs,
                    last_message: runner.state.last_message.clone(),
                    chunk_total: runner.state.chunk_total,
                    quality: runner.state.quality,
                    srt_preview,
                });
                app.switch_anim(if success { "complete" } else { "error" });
            }
        }

        // Refresh the live SRT tail once per tick so the renderer
        // can pulse newly-arrived cues without doing I/O inside
        // render(). Voice priors are throttled internally.
        if matches!(app.screen, Screen::Running(_)) {
            app.refresh_cue_preview();
            app.maybe_refresh_voice_priors();
            let emerge_perm = if app.viz.mode == crate::runner_viz::VizMode::Emerge {
                match &app.screen {
                    Screen::Running(rs) => {
                        let frame = &app.current_anim().frames[app.frame_idx];
                        Some((
                            rs.runner.input_path.to_string_lossy().into_owned(),
                            crate::runner_viz::frame_cell_count(frame),
                        ))
                    }
                    _ => None,
                }
            } else {
                None
            };
            if let Some((key, cell_count)) = emerge_perm {
                app.viz.ensure_emerge_perm(&key, cell_count);
            }
            // Step the Running-screen visualizer. We don't have the
            // exact panel rect here, so we pass an over-approximation
            // (55% of terminal width × full height) — the viz adapts
            // its canvas to whatever it actually gets at render time.
            let (cw, ct) = match &app.screen {
                Screen::Running(rs) => (rs.runner.state.chunk_current, rs.runner.state.chunk_total),
                _ => (0, 0),
            };
            let term = terminal.size()?;
            let viz_w = ((term.width as u32 * 55) / 100).saturating_sub(4) as u16;
            let viz_h = term.height.saturating_sub(8);
            let phase = app.loop_started.elapsed().as_secs_f32();
            app.viz.step(viz_w, viz_h, phase, cw, ct);
        }
        terminal.draw(|f| render(f, app))?;
        app.advance_frame_if_due();

        // Idle redraw throttle. The default is the 50 ms (~20 Hz) cadence
        // animations need to look smooth. When the user opted into
        // reduced motion, the animation pins to its final frame, so we
        // can slow the loop to 250 ms — still responsive on keypress,
        // but a quarter the wakeups per second and a quarter the
        // battery cost on a laptop sitting at the splash screen.
        let poll_ms = if crate::accessibility::Accessibility::from_env().reduced_motion {
            250
        } else {
            50
        };
        if event::poll(Duration::from_millis(poll_ms))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                if matches!(k.code, KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL))
                {
                    return Ok(());
                }
                // Splash takeover: any key skips the remaining splash
                // duration. Don't consume the key — fall through so
                // the user's intent (e.g., Enter to start) still
                // applies on the very same tick.
                if let Some(until) = app.splash_until {
                    if Instant::now() < until {
                        app.splash_until = None;
                    }
                }
                // Help overlay: any key dismisses it and is consumed.
                if app.help_overlay {
                    app.help_overlay = false;
                    continue;
                }
                // Boss key takeover: while active, any key returns us
                // to normal rendering. Classic behaviour — instant
                // visibility on a single keystroke from any angle.
                if app.easter.boss {
                    app.easter.boss = false;
                    continue;
                }
                // Suppressed on Picker so they don't collide with text entry.
                let on_text_input = matches!(&app.screen, Screen::Picker(_));
                if !on_text_input {
                    if matches!(k.code, KeyCode::Char('P')) && app.easter.panel.is_some() {
                        app.easter.panel_open = !app.easter.panel_open;
                        if app.easter.panel_open {
                            app.panel_opened_at = Some(Instant::now());
                        }
                        continue;
                    }
                    if matches!(k.code, KeyCode::Char('B')) && app.easter.panel.is_some() {
                        app.easter.toggle_boss();
                        continue;
                    }
                }
                // Vim ex-line takes priority over everything else when
                // it's open. Open it on `:` from any screen except
                // while the user is typing into a text field
                // (Picker / Configure have their own char handlers).
                if input::handle_exline_key(app, k) {
                    continue;
                }
                // Panel hotkeys when it's visible — themes, boss key,
                // close — only apply when the panel is open.
                if app.easter.panel_open && input::handle_panel_key(app, k) {
                    continue;
                }
                let next = input::handle_key(app, k);
                if let Some(rc) = next {
                    return rc;
                }
            }
        }
    }
}

fn render(f: &mut ratatui::Frame, app: &App) {
    let area = f.area();

    // Splash logo for the first SPLASH_DURATION of the session.
    if let Some(until) = app.splash_until {
        if Instant::now() < until {
            let started = until - crate::splash::SPLASH_DURATION;
            crate::splash::render(f, area, started, theme_accent(app.easter.theme));
            return;
        }
    }

    // Boss key wins over everything else: render a fake "compiling..."
    // screen until toggled off. This makes the easter-egg state
    // instantly safe-for-meeting without quitting.
    if app.easter.boss {
        render_boss_screen(f, area);
        return;
    }

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // title
            Constraint::Length(1), // gap
            Constraint::Min(0),    // body
            Constraint::Length(1), // gap
            Constraint::Length(1), // hint or ex-line
        ])
        .split(area);

    render_titlebar(f, sections[0], app);
    match &app.screen {
        Screen::Greeting => render_greeting(f, sections[2], app),
        Screen::Picker(ps) => render_picker(f, sections[2], app, ps),
        Screen::Configure(cs) => render_configure(f, sections[2], app, cs),
        Screen::Running(rs) => render_running(f, sections[2], app, rs),
        Screen::Result(rs) => render_result(f, sections[2], app, rs),
    }
    // Ex-line replaces the hint bar when active; otherwise the hint
    // bar renders as normal. Panel overlay floats on top of body.
    if app.exline.is_some() {
        render_exline(f, sections[4], app);
    } else {
        render_hint(f, sections[4], app);
    }
    if app.easter.panel_open && app.easter.panel.is_some() {
        render_panel_overlay(f, sections[2], app);
    }
    if app.help_overlay {
        render_help_overlay(f, sections[2]);
    }
}

/// Centred help overlay listing all keybinds and ex-line commands.
/// Dismissed on the next keystroke (handled in the key dispatcher).
fn render_help_overlay(f: &mut ratatui::Frame, body: Rect) {
    let lines: Vec<Line<'static>> = vec![
        Line::from(Span::styled(
            " voidex-tui · help ",
            Style::default()
                .fg(palette::ACCENT_2)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            " ── visible ──────────────────────────────",
            Style::default().fg(palette::MUTED),
        )),
        help_row("Enter", "start / proceed"),
        help_row("Tab / ⇧Tab", "cycle animation"),
        help_row("1-7", "preview animation slot"),
        help_row("m / +/-", "audio mute / volume"),
        help_row("p", "cycle quality profile (Greeting)"),
        help_row("c", "cancel running translation"),
        help_row("Esc", "back"),
        help_row("q", "quit"),
        Line::raw(""),
        Line::from(Span::styled(
            " ── ex-line (after `:`) ──────────────────",
            Style::default().fg(palette::MUTED),
        )),
        help_row(":help", "this screen"),
        help_row(":snap NAME", "save current frame to NAME"),
        help_row(":save PATH", "save the last output srt to PATH"),
        Line::raw(""),
        Line::from(Span::styled(
            " any key to dismiss ",
            Style::default()
                .fg(palette::FAINT)
                .add_modifier(Modifier::ITALIC),
        )),
    ];
    let widest = lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.chars().count())
                .sum::<usize>()
        })
        .max()
        .unwrap_or(48);
    let want_w = (widest as u16 + 4).min(body.width);
    let want_h = (lines.len() as u16 + 2).min(body.height);
    let rect = Rect {
        x: body.x + body.width.saturating_sub(want_w) / 2,
        y: body.y + body.height.saturating_sub(want_h) / 2,
        width: want_w,
        height: want_h,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::ACCENT_2));
    let inner = block.inner(rect);
    f.render_widget(block, rect);
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

fn help_row(key: &'static str, label: &'static str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key:<10}"), Style::default().fg(palette::PINK)),
        Span::styled(label, Style::default().fg(palette::TEXT)),
    ])
}

/// Vim ex-line: a single bottom-of-screen text input prefixed by `:`.
/// The Enter handler in `handle_exline_key` stuffs a one-line flash
/// message into the buffer prefixed with `\x00`; we strip that here
/// and render it in a different colour.
fn render_exline(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let buf = app.exline.as_deref().unwrap_or("");
    let (prefix, body, style) = if let Some(msg) = buf.strip_prefix('\x00') {
        ("", msg, Style::default().fg(palette::ACCENT))
    } else {
        (":", buf, Style::default().fg(palette::TEXT))
    };
    let line = Line::from(vec![
        Span::styled(prefix, Style::default().fg(palette::ACCENT_2)),
        Span::styled(body, style),
        // Block cursor: only show during real typing, not on flash.
        if buf.starts_with('\x00') {
            Span::raw("")
        } else {
            Span::styled("\u{2588}", Style::default().fg(palette::MUTED))
        },
    ]);
    f.render_widget(Paragraph::new(line), area);
}

/// Tiny floating list overlay rendered in the top-right of the body
/// area when the hidden panel is open. Shows the unlocked items, the
/// active theme, and the hotkey hints (t = theme, b = boss, p = close,
/// Enter = play, j/k = navigate).
fn render_panel_overlay(f: &mut ratatui::Frame, body_area: Rect, app: &App) {
    let Some(panel) = app.easter.panel.as_ref() else {
        return;
    };
    let item_count = panel.manifest.items.len();
    let mut lines: Vec<Line> = Vec::with_capacity(item_count + 4);
    lines.push(Line::from(Span::styled(
        " HIDDEN PANEL ",
        Style::default()
            .fg(palette::ACCENT_2)
            .add_modifier(Modifier::BOLD | Modifier::REVERSED),
    )));
    lines.push(Line::raw(""));
    for (i, item) in panel.manifest.items.iter().enumerate() {
        let marker = if i == app.easter.panel_cursor {
            "▸ "
        } else {
            "  "
        };
        let label_style = if i == app.easter.panel_cursor {
            Style::default()
                .fg(palette::ACCENT_2)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(palette::TEXT)
        };
        lines.push(Line::from(vec![
            Span::styled(marker, Style::default().fg(palette::ACCENT)),
            Span::styled(
                format!("[{}] ", item.kind),
                Style::default().fg(palette::MUTED),
            ),
            Span::styled(item.label.clone(), label_style),
        ]));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        format!(" theme: {}", app.easter.theme.label()),
        Style::default().fg(palette::PINK),
    )));
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        " t·theme  b·boss  Enter·play  j/k·nav  p/Esc·close ",
        Style::default().fg(palette::MUTED),
    )));

    let widest = lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.chars().count())
                .sum::<usize>()
        })
        .max()
        .unwrap_or(40);
    let full_w = (widest as u16 + 4).min(body_area.width.saturating_sub(2));
    let want_h = (lines.len() as u16 + 2).min(body_area.height.saturating_sub(2));
    // Panel slide-in animation: ease the width from 0 → full over
    // ~220ms after the panel opens. Quadratic ease-out so the motion
    // settles softly rather than snapping.
    let want_w = match app.panel_opened_at {
        Some(t) => {
            let elapsed_ms = t.elapsed().as_millis() as u32;
            const SLIDE_MS: u32 = 220;
            if elapsed_ms >= SLIDE_MS {
                full_w
            } else {
                let frac = elapsed_ms as f32 / SLIDE_MS as f32;
                let eased = 1.0 - (1.0 - frac).powi(2);
                ((full_w as f32) * eased).max(8.0).round() as u16
            }
        }
        None => full_w,
    };
    let rect = Rect {
        x: body_area.x + body_area.width.saturating_sub(want_w + 1),
        y: body_area.y + 1,
        width: want_w,
        height: want_h,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::PINK));
    let inner = block.inner(rect);
    f.render_widget(block, rect);
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);

    // Audio waveform strip rendered beneath the panel content when
    // music is currently playing and the active animation has an
    // envelope sidecar (otherwise falls back to a moving sine).
    if let Some(audio) = app.audio.as_ref() {
        if audio.is_playing() {
            let strip_rect = Rect {
                x: rect.x,
                y: rect.y.saturating_add(rect.height),
                width: rect.width,
                height: 3.min(body_area.height.saturating_sub(rect.height + 1)),
            };
            if strip_rect.height >= 1 && strip_rect.y < body_area.y + body_area.height {
                render_waveform_strip(f, strip_rect, app);
            }
        }
    }
}

/// Compact one-row braille-block waveform for the music currently
/// playing. Driven by the envelope sidecar when present, otherwise a
/// moving sine so the strip never looks dead.
fn render_waveform_strip(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::FAINT));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width < 4 || inner.height < 1 {
        return;
    }
    let bars = inner.width as usize;
    let cursor_ms = app.loop_started.elapsed().as_millis() as u64;
    let values = match app.audio_envelope.as_ref() {
        Some(env) => env.window(cursor_ms, 1_500, bars),
        None => crate::waveform::fallback_bars(bars, app.loop_started.elapsed().as_secs_f64()),
    };
    let line: String = values
        .iter()
        .map(|v| crate::waveform::block_for(*v))
        .collect();
    let body = Paragraph::new(Line::from(Span::styled(
        line,
        Style::default().fg(theme_accent(app.easter.theme)),
    )));
    f.render_widget(body, inner);
}

/// Boss-key full-screen takeover: a fake "compiling..." screen any
/// passer-by would assume is a real build log.
fn render_boss_screen(f: &mut ratatui::Frame, area: Rect) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let lines: Vec<Line> = vec![
        Line::raw(""),
        Line::from(Span::styled(
            "  Compiling voidex v0.1.0 (D:\\BVoid\\Work\\new-project)",
            Style::default().fg(palette::TEXT),
        )),
        Line::from(Span::styled(
            "     Checking dependencies...",
            Style::default().fg(palette::MUTED),
        )),
        Line::from(Span::styled(
            format!(
                "     Building [============>      ] {} / 218",
                (now % 200) + 18
            ),
            Style::default().fg(palette::ACCENT_2),
        )),
        Line::from(Span::styled(
            "        Compiling tokio v1.49.0",
            Style::default().fg(palette::MUTED),
        )),
        Line::from(Span::styled(
            "        Compiling serde v1.0.219",
            Style::default().fg(palette::MUTED),
        )),
        Line::from(Span::styled(
            "        Compiling ratatui v0.28.0",
            Style::default().fg(palette::MUTED),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            "  (any key returns to normal view)",
            Style::default().fg(palette::FAINT),
        )),
    ];
    let body = Paragraph::new(lines);
    f.render_widget(body, area);
}

fn render_titlebar(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(0),
            Constraint::Length(22), // volume widget
            Constraint::Length(20), // status
            Constraint::Length(2),
        ])
        .split(area);
    let (status_label, status_color) = status_for_screen(&app.screen);
    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            "VoiDex",
            Style::default()
                .fg(palette::ACCENT_2)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ·  brigade", Style::default().fg(palette::MUTED)),
    ]));
    f.render_widget(title, cols[1]);
    render_volume_widget(f, cols[2], app);
    let online = Paragraph::new(Line::from(vec![
        Span::styled("●", Style::default().fg(status_color)),
        Span::styled(
            format!(" {}", status_label),
            Style::default().fg(palette::MUTED),
        ),
    ]))
    .alignment(Alignment::Right);
    f.render_widget(online, cols[3]);
}

/// 8-bar volume meter that appears only when the active animation
/// carries audio. Muted state shows `🔇` (rendered as `M` in
/// ASCII-only terminals) plus the dim bar.
fn render_volume_widget(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let Some(audio) = app.audio.as_ref() else {
        return;
    };
    if !app.current_anim_has_audio() {
        return;
    }
    let bars = 8usize;
    let filled = ((audio.volume() * bars as f32).round() as usize).min(bars);
    let label_color = if audio.is_muted() {
        palette::MUTED
    } else {
        palette::ACCENT_2
    };
    let mut spans: Vec<Span> = Vec::with_capacity(bars + 4);
    spans.push(Span::styled(
        if audio.is_muted() { "muted " } else { "vol   " },
        Style::default().fg(label_color),
    ));
    for i in 0..bars {
        if audio.is_muted() {
            spans.push(Span::styled("▂", Style::default().fg(palette::FAINT)));
        } else if i < filled {
            spans.push(Span::styled("▰", Style::default().fg(palette::ACCENT)));
        } else {
            spans.push(Span::styled("▱", Style::default().fg(palette::FAINT)));
        }
    }
    let p = Paragraph::new(Line::from(spans)).alignment(Alignment::Right);
    f.render_widget(p, area);
}

fn status_for_screen(s: &Screen) -> (&'static str, Color) {
    match s {
        Screen::Greeting => ("online", palette::GREEN),
        Screen::Picker(_) => ("pick", palette::GREEN),
        Screen::Configure(_) => ("configure", palette::ACCENT_2),
        Screen::Running(rs) => match rs.runner.state.anim_state {
            "error" => ("error", palette::RED),
            "warning" => ("warn", palette::ACCENT_2),
            _ => ("running", palette::ACCENT),
        },
        Screen::Result(rs) => {
            if rs.success {
                ("complete", palette::GREEN)
            } else {
                ("error", palette::RED)
            }
        }
    }
}

// ─── Greeting ────────────────────────────────────────────────────────

fn render_greeting(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);
    render_animation_centered(f, rows[0], app, true);
    render_tabs(f, rows[1], app);
    render_progress(f, rows[2], app);
}

// ─── Picker ──────────────────────────────────────────────────────────

fn render_picker(f: &mut ratatui::Frame, area: Rect, app: &App, ps: &PickerState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Length(7),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(area);

    render_animation_centered(f, rows[0], app, false);

    // Picker input form.
    let form_rect = centered_rect(rows[1], 80, rows[1].height);
    let input_active = ps.selected_recent.is_none();
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if input_active {
            palette::ACCENT
        } else {
            palette::FAINT
        }))
        .title(Span::styled(
            " choose a file ",
            Style::default()
                .fg(palette::ACCENT_2)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(form_rect);
    f.render_widget(block, form_rect);

    let cursor_marker = if input_active { "█" } else { " " };
    let prompt = Line::from(vec![
        Span::styled("▏ ", Style::default().fg(palette::ACCENT)),
        Span::styled(&ps.input, Style::default().fg(palette::TEXT)),
        Span::styled(cursor_marker, Style::default().fg(palette::ACCENT_2)),
    ]);
    let mut content = vec![Line::raw(""), prompt, Line::raw("")];
    if let Some(err) = &ps.error {
        content.push(Line::from(Span::styled(
            err.clone(),
            Style::default()
                .fg(palette::RED)
                .add_modifier(Modifier::DIM),
        )));
    } else {
        content.push(Line::from(Span::styled(
            "type or paste a video / audio path · ↓ select a recent file",
            Style::default()
                .fg(palette::MUTED)
                .add_modifier(Modifier::DIM),
        )));
    }
    f.render_widget(Paragraph::new(content), inner);

    // Recents list, if any.
    if !app.recents.entries.is_empty() {
        render_recents(f, rows[3], app, ps);
    }
}

fn render_recents(f: &mut ratatui::Frame, area: Rect, app: &App, ps: &PickerState) {
    let rect = centered_rect(area, 80, area.height);
    let mut lines = Vec::with_capacity(app.recents.entries.len() + 2);
    lines.push(Line::from(Span::styled(
        "  recents",
        Style::default()
            .fg(palette::MUTED)
            .add_modifier(Modifier::DIM),
    )));
    for (i, entry) in app.recents.entries.iter().enumerate().take(8) {
        let active = ps.selected_recent == Some(i);
        let bullet = if active { "▶ " } else { "  " };
        let bullet_style = if active {
            Style::default().fg(palette::ACCENT_2)
        } else {
            Style::default().fg(palette::FAINT)
        };
        let path_style = if active {
            Style::default()
                .fg(palette::ACCENT_2)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(palette::TEXT)
        };
        let display = entry
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| entry.path.to_str().unwrap_or(""))
            .to_string();
        lines.push(Line::from(vec![
            Span::styled(bullet, bullet_style),
            Span::styled(display, path_style),
        ]));
    }
    f.render_widget(Paragraph::new(lines), rect);
}

// ─── Configure ───────────────────────────────────────────────────────

fn render_configure(f: &mut ratatui::Frame, area: Rect, app: &App, cs: &ConfigureState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Length(13),
            Constraint::Min(0),
        ])
        .split(area);

    render_animation_centered(f, rows[0], app, false);

    let form_rect = centered_rect(rows[1], 84, rows[1].height);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::ACCENT))
        .title(Span::styled(
            " configure ",
            Style::default()
                .fg(palette::ACCENT_2)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(form_rect);
    f.render_widget(block, form_rect);

    let path_disp = shorten(&cs.input_path.to_string_lossy(), 70);
    let mut lines = vec![
        Line::from(vec![
            Span::styled("  file       ", Style::default().fg(palette::MUTED)),
            Span::styled(path_disp, Style::default().fg(palette::TEXT)),
        ]),
        Line::raw(""),
        config_row(
            "source",
            &cs.config.source_lang,
            cs.cursor == ConfigField::SourceLang,
        ),
        config_row(
            "target",
            &cs.config.target_lang,
            cs.cursor == ConfigField::TargetLang,
        ),
        config_row(
            "profile",
            &cs.config.profile,
            cs.cursor == ConfigField::Profile,
        ),
        config_row(
            "gpu",
            if cs.config.gpu { "on" } else { "off" },
            cs.cursor == ConfigField::Gpu,
        ),
        config_row(
            "workers",
            &cs.config.workers.to_string(),
            cs.cursor == ConfigField::Workers,
        ),
        Line::raw(""),
        Line::from({
            let active = cs.cursor == ConfigField::Start;
            let style = if active {
                Style::default()
                    .fg(palette::ACCENT_2)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
                    .fg(palette::ACCENT)
                    .add_modifier(Modifier::DIM)
            };
            vec![Span::raw("  "), Span::styled("▶  start engine", style)]
        }),
    ];
    let _ = lines.last_mut();
    f.render_widget(Paragraph::new(lines), inner);
}

fn config_row(label: &str, value: &str, active: bool) -> Line<'static> {
    let label_style = if active {
        Style::default()
            .fg(palette::ACCENT_2)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(palette::MUTED)
    };
    let value_style = if active {
        Style::default()
            .fg(palette::ACCENT_2)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(palette::TEXT)
    };
    let arrows = if active {
        Span::styled("  ◀ ", Style::default().fg(palette::PINK))
    } else {
        Span::raw("    ")
    };
    let arrows_r = if active {
        Span::styled(" ▶", Style::default().fg(palette::PINK))
    } else {
        Span::raw("  ")
    };
    Line::from(vec![
        Span::styled(format!("  {label:<10}"), label_style),
        arrows,
        Span::styled(value.to_string(), value_style),
        arrows_r,
    ])
}

// ─── Running ─────────────────────────────────────────────────────────

fn render_running(f: &mut ratatui::Frame, area: Rect, app: &App, rs: &RunningState) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);

    render_running_animation(f, cols[0], app, rs);

    // Right side: engine info + live telemetry + cue preview + log tail.
    // Telemetry block grows to fit per-speaker voice signatures
    // when the engine has populated voice_priors.json. Without
    // signatures, 8 lines is plenty for sparkline + voice line +
    // gate quality. With signatures, give it more room.
    let telemetry_h = if app.voice_priors_cache.is_empty() {
        8
    } else {
        (8 + app.voice_priors_cache.len() as u16).min(16)
    };
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),          // engine info
            Constraint::Length(telemetry_h), // telemetry
            Constraint::Min(8),              // cue preview
            Constraint::Length(6),           // log tail
        ])
        .split(cols[1]);

    render_engine_info(f, right[0], app, rs);
    render_telemetry(f, right[1], app, rs);
    render_cue_preview(f, right[2], app);
    render_log_tail(f, right[3], rs);
}

/// Live cue-preview pane. Reads `app.cue_preview_cache` (refreshed by
/// `App::refresh_cue_preview` in main_loop) so render stays
/// immutable. Newly-arrived cues — those whose index appeared in the
/// last refresh — flash in accent for ~600 ms before settling.
fn render_cue_preview(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::FAINT))
        .title(Span::styled(
            " live cues ",
            Style::default().fg(palette::ACCENT),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width < 16 || inner.height < 2 {
        return;
    }

    if app.cue_preview_cache.is_empty() {
        let placeholder = Paragraph::new(Line::from(Span::styled(
            "  waiting for first cue …",
            Style::default()
                .fg(palette::FAINT)
                .add_modifier(Modifier::ITALIC),
        )));
        f.render_widget(placeholder, inner);
        return;
    }

    let now = Instant::now();
    const SWOOSH_MS: u128 = 700;
    const HOLD_MS: u128 = 700;
    let visible = app
        .cue_preview_cache
        .len()
        .saturating_sub(inner.height as usize);
    let line_width = inner.width as usize;
    let lines: Vec<Line> = app.cue_preview_cache[visible..]
        .iter()
        .map(|(idx, text)| {
            let pulse_age = app
                .new_cue_arrivals
                .get(idx)
                .map(|t| now.duration_since(*t).as_millis());
            match pulse_age {
                Some(age) if age < SWOOSH_MS => {
                    // Mid-swoosh: split the line at the sweep cursor.
                    let frac = age as f32 / SWOOSH_MS as f32;
                    // ease-out so motion settles on the last few chars
                    let eased = 1.0 - (1.0 - frac).powi(2);
                    let cursor = ((line_width as f32) * eased) as usize;
                    let prefix_len = format!(" {idx:>4}  ").chars().count();
                    let text_chars: Vec<char> = text.chars().collect();
                    let cut = cursor.saturating_sub(prefix_len).min(text_chars.len());
                    let head: String = text_chars[..cut].iter().collect();
                    let tail: String = text_chars[cut..].iter().collect();
                    Line::from(vec![
                        Span::styled(
                            format!(" {:>4}  ", idx),
                            Style::default()
                                .fg(palette::ACCENT_2)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            head,
                            Style::default()
                                .fg(palette::ACCENT_2)
                                .add_modifier(Modifier::BOLD),
                        ),
                        // Swoosh head: a single bright accent character
                        // pulled across the line as the sweep advances.
                        Span::styled(
                            "▌",
                            Style::default()
                                .fg(palette::PINK)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(tail, Style::default().fg(palette::MUTED)),
                    ])
                }
                Some(age) if age < SWOOSH_MS + HOLD_MS => {
                    // Hold phase: accent persists, no sweep cursor.
                    let style = Style::default()
                        .fg(palette::ACCENT_2)
                        .add_modifier(Modifier::BOLD);
                    Line::from(vec![
                        Span::styled(format!(" {:>4}  ", idx), style),
                        Span::styled(text.clone(), style),
                    ])
                }
                _ => Line::from(vec![
                    Span::styled(
                        format!(" {:>4}  ", idx),
                        Style::default().fg(palette::MUTED),
                    ),
                    Span::styled(text.clone(), Style::default().fg(palette::TEXT)),
                ]),
            }
        })
        .collect();
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

/// Compact telemetry panel: throughput sparkline of recent chunk
/// completions, voice-consistency aggregate from the F.2 voice
/// prior stage, and (when the engine has produced voice priors) one
/// per-speaker 6-bar "voice signature" row reflecting the learned
/// character voice. Sits between engine_info and cue preview.
fn render_telemetry(f: &mut ratatui::Frame, area: Rect, app: &App, rs: &RunningState) {
    let s = &rs.runner.state;
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::FAINT))
        .title(Span::styled(
            " telemetry ",
            Style::default().fg(palette::MUTED),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width < 16 || inner.height < 3 {
        return;
    }

    // Throughput sparkline. We bucket the trailing `inner.width` seconds
    // and count completions per bucket. Block-element heights give a
    // tiny live chart without any external chart library.
    let bar_width = inner.width.saturating_sub(12) as usize;
    let now = std::time::Instant::now();
    let mut buckets = vec![0u32; bar_width.max(1)];
    for t in s.chunk_completions.iter() {
        let secs_ago = now.duration_since(*t).as_secs_f64();
        let bucket = bar_width
            .saturating_sub(1)
            .saturating_sub(secs_ago as usize);
        if bucket < buckets.len() {
            buckets[bucket] = buckets[bucket].saturating_add(1);
        }
    }
    let max_bucket = buckets.iter().copied().max().unwrap_or(0).max(1);
    let glyphs = [
        ' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
        '\u{2588}',
    ];
    let spark: String = buckets
        .iter()
        .map(|&c| {
            let lvl = ((c as f64 * 8.0 / max_bucket as f64).round() as usize).min(8);
            glyphs[lvl]
        })
        .collect();
    let total_done = s.chunk_completions.len();
    let throughput_line = Line::from(vec![
        Span::styled("  chunks/s  ", Style::default().fg(palette::MUTED)),
        Span::styled(spark, Style::default().fg(palette::ACCENT_2)),
        Span::styled(
            format!(" {total_done:>3}"),
            Style::default().fg(palette::TEXT),
        ),
    ]);

    // Voice-consistency block. Only populated once the engine has run
    // the voice_consistency stage on at least one input.
    let voice_line = match (s.voice_speakers_observed, s.voice_mean_deviation) {
        (Some(speakers), Some(mean)) => {
            let p95 = s.voice_p95_deviation.unwrap_or(0.0);
            let cues = s.voice_cues_scored.unwrap_or(0);
            Line::from(vec![
                Span::styled("  voice     ", Style::default().fg(palette::MUTED)),
                Span::styled(
                    format!("μ={mean:.3} p95={p95:.3}"),
                    Style::default().fg(palette::PINK),
                ),
                Span::styled(
                    format!("  ({} speakers, {} cues)", speakers, cues),
                    Style::default().fg(palette::MUTED),
                ),
            ])
        }
        _ => Line::from(vec![
            Span::styled("  voice     ", Style::default().fg(palette::MUTED)),
            Span::styled("—", Style::default().fg(palette::FAINT)),
        ]),
    };

    // Quality gate readout for the last completed chunk.
    let q_line = match s.quality {
        Some(q) => {
            let colour = if q >= 0.7 {
                palette::GREEN
            } else if q >= 0.5 {
                palette::ACCENT_2
            } else {
                palette::RED
            };
            Line::from(vec![
                Span::styled("  gate q    ", Style::default().fg(palette::MUTED)),
                Span::styled(format!("{q:.2}"), Style::default().fg(colour)),
            ])
        }
        None => Line::from(vec![
            Span::styled("  gate q    ", Style::default().fg(palette::MUTED)),
            Span::styled("—", Style::default().fg(palette::FAINT)),
        ]),
    };

    // Per-speaker voice signatures. Each speaker becomes one line of
    // six block-element bars; column legend ("ctr pol 1st len qst itj")
    // is prepended once at the top.
    let mut lines = vec![throughput_line, voice_line, q_line];
    if !app.voice_priors_cache.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("  voices    ", Style::default().fg(palette::MUTED)),
            Span::styled(
                crate::voice_view::FEATURE_HEADER,
                Style::default().fg(palette::FAINT),
            ),
        ]));
        let limit = (inner.height as usize).saturating_sub(4).max(0);
        for sig in app.voice_priors_cache.iter().take(limit) {
            let bars: String = sig
                .bars
                .iter()
                .map(|v| {
                    let g = crate::voice_view::bar_glyph(*v);
                    format!("{g}{g}{g} ")
                })
                .collect();
            let name = if sig.name.len() > 8 {
                format!("{}…", &sig.name[..7])
            } else {
                sig.name.clone()
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {:<8}  ", name),
                    Style::default().fg(palette::TEXT),
                ),
                Span::styled(bars, Style::default().fg(theme_accent(app.easter.theme))),
                Span::styled(
                    format!(" ({})", sig.samples),
                    Style::default().fg(palette::FAINT),
                ),
            ]));
        }
    }
    let body = Paragraph::new(lines);
    f.render_widget(body, inner);
}

fn render_engine_info(f: &mut ratatui::Frame, area: Rect, app: &App, rs: &RunningState) {
    let s = &rs.runner.state;
    let stage = s.current_stage.as_deref().unwrap_or("init");
    let elapsed = s
        .started_at
        .map(|t| format_duration(t.elapsed()))
        .unwrap_or_default();
    let eta = s
        .eta_secs
        .map(|e| format_duration(Duration::from_secs_f64(e.max(0.0))))
        .unwrap_or_else(|| "—".into());
    let chunks = if s.chunk_total > 0 {
        format!("{} / {}", s.chunk_current, s.chunk_total)
    } else {
        "—".to_string()
    };
    let cues = s
        .cue_count
        .map(|count| count.to_string())
        .unwrap_or_else(|| "—".to_string());
    let tick = app.loop_started.elapsed().as_millis();

    let lines: Vec<Line> = vec![
        kv(
            "source",
            &shorten(rs.runner.input_path.to_string_lossy().as_ref(), 36),
        ),
        pipeline_flowchart_line(stage, &s.stages_complete, tick),
        chunk_timeline_line(s.chunk_current, s.chunk_total, area.width, tick),
        kv("chunks", &chunks),
        kv("cues", &cues),
        kv("elapsed", &elapsed),
        kv("eta", &eta),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::FAINT))
        .title(Span::styled(
            " engine ",
            Style::default()
                .fg(palette::ACCENT_2)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);
    f.render_widget(Paragraph::new(lines), inner);
}

/// Animated per-chunk timeline strip. Each chunk becomes one cell:
/// • done    → solid block in green
/// • active  → shimmering accent block (alternates ▓/█ on tick)
/// • pending → faint half-block placeholder
/// Cells scale down automatically when there are more chunks than
/// available columns — we draw at most `panel_width − 12` cells.
fn chunk_timeline_line(current: u32, total: u32, panel_width: u16, tick: u128) -> Line<'static> {
    if total == 0 {
        return Line::from(vec![
            Span::styled("  chunks ", Style::default().fg(palette::MUTED)),
            Span::styled("—", Style::default().fg(palette::FAINT)),
        ]);
    }
    let max_cells = (panel_width.saturating_sub(12) as usize).max(8);
    let cells_to_draw = (total as usize).min(max_cells);
    let scale = total as f64 / cells_to_draw as f64;
    let cur_cell = ((current as f64 - 0.5) / scale).clamp(0.0, cells_to_draw as f64 - 1.0) as usize;

    let mut spans: Vec<Span> = vec![Span::styled(
        "  chunks ",
        Style::default().fg(palette::MUTED),
    )];
    // Shimmer: alternate between ▓ and █ every 200 ms for the active
    // cell so the eye picks it out without strobing.
    let shimmer_on = (tick / 200).is_multiple_of(2);
    let active_glyph = if shimmer_on { '█' } else { '▓' };

    let mut chunk_str = String::with_capacity(cells_to_draw);
    let mut styles: Vec<(Style, usize)> = Vec::new(); // (style, char count) runs
    let mut prev_state: Option<u8> = None; // 0=done,1=active,2=pending
    let mut run_len = 0usize;

    let push_run = |styles: &mut Vec<(Style, usize)>, state: u8, n: usize| {
        let st = match state {
            0 => Style::default().fg(palette::GREEN),
            1 => Style::default()
                .fg(palette::ACCENT_2)
                .add_modifier(Modifier::BOLD),
            _ => Style::default().fg(palette::FAINT),
        };
        styles.push((st, n));
    };

    for i in 0..cells_to_draw {
        let cell_progress = (i as f64) * scale;
        let state = if cell_progress < current as f64 - scale * 0.5 {
            0u8
        } else if i == cur_cell {
            1u8
        } else {
            2u8
        };
        let glyph = match state {
            0 => '█',
            1 => active_glyph,
            _ => '▁',
        };
        if Some(state) != prev_state && run_len > 0 {
            push_run(&mut styles, prev_state.unwrap(), run_len);
            run_len = 0;
        }
        chunk_str.push(glyph);
        run_len += 1;
        prev_state = Some(state);
    }
    if let Some(state) = prev_state {
        push_run(&mut styles, state, run_len);
    }

    // Re-emit chunk_str split into styled spans matching the runs.
    let mut cursor = 0;
    for (style, len) in styles {
        let slice: String = chunk_str.chars().skip(cursor).take(len).collect();
        spans.push(Span::styled(slice, style));
        cursor += len;
    }
    Line::from(spans)
}

/// Animated ASCII dataflow: each pipeline stage drawn as a box, joined
/// by particle-streaming arrows. The currently-active stage glows in
/// the accent colour with a rotating braille spinner; completed stages
/// render in green; future stages stay faint. Particles in the arrows
/// move forward by one slot every ~100ms so the diagram visibly
/// breathes while the engine runs.
///
/// `tick` is `loop_started.elapsed().as_millis()` — used as the
/// universal animation clock so every visual element stays in phase.
fn pipeline_flowchart_line(current: &str, done: &[String], tick: u128) -> Line<'static> {
    const STAGES: &[(&str, &str)] = &[
        ("vad", "VAD"),
        ("transcribe", "ASR"),
        ("translate", "MT"),
        ("stitch", "STITCH"),
        ("voice_consistency", "VOICE"),
        ("write_output_srt", "OUT"),
    ];
    // 8-step braille spinner. ~120 ms per step = 7 fps, easy on the eye.
    const SPINNER: &[char] = &['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'];
    let phase = (tick / 120) as usize;
    let spinner = SPINNER[phase % SPINNER.len()];
    // Two-slot particle: `· ` shifts to ` ·` every 200 ms. Visible
    // motion without distracting from the labels.
    let particle_phase = (tick / 200) as usize;
    let arrow = if particle_phase.is_multiple_of(2) {
        "·→"
    } else {
        " ⇒"
    };

    let mut spans: Vec<Span> = vec![Span::styled(
        "  stage  ",
        Style::default().fg(palette::MUTED),
    )];
    let any_active = STAGES.iter().any(|(k, _)| *k == current);
    for (i, (key, label)) in STAGES.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(
                arrow.to_string(),
                Style::default().fg(palette::FAINT),
            ));
        }
        // "translate" is the catch-all for the long MT stage even
        // before it's emitted as stage_complete, so promote nearby
        // stages too.
        let is_current = *key == current || (*key == "translate" && current == "transcribe");
        let is_done = done.iter().any(|s| s == key);
        let style = if is_current {
            Style::default()
                .fg(palette::ACCENT_2)
                .add_modifier(Modifier::BOLD)
        } else if is_done {
            Style::default().fg(palette::GREEN)
        } else {
            Style::default().fg(palette::FAINT)
        };
        let glyph = if is_current && any_active {
            // Lead the box with the spinner so the active stage feels
            // alive even when the rest of the engine is silent.
            format!(" {spinner} {label} ")
        } else {
            format!(" {label} ")
        };
        spans.push(Span::styled(glyph, style));
    }
    Line::from(spans)
}

fn render_log_tail(f: &mut ratatui::Frame, area: Rect, rs: &RunningState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::FAINT))
        .title(Span::styled(" log ", Style::default().fg(palette::MUTED)));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let limit = inner.height as usize;
    let log = &rs.runner.state.log;
    let lines: Vec<Line> = log
        .iter()
        .rev()
        .take(limit)
        .rev()
        .map(|line| {
            let trimmed = line.trim();
            let color = if trimmed.starts_with("error") {
                palette::RED
            } else if trimmed.starts_with("warning") {
                palette::ACCENT_2
            } else if trimmed.starts_with("ibvoid-doom-qlock") || trimmed.starts_with("translator")
            {
                palette::ACCENT
            } else if trimmed.starts_with("chunk_complete") || trimmed.starts_with("complete") {
                palette::GREEN
            } else {
                palette::MUTED
            };
            Line::from(Span::styled(line.clone(), Style::default().fg(color)))
        })
        .collect();
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

// ─── Result ──────────────────────────────────────────────────────────

fn render_result(f: &mut ratatui::Frame, area: Rect, app: &App, rs: &ResultState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Length(11),
            Constraint::Min(0),
        ])
        .split(area);

    render_animation_centered(f, rows[0], app, false);

    // Summary card.
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if rs.success {
            palette::GREEN
        } else {
            palette::RED
        }))
        .title(Span::styled(
            if rs.success {
                " run complete "
            } else {
                " run failed "
            },
            Style::default()
                .fg(if rs.success {
                    palette::GREEN
                } else {
                    palette::RED
                })
                .add_modifier(Modifier::BOLD),
        ));
    let summary_rect = centered_rect(rows[1], 96, rows[1].height);
    let inner = block.inner(summary_rect);
    f.render_widget(block, summary_rect);

    let mut lines: Vec<Line> = vec![
        Line::raw(""),
        kv("input", &shorten(rs.input.to_string_lossy().as_ref(), 80)),
        kv(
            "config",
            &format!(
                "{} → {}  ·  profile {}  ·  workers {}  ·  gpu {}",
                rs.config.source_lang,
                rs.config.target_lang,
                rs.config.profile,
                rs.config.workers,
                if rs.config.gpu { "on" } else { "off" }
            ),
        ),
        kv("duration", &format_duration(rs.duration)),
        kv("chunks", &rs.chunk_total.to_string()),
        kv(
            "gate q",
            &rs.quality
                .map(|q| format!("{:.2}", q))
                .unwrap_or_else(|| "—".into()),
        ),
    ];
    if let Some(msg) = &rs.last_message {
        lines.push(kv("last", msg));
    }
    if !rs.output_files.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            "outputs",
            Style::default().fg(palette::MUTED),
        )));
        for out in &rs.output_files {
            lines.push(Line::from(vec![
                Span::styled("  • ", Style::default().fg(palette::ACCENT)),
                Span::styled(
                    out.to_string_lossy().to_string(),
                    Style::default().fg(palette::TEXT),
                ),
            ]));
        }
    }
    f.render_widget(Paragraph::new(lines), inner);

    // SRT preview, if available.
    if !rs.srt_preview.is_empty() {
        render_srt_preview(f, rows[2], &rs.srt_preview);
    }
}

fn render_srt_preview(f: &mut ratatui::Frame, area: Rect, preview: &[String]) {
    let rect = centered_rect(area, 96, area.height);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::FAINT))
        .title(Span::styled(
            " subtitles preview ",
            Style::default().fg(palette::MUTED),
        ));
    let inner = block.inner(rect);
    f.render_widget(block, rect);

    let lines: Vec<Line> = preview
        .iter()
        .map(|line| {
            let trimmed = line.trim_end();
            let style = if trimmed.contains("-->") {
                Style::default().fg(palette::ACCENT)
            } else if trimmed.chars().all(|c| c.is_ascii_digit()) && !trimmed.is_empty() {
                Style::default().fg(palette::MUTED)
            } else {
                Style::default().fg(palette::TEXT)
            };
            Line::from(Span::styled(format!("  {}", trimmed), style))
        })
        .collect();
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

// ─── Shared chrome helpers ───────────────────────────────────────────

/// Dispatch the Running-screen animation panel through the active
/// `runner_viz::VizMode`. Original keeps the existing braille
/// playback. Emerge reveals that same playback cell-by-cell as
/// progress climbs. Generative paints the flow-field canvas the
/// `Viz::step` integration has been building each tick.
fn render_running_animation(f: &mut ratatui::Frame, area: Rect, app: &App, rs: &RunningState) {
    match app.viz.mode {
        crate::runner_viz::VizMode::Original => {
            render_animation_centered(f, area, app, false);
        }
        crate::runner_viz::VizMode::Emerge => render_emerge_panel(f, area, app, rs),
        crate::runner_viz::VizMode::Generative => render_generative_panel(f, area, app),
    }
}

/// Cell-by-cell reveal of the current animation frame. Each cell's
/// visibility is decided by a deterministic hash of (cell index,
/// input filename), so the same input always emerges in the same
/// order. Hidden cells render as empty braille (`⠀`) to preserve
/// the panel dimensions.
fn render_emerge_panel(f: &mut ratatui::Frame, area: Rect, app: &App, rs: &RunningState) {
    let frame = &app.current_anim().frames[app.frame_idx];
    let max_w = (frame.width + 2).min(area.width.saturating_sub(2));
    let max_h = (frame.height + 2).min(area.height.saturating_sub(1));
    if max_w < 6 || max_h < 4 {
        return;
    }
    let outer = Rect {
        x: area.x + (area.width.saturating_sub(max_w)) / 2,
        y: area.y + (area.height.saturating_sub(max_h)) / 2,
        width: max_w,
        height: max_h,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme_accent(app.easter.theme)))
        .title(Span::styled(
            " emerge ",
            Style::default().fg(palette::ACCENT_2),
        ));
    let inner = block.inner(outer);
    f.render_widget(block, outer);

    let progress = if rs.runner.state.chunk_total > 0 {
        rs.runner.state.chunk_current as f32 / rs.runner.state.chunk_total as f32
    } else {
        0.0
    };
    let cols = (inner.width as usize).min(frame.width as usize);
    let rows = (inner.height as usize).min(frame.height as usize);
    let mut lines = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut spans = Vec::with_capacity(cols);
        let mut buf = String::new();
        let mut run_idx: Option<u8> = None;
        for col in 0..cols {
            let off = row * frame.width as usize + col;
            let cell = &frame.cells[off];
            let (ch, r, g, b) =
                crate::runner_viz::cell_or_empty(cell, app.viz.emerge_visible(off, progress));
            let here = rgb_to_xterm256(r, g, b);
            if run_idx != Some(here) {
                if !buf.is_empty() {
                    spans.push(Span::styled(
                        std::mem::take(&mut buf),
                        Style::default().fg(Color::Indexed(run_idx.unwrap_or(231))),
                    ));
                }
                run_idx = Some(here);
            }
            buf.push(ch);
        }
        if !buf.is_empty() {
            spans.push(Span::styled(
                buf,
                Style::default().fg(Color::Indexed(run_idx.unwrap_or(231))),
            ));
        }
        lines.push(Line::from(spans));
    }
    let body_rect = Rect {
        x: inner.x + (inner.width.saturating_sub(cols as u16)) / 2,
        y: inner.y + (inner.height.saturating_sub(rows as u16)) / 2,
        width: cols as u16,
        height: rows as u16,
    };
    f.render_widget(Paragraph::new(lines), body_rect);
}

/// Flow-field particle visualization. The simulation runs in
/// `App::viz` (driven by main_loop's `viz.step`) and writes alpha
/// into a per-dot canvas; here we just sample it cell-by-cell and
/// emit the corresponding braille glyph with theme-tinted fade.
fn render_generative_panel(f: &mut ratatui::Frame, area: Rect, app: &App) {
    if area.width < 8 || area.height < 4 {
        return;
    }
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme_accent(app.easter.theme)))
        .title(Span::styled(
            " generative ",
            Style::default().fg(palette::PINK),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let cols = inner.width as usize;
    let rows = inner.height as usize;
    let base = theme_accent(app.easter.theme);
    let mut lines = Vec::with_capacity(rows);
    for cy in 0..rows {
        let mut spans = Vec::with_capacity(cols);
        let mut buf = String::new();
        let mut run_intensity: Option<u8> = None;
        for cx in 0..cols {
            let (ch, alpha) = app.viz.sample_cell(cx, cy);
            // Bucket alpha into 16 levels so adjacent cells with
            // imperceptible alpha differences share a span, just like
            // the xterm-256 quantiser does for video.
            let bucket = (alpha * 15.0) as u8;
            if run_intensity != Some(bucket) {
                if !buf.is_empty() {
                    let a = (run_intensity.unwrap_or(0) as f32) / 15.0;
                    spans.push(Span::styled(
                        std::mem::take(&mut buf),
                        Style::default().fg(crate::runner_viz::fade(base, a)),
                    ));
                }
                run_intensity = Some(bucket);
            }
            buf.push(ch);
        }
        if !buf.is_empty() {
            let a = (run_intensity.unwrap_or(0) as f32) / 15.0;
            spans.push(Span::styled(
                buf,
                Style::default().fg(crate::runner_viz::fade(base, a)),
            ));
        }
        lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn render_animation_centered(f: &mut ratatui::Frame, area: Rect, app: &App, hero_size: bool) {
    let frame = &app.current_anim().frames[app.frame_idx];
    // hero_size = true → use full animation dimensions; false → may
    // shrink to fit inside compact panels (Picker / Running / Result).
    let max_w = if hero_size {
        frame.width + 2
    } else {
        (frame.width + 2).min(area.width.saturating_sub(2))
    };
    let max_h = if hero_size {
        frame.height + 2
    } else {
        (frame.height + 2).min(area.height.saturating_sub(1))
    };
    let want_w = max_w.min(area.width);
    let want_h = max_h.min(area.height);
    if want_w < 6 || want_h < 4 {
        return;
    }
    let outer = Rect {
        x: area.x + (area.width.saturating_sub(want_w)) / 2,
        y: area.y + (area.height.saturating_sub(want_h)) / 2,
        width: want_w,
        height: want_h,
    };

    let base = theme_accent(app.easter.theme);
    let border_color = match app.theme_changed_at {
        Some(t) => {
            let elapsed = t.elapsed().as_millis() as u32;
            if elapsed > 600 {
                base
            } else {
                let mix = (elapsed as f32 / 600.0).clamp(0.0, 1.0);
                lerp_color(brighten(base), base, mix)
            }
        }
        None => base,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(
            Style::default().fg(border_color).add_modifier(
                if app
                    .theme_changed_at
                    .map(|t| t.elapsed().as_millis() < 600)
                    .unwrap_or(false)
                {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                },
            ),
        );
    let inner = block.inner(outer);
    f.render_widget(block, outer);

    let cols = (inner.width as usize).min(frame.width as usize);
    let rows = (inner.height as usize).min(frame.height as usize);
    let mut lines = Vec::with_capacity(rows);
    // Quantise RGB to xterm-256 + coalesce identical-palette runs.
    for row in 0..rows {
        let mut spans = Vec::with_capacity(cols);
        let mut buf = String::new();
        let mut run_idx: Option<u8> = None;
        for col in 0..cols {
            let off = row * frame.width as usize + col;
            let cell = &frame.cells[off];
            let here = rgb_to_xterm256(cell.r, cell.g, cell.b);
            if run_idx != Some(here) {
                if !buf.is_empty() {
                    let idx = run_idx.unwrap_or(231); // white
                    spans.push(Span::styled(
                        std::mem::take(&mut buf),
                        Style::default().fg(Color::Indexed(idx)),
                    ));
                }
                run_idx = Some(here);
            }
            buf.push(cell.ch);
        }
        if !buf.is_empty() {
            let idx = run_idx.unwrap_or(231);
            spans.push(Span::styled(buf, Style::default().fg(Color::Indexed(idx))));
        }
        lines.push(Line::from(spans));
    }
    let body = Paragraph::new(lines);
    let body_rect = Rect {
        x: inner.x + (inner.width.saturating_sub(cols as u16)) / 2,
        y: inner.y + (inner.height.saturating_sub(rows as u16)) / 2,
        width: cols as u16,
        height: rows as u16,
    };
    f.render_widget(body, body_rect);
}

fn render_tabs(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let mut spans: Vec<Span> = Vec::with_capacity(STATES.len() * 2);
    for (i, name) in STATES.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("  ·  ", Style::default().fg(palette::FAINT)));
        }
        let is_active = *name == app.current_state_name;
        let available = app.pack.contains_key(*name);
        let style = if is_active {
            Style::default()
                .fg(palette::ACCENT_2)
                .add_modifier(Modifier::BOLD)
        } else if available {
            Style::default().fg(palette::TEXT)
        } else {
            Style::default()
                .fg(palette::FAINT)
                .add_modifier(Modifier::DIM)
        };
        spans.push(Span::styled(*name, style));
    }
    let p = Paragraph::new(Line::from(spans)).alignment(Alignment::Center);
    f.render_widget(p, area);
}

fn render_progress(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let anim = app.current_anim();
    let total = anim.frames.len().max(1);
    let cur = app.frame_idx + 1;
    let bar_width = (anim.width as usize).min(area.width as usize / 2).max(20);
    let filled = (bar_width * cur) / total;
    let mut spans = Vec::with_capacity(bar_width + 4);
    for i in 0..bar_width {
        if i < filled {
            spans.push(Span::styled("▰", Style::default().fg(palette::ACCENT)));
        } else {
            spans.push(Span::styled("▱", Style::default().fg(palette::FAINT)));
        }
    }
    spans.push(Span::raw("   "));
    spans.push(Span::styled(
        format!("frame {} / {}", cur, total),
        Style::default().fg(palette::MUTED),
    ));
    let p = Paragraph::new(Line::from(spans)).alignment(Alignment::Center);
    f.render_widget(p, area);
}

fn render_hint(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let pairs: Vec<(&str, &str)> = match &app.screen {
        Screen::Greeting if app.current_anim_has_audio() => vec![
            ("Enter", "pick"),
            ("r", "run last"),
            ("p", "profile"),
            ("Tab", "next anim"),
            ("m", "mute"),
            ("+/-", "volume"),
            (":", "cmd"),
            ("q", "quit"),
        ],
        Screen::Greeting => vec![
            ("Enter", "pick"),
            ("r", "run last"),
            ("p", "profile"),
            ("Tab", "next anim"),
            (":", "cmd"),
            ("q", "quit"),
        ],
        Screen::Picker(_) => vec![
            ("Enter", "next"),
            ("Tab", "complete path"),
            ("↑↓", "recents"),
            ("Esc", "back"),
            ("q", "quit"),
        ],
        Screen::Configure(_) => vec![
            ("Tab", "next"),
            ("←→", "change"),
            ("Enter", "start"),
            ("Esc", "back"),
        ],
        Screen::Running(_) => vec![
            ("g", "viz mode"),
            ("c", "cancel"),
            ("q/Esc", "abort"),
            (":", "cmd"),
        ],
        Screen::Result(_) => vec![
            ("Enter", "new run"),
            ("r", "retry"),
            ("s", "save as"),
            ("o", "open folder"),
            ("q", "quit"),
        ],
    };
    let mut spans: Vec<Span> = Vec::new();
    for (i, (k, label)) in pairs.iter().enumerate() {
        if i > 0 {
            spans.push(Span::raw("    "));
        }
        spans.push(Span::styled(*k, Style::default().fg(palette::PINK)));
        spans.push(Span::raw(" "));
        spans.push(Span::styled(*label, Style::default().fg(palette::MUTED)));
    }
    let p = Paragraph::new(Line::from(spans)).alignment(Alignment::Center);
    f.render_widget(p, area);
}

// ─── Layout / formatting helpers ─────────────────────────────────────

fn centered_rect(area: Rect, want_w: u16, want_h: u16) -> Rect {
    let w = want_w.min(area.width);
    let h = want_h.min(area.height);
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

fn kv(key: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {key:<10}"), Style::default().fg(palette::MUTED)),
        Span::styled(value.to_string(), Style::default().fg(palette::TEXT)),
    ])
}

fn shorten(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else {
        let take = max.saturating_sub(1);
        let tail: String = chars
            .iter()
            .rev()
            .take(take)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        format!("…{}", tail)
    }
}

fn format_duration(d: Duration) -> String {
    let total = d.as_secs();
    let m = total / 60;
    let s = total % 60;
    if m > 0 {
        format!("{}m {:02}s", m, s)
    } else {
        format!("{}s", s)
    }
}

/// Snap a 24-bit RGB triple to an xterm-256 palette index.
///
/// Uses the 6×6×6 colour cube (indices 16–231) for chromatic pixels
/// and the 24-step grayscale ramp (232–255) for near-monochrome ones,
/// which gives finer detail in shadows and skin shading than the cube
/// alone. Output is in [16, 255]; index 16 is xterm "black" and is
/// emitted for both true black and very-dark grays.
///
/// Cube component bins follow the canonical xterm boundaries:
/// 0/95/135/175/215/255 with nearest-neighbour quantisation.
fn rgb_to_xterm256(r: u8, g: u8, b: u8) -> u8 {
    let max_c = r.max(g).max(b);
    let min_c = r.min(g).min(b);
    // Tight tolerance: only treat near-equal channels as gray. A loose
    // threshold here would push muted colours (skin, sky) onto the
    // grayscale ramp and lose chroma; 8/255 is the sweet spot empirically.
    if max_c.saturating_sub(min_c) <= 8 {
        let luma = ((r as u16 + g as u16 + b as u16) / 3) as u8;
        if luma < 8 {
            return 16;
        }
        if luma > 238 {
            return 231;
        }
        return 232 + (luma - 8) / 10;
    }
    let q = |c: u8| -> u8 {
        if c < 48 {
            0
        } else if c < 115 {
            1
        } else {
            (((c - 35) as u16) / 40).min(5) as u8
        }
    };
    16 + 36 * q(r) + 6 * q(g) + q(b)
}

#[cfg(test)]
mod tests {
    use super::rgb_to_xterm256;

    #[test]
    fn pure_black_maps_to_16() {
        assert_eq!(rgb_to_xterm256(0, 0, 0), 16);
    }

    #[test]
    fn pure_white_maps_to_231() {
        assert_eq!(rgb_to_xterm256(255, 255, 255), 231);
    }

    #[test]
    fn midgray_lands_on_grayscale_ramp() {
        let idx = rgb_to_xterm256(128, 128, 128);
        assert!((232..=255).contains(&idx), "got {idx}");
    }

    #[test]
    fn near_neighbour_rgb_collapses_to_same_index() {
        // The whole point of the quantiser: imperceptible RGB wobble
        // (within the same cube cell) must produce identical palette
        // indices, otherwise span-coalescing buys nothing on natural
        // footage. Both (150,70,70) and (152,73,71) sit inside cube
        // bucket (2,1,1) so they share index 16 + 36*2 + 6*1 + 1 = 95.
        let a = rgb_to_xterm256(150, 70, 70);
        let b = rgb_to_xterm256(152, 73, 71);
        assert_eq!(a, b, "neighbouring pixels should share a palette bucket");
    }

    #[test]
    fn distinct_colours_get_distinct_indices() {
        assert_ne!(rgb_to_xterm256(255, 0, 0), rgb_to_xterm256(0, 255, 0));
        assert_ne!(rgb_to_xterm256(255, 0, 0), rgb_to_xterm256(0, 0, 255));
    }

    #[test]
    fn cube_component_boundaries() {
        // Sanity-check the standard xterm cube boundaries.
        assert_eq!(rgb_to_xterm256(0, 0, 95), 16 + 1); // 0,0,95 = (0,0,1)
        assert_eq!(rgb_to_xterm256(95, 0, 0), 16 + 36); // 95,0,0 = (1,0,0)
        assert_eq!(rgb_to_xterm256(255, 255, 0), 16 + 36 * 5 + 6 * 5); // (5,5,0)
    }
}
