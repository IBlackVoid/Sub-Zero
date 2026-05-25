use std::{error::Error, path::PathBuf, time::Instant};

use crossterm::event::{KeyCode, KeyEvent};
use zeroize::Zeroize;

use crate::engine::{locate_engine_binary, EngineRunner};
use crate::paths::{complete_path, open_directory, sanitize_path_input};

use super::{
    App, ConfigField, ConfigureState, PickerState, RunningState, Screen, GREETING_DEFAULT_STATE,
    LANGS, PROFILES, STATES,
};

type KeyResult = Result<(), Box<dyn Error>>;
const EXLINE_MAX_LEN: usize = 512;

enum KeyFlow {
    Continue(Box<Screen>),
    Exit(KeyResult),
}

impl KeyFlow {
    fn continue_with(screen: Screen) -> Self {
        Self::Continue(Box::new(screen))
    }
}

/// Vim-style ex-line input handler.
///
/// Returns `true` when the keystroke was consumed by the ex-line and
/// must not be dispatched to the rest of the app.
pub(super) fn handle_exline_key(app: &mut App, k: KeyEvent) -> bool {
    if app.exline.is_none() {
        let on_text_input = matches!(&app.screen, Screen::Picker(_));
        if !on_text_input && matches!(k.code, KeyCode::Char(':')) {
            app.exline = Some(String::new());
            return true;
        }
        return false;
    }

    let buf = app.exline.as_mut().expect("ex-line open");
    match k.code {
        KeyCode::Esc => {
            app.exline = None;
        }
        KeyCode::Backspace => {
            buf.pop();
        }
        KeyCode::Enter => {
            let mut phrase = std::mem::take(buf);
            app.exline = None;
            let msg = app.try_unlock_phrase(&phrase);
            phrase.zeroize();
            app.exline = Some(format!("\x00{msg}"));
        }
        KeyCode::Char(c) => {
            if buf.starts_with('\x00') {
                buf.clear();
            }
            if buf.len() < EXLINE_MAX_LEN {
                buf.push(c);
            }
        }
        _ => {}
    }
    true
}

/// Hidden panel hotkey handler. Only invoked while `easter.panel_open`.
/// Returns true when the keystroke was consumed.
pub(super) fn handle_panel_key(app: &mut App, k: KeyEvent) -> bool {
    match k.code {
        KeyCode::Char('t') => {
            app.easter.cycle_theme();
            app.theme_changed_at = Some(Instant::now());
            app.prefs.theme = app.easter.theme.ident().to_string();
            let _ = app.prefs.save();
            true
        }
        KeyCode::Char('b') => {
            app.easter.toggle_boss();
            true
        }
        KeyCode::Char('p') | KeyCode::Esc => {
            app.easter.panel_open = false;
            true
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if let Some(panel) = app.easter.panel.as_ref() {
                let n = panel.manifest.items.len();
                if n > 0 {
                    app.easter.panel_cursor = (app.easter.panel_cursor + 1) % n;
                }
            }
            true
        }
        KeyCode::Up | KeyCode::Char('k') => {
            if let Some(panel) = app.easter.panel.as_ref() {
                let n = panel.manifest.items.len();
                if n > 0 {
                    app.easter.panel_cursor = (app.easter.panel_cursor + n - 1) % n;
                }
            }
            true
        }
        KeyCode::Enter => {
            if app.easter.panel.is_some() {
                let cursor = app.easter.panel_cursor;
                let name = app
                    .pack
                    .keys()
                    .filter(|k| k.starts_with("secret_a_"))
                    .nth(cursor)
                    .cloned();
                if let Some(name) = name {
                    app.force_anim(&name);
                    app.easter.panel_open = false;
                }
            }
            true
        }
        _ => false,
    }
}

/// Returns Some(result) when the loop should exit; otherwise None.
pub(super) fn handle_key(app: &mut App, k: KeyEvent) -> Option<KeyResult> {
    let screen = std::mem::replace(&mut app.screen, Screen::Greeting);
    match dispatch_key(app, screen, k) {
        KeyFlow::Continue(next) => {
            app.screen = *next;
            None
        }
        KeyFlow::Exit(result) => Some(result),
    }
}

fn dispatch_key(app: &mut App, screen: Screen, k: KeyEvent) -> KeyFlow {
    match screen {
        Screen::Greeting => handle_greeting_key(app, k),
        Screen::Picker(ps) => handle_picker_key(app, ps, k),
        Screen::Configure(cs) => handle_configure_key(app, cs, k),
        Screen::Running(rs) => handle_running_key(app, rs, k),
        Screen::Result(rs) => handle_result_key(app, rs, k),
    }
}

fn handle_greeting_key(app: &mut App, k: KeyEvent) -> KeyFlow {
    match k.code {
        KeyCode::Esc if app.anim_override => {
            app.release_anim_override();
        }
        KeyCode::Char('q') | KeyCode::Esc => return KeyFlow::Exit(Ok(())),
        KeyCode::Enter => {
            return KeyFlow::continue_with(Screen::Picker(PickerState {
                input: String::new(),
                error: None,
                selected_recent: None,
            }));
        }
        KeyCode::Tab => app.cycle_anim(1),
        KeyCode::BackTab => app.cycle_anim(-1),
        KeyCode::Char(c) if ('1'..='7').contains(&c) => {
            let idx = (c as u8 - b'1') as usize;
            if let Some(name) = STATES.get(idx) {
                app.force_anim(name);
            }
        }
        KeyCode::Char('m') if app.current_anim_has_audio() => {
            if let Some(audio) = app.audio.as_mut() {
                audio.toggle_mute();
            }
        }
        KeyCode::Char('+') | KeyCode::Char('=') if app.current_anim_has_audio() => {
            if let Some(audio) = app.audio.as_mut() {
                audio.nudge_volume(0.05);
            }
        }
        KeyCode::Char('-') | KeyCode::Char('_') if app.current_anim_has_audio() => {
            if let Some(audio) = app.audio.as_mut() {
                audio.nudge_volume(-0.05);
            }
        }
        KeyCode::Char('p') => {
            let next = cycle_profile(&app.last_config.profile);
            app.last_config.profile = next.to_string();
            app.exline = Some(format!("\x00profile -> {next}"));
        }
        KeyCode::Char('r') => {
            if let Some(entry) = app.recents.entries.last().cloned() {
                if entry.path.is_file() {
                    return KeyFlow::continue_with(Screen::Configure(ConfigureState {
                        input_path: entry.path,
                        config: app.last_config.clone(),
                        cursor: ConfigField::Profile,
                    }));
                }
                app.exline = Some(format!("\x00recent missing: {}", entry.path.display()));
            } else {
                app.exline = Some("\x00no recent input".to_string());
            }
        }
        _ => {}
    }
    KeyFlow::continue_with(Screen::Greeting)
}

fn handle_picker_key(app: &mut App, mut ps: PickerState, k: KeyEvent) -> KeyFlow {
    match k.code {
        KeyCode::Esc => {
            app.switch_anim(GREETING_DEFAULT_STATE);
            return KeyFlow::continue_with(Screen::Greeting);
        }
        KeyCode::Char('q') if ps.input.is_empty() && ps.selected_recent.is_none() => {
            return KeyFlow::Exit(Ok(()));
        }
        KeyCode::Up => {
            let n = app.recents.entries.len();
            if n > 0 {
                ps.selected_recent = Some(match ps.selected_recent {
                    None => n - 1,
                    Some(0) => 0,
                    Some(i) => i - 1,
                });
            }
        }
        KeyCode::Down => {
            let n = app.recents.entries.len();
            if n > 0 {
                ps.selected_recent = Some(match ps.selected_recent {
                    None => 0,
                    Some(i) if i + 1 < n => i + 1,
                    Some(_) => return KeyFlow::continue_with(Screen::Picker(ps)),
                });
            }
        }
        KeyCode::Tab => {
            ps.input = sanitize_path_input(&ps.input);
            if let Some(completed) = complete_path(&ps.input) {
                ps.input = completed;
            }
            ps.error = None;
        }
        KeyCode::Enter => {
            let raw = sanitize_path_input(&ps.input);
            let path = if let Some(idx) = ps.selected_recent {
                app.recents
                    .entries
                    .get(idx)
                    .map(|r| r.path.clone())
                    .unwrap_or_else(|| PathBuf::from(&raw))
            } else {
                PathBuf::from(&raw)
            };
            if !path.is_file() {
                ps.error = Some(format!("not a file: {}", path.display()));
            } else {
                app.recents.record(&path);
                return KeyFlow::continue_with(Screen::Configure(ConfigureState {
                    input_path: path,
                    config: app.last_config.clone(),
                    cursor: ConfigField::Profile,
                }));
            }
        }
        KeyCode::Backspace => {
            if ps.selected_recent.is_some() {
                ps.selected_recent = None;
            } else {
                ps.input.pop();
            }
            ps.error = None;
        }
        KeyCode::Char(c) => {
            ps.selected_recent = None;
            ps.input.push(c);
            ps.error = None;
        }
        _ => {}
    }
    KeyFlow::continue_with(Screen::Picker(ps))
}

fn handle_configure_key(app: &mut App, mut cs: ConfigureState, k: KeyEvent) -> KeyFlow {
    match k.code {
        KeyCode::Esc => {
            return KeyFlow::continue_with(Screen::Picker(PickerState {
                input: cs.input_path.to_string_lossy().to_string(),
                error: None,
                selected_recent: None,
            }));
        }
        KeyCode::Char('q') => return KeyFlow::Exit(Ok(())),
        KeyCode::Tab | KeyCode::Down => cs.cursor = cs.cursor.next(),
        KeyCode::BackTab | KeyCode::Up => cs.cursor = cs.cursor.prev(),
        KeyCode::Left => adjust_config(&mut cs, -1),
        KeyCode::Right => adjust_config(&mut cs, 1),
        KeyCode::Enter => match cs.cursor {
            ConfigField::Start => return KeyFlow::continue_with(start_engine(app, &cs)),
            _ => cs.cursor = cs.cursor.next(),
        },
        _ => {}
    }
    KeyFlow::continue_with(Screen::Configure(cs))
}

fn start_engine(app: &mut App, cs: &ConfigureState) -> Screen {
    let engine = match locate_engine_binary() {
        Some(path) => path,
        None => {
            return Screen::Picker(PickerState {
                input: cs.input_path.to_string_lossy().into(),
                error: Some(
                    "sub-zero engine binary not found (build with `cargo build --release`)".into(),
                ),
                selected_recent: None,
            });
        }
    };

    let cfg = cs.config.clone();
    let path = cs.input_path.clone();
    match EngineRunner::spawn(&engine, &path, &cfg) {
        Ok(runner) => {
            app.last_config = cfg.clone();
            app.viz.reset_for_run();
            app.switch_anim("running");
            Screen::Running(Box::new(RunningState {
                runner,
                config: cfg,
            }))
        }
        Err(error) => Screen::Picker(PickerState {
            input: path.to_string_lossy().into(),
            error: Some(format!("failed to spawn engine: {error}")),
            selected_recent: None,
        }),
    }
}

fn handle_running_key(app: &mut App, mut rs: Box<RunningState>, k: KeyEvent) -> KeyFlow {
    match k.code {
        KeyCode::Char('q') | KeyCode::Esc | KeyCode::Char('c') => rs.runner.abort(),
        KeyCode::Char('g') => {
            app.viz.cycle_mode();
            app.exline = Some(format!("\x00viz -> {}", app.viz.mode.label()));
        }
        _ => {}
    }
    KeyFlow::continue_with(Screen::Running(rs))
}

fn handle_result_key(app: &mut App, rs: super::ResultState, k: KeyEvent) -> KeyFlow {
    match k.code {
        KeyCode::Char('q') | KeyCode::Esc => return KeyFlow::Exit(Ok(())),
        KeyCode::Enter => {
            app.switch_anim(GREETING_DEFAULT_STATE);
            return KeyFlow::continue_with(Screen::Greeting);
        }
        KeyCode::Char('r') => {
            if let Some(engine) = locate_engine_binary() {
                let cfg = rs.config.clone();
                let path = rs.input.clone();
                if let Ok(runner) = EngineRunner::spawn(&engine, &path, &cfg) {
                    app.viz.reset_for_run();
                    app.switch_anim("running");
                    return KeyFlow::continue_with(Screen::Running(Box::new(RunningState {
                        runner,
                        config: cfg,
                    })));
                }
            }
        }
        KeyCode::Char('o') => {
            let target = rs
                .output_files
                .first()
                .and_then(|p| p.parent())
                .or_else(|| rs.input.parent())
                .map(|p| p.to_path_buf());
            if let Some(dir) = target {
                open_directory(&dir);
            }
        }
        KeyCode::Char('s') => {
            if let Some(out) = rs.output_files.first() {
                let default = out
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "output.srt".to_string());
                app.exline = Some(format!("save {default}"));
            }
        }
        _ => {}
    }
    KeyFlow::continue_with(Screen::Result(rs))
}

fn adjust_config(cs: &mut ConfigureState, delta: i32) {
    match cs.cursor {
        ConfigField::SourceLang => cycle_in(LANGS, &mut cs.config.source_lang, delta),
        ConfigField::TargetLang => cycle_in(LANGS, &mut cs.config.target_lang, delta),
        ConfigField::Profile => cycle_in(PROFILES, &mut cs.config.profile, delta),
        ConfigField::Gpu => {
            cs.config.gpu = !cs.config.gpu;
        }
        ConfigField::Workers => {
            let new = (cs.config.workers as i32 + delta).clamp(1, 8);
            cs.config.workers = new as u32;
        }
        ConfigField::Start => {}
    }
}

fn cycle_in(options: &[&str], current: &mut String, delta: i32) {
    let idx = options
        .iter()
        .position(|s| *s == current.as_str())
        .unwrap_or(0) as i32;
    let n = options.len() as i32;
    let next = ((idx + delta).rem_euclid(n)) as usize;
    *current = options[next].to_string();
}

fn cycle_profile(current: &str) -> &'static str {
    match current {
        "fast" => "balanced",
        "balanced" => "strict",
        _ => "fast",
    }
}
