//! Subprocess driver for the Sub-Zero engine.
//!
//! The TUI does not embed the engine — it spawns `sub-zero.exe` with
//! `--events-file <path>` and tails the JSONL stream. This file owns
//! the spawn / wait / abort logic and the event parser that feeds the
//! Running screen.

use std::collections::VecDeque;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde::Deserialize;

/// One JSONL event from the engine.
///
/// Schema lives in `src/engine/events.rs` and friends:
/// the discriminator field is `event` (not `kind`). Real event names
/// in this codebase: `input_start`, `input_complete`, `chunk_start`,
/// `chunk_started`, `chunk_complete`, `chunk_timeout`, `chunk_failure`,
/// `stage_complete`, `vad_segment`, `vad_complete`, `asr_complete`,
/// `mt_complete`, `replan`. Every payload field below is optional and
/// only relevant for the matching event kind.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct EngineEvent {
    #[serde(default)]
    pub event: String,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default, alias = "chunk")]
    pub chunk_index: Option<u32>,
    #[serde(default, alias = "chunks")]
    pub chunk_total: Option<u32>,
    #[serde(default, alias = "elapsed", alias = "secs")]
    pub elapsed_secs: Option<f64>,
    #[serde(default, alias = "eta")]
    pub eta_secs: Option<f64>,
    #[serde(default, alias = "quality", alias = "health_score")]
    pub health_score: Option<f64>,
    #[serde(default, alias = "cue_count")]
    pub cues: Option<u32>,
    #[serde(default, alias = "output_file", alias = "out")]
    pub output: Option<String>,
    #[serde(default, alias = "input_file")]
    pub input: Option<String>,
    #[serde(default)]
    pub attempt: Option<u32>,
    #[serde(default)]
    pub success: Option<bool>,
    /// Nested per-event payload. The engine puts stage-specific stats
    /// here (voice-consistency aggregates, runtime stage timings, etc.)
    /// so we parse it as a generic value and dig into it from the
    /// stage-specific handlers.
    #[serde(default)]
    pub details: Option<serde_json::Value>,
}

/// State the Running screen reads from. Updated by every poll cycle.
#[derive(Debug, Default)]
pub struct EngineState {
    pub started_at: Option<Instant>,
    pub current_stage: Option<String>,
    pub stages_complete: Vec<String>,
    pub chunk_current: u32,
    pub chunk_total: u32,
    pub eta_secs: Option<f64>,
    pub quality: Option<f64>,
    pub log: VecDeque<String>,
    pub output_files: Vec<PathBuf>,
    pub finished: bool,
    pub success: Option<bool>,
    pub last_message: Option<String>,
    /// When set in the future, the animation should hold the given
    /// state name until that instant; used for victory / warning flashes.
    pub transient_anim_until: Option<Instant>,
    pub transient_anim_state: Option<&'static str>,
    pub anim_state: &'static str,
    /// Wall-clock timestamps of recent chunk_complete events.
    /// Bounded; the telemetry sparkline buckets them into the trailing
    /// few seconds for a live throughput readout.
    pub chunk_completions: VecDeque<Instant>,
    /// Voice-consistency aggregate from the corresponding pipeline
    /// stage (F.2 §6.3 / Theorem 4). `None` until the engine has
    /// emitted the `voice_consistency` stage_complete event.
    pub voice_mean_deviation: Option<f32>,
    pub voice_p95_deviation: Option<f32>,
    pub voice_max_deviation: Option<f32>,
    pub voice_speakers_observed: Option<u32>,
    pub voice_cues_scored: Option<u32>,
}

impl EngineState {
    fn push_log(&mut self, line: String) {
        const MAX: usize = 80;
        if self.log.len() == MAX {
            self.log.pop_front();
        }
        self.log.push_back(line);
    }

    fn flash(&mut self, name: &'static str, dur: Duration) {
        self.transient_anim_state = Some(name);
        self.transient_anim_until = Some(Instant::now() + dur);
    }

    /// Resolve the animation state to display this frame. Honours any
    /// active flash, otherwise returns the steady-state animation.
    pub fn effective_anim_state(&mut self) -> &'static str {
        if let Some(until) = self.transient_anim_until {
            if Instant::now() < until {
                return self.transient_anim_state.unwrap_or(self.anim_state);
            }
            self.transient_anim_until = None;
            self.transient_anim_state = None;
        }
        self.anim_state
    }
}

/// User-supplied engine settings collected on the Configure screen.
/// `EngineRunner::spawn` translates these into CLI flags.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub source_lang: String,
    pub target_lang: String,
    pub profile: String,    // fast | balanced | strict
    pub gpu: bool,
    pub workers: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            source_lang: "ja".into(),
            target_lang: "en".into(),
            profile: "fast".into(),
            gpu: true,
            workers: 1,
        }
    }
}

/// Spawned engine + its events tail. `Drop` ensures the child is
/// cleaned up so a TUI quit never leaves a zombie subprocess.
pub struct EngineRunner {
    child: Option<Child>,
    events_path: PathBuf,
    events_pos: u64,
    events_size: u64,
    pub state: EngineState,
    pub input_path: PathBuf,
    pub config: EngineConfig,
}

impl EngineRunner {
    /// Spawn the engine on `input_path` with the supplied config.
    /// Events go to a sibling temp file we own; the path is in
    /// `events_path` for inspection.
    pub fn spawn(
        engine_exe: &Path,
        input_path: &Path,
        config: &EngineConfig,
    ) -> io::Result<Self> {
        let events_path = events_temp_path(input_path);
        // Truncate any prior events file from a previous run.
        let _ = fs::write(&events_path, b"");

        let mut cmd = Command::new(engine_exe);
        cmd.arg(input_path)
            .arg("--source-lang").arg(&config.source_lang)
            .arg("--lang").arg(&config.target_lang)
            .arg("--offline")
            .arg("--transcribe")
            .arg("--workers").arg(config.workers.to_string())
            .arg("--mt-force-cpu")
            .arg("--mt-no-quality-floor")
            .arg("--profile").arg(&config.profile)
            .arg("--events-file").arg(&events_path)
            .arg("--events-json");
        if config.gpu {
            cmd.arg("--gpu");
        }
        cmd.stdin(Stdio::null())
           .stdout(Stdio::null())
           .stderr(Stdio::null());

        let child = cmd.spawn()?;
        let mut state = EngineState::default();
        state.started_at = Some(Instant::now());
        state.anim_state = "running";
        Ok(Self {
            child: Some(child),
            events_path,
            events_pos: 0,
            events_size: 0,
            state,
            input_path: input_path.to_path_buf(),
            config: config.clone(),
        })
    }

    /// Poll new events and process the child's exit status. Returns
    /// `true` when the engine has finished (cleanly or otherwise).
    pub fn poll(&mut self) -> bool {
        // Drain new event lines.
        for line in self.tail_events() {
            self.process_event_line(&line);
        }

        // Observe child exit.
        if let Some(child) = self.child.as_mut() {
            if let Ok(Some(status)) = child.try_wait() {
                let success = status.success();
                self.state.finished = true;
                self.state.success = Some(success);
                self.state.anim_state = if success { "complete" } else { "error" };
                self.state.flash(
                    if success { "complete" } else { "error" },
                    Duration::from_secs(10),
                );
                self.child = None;
                return true;
            }
        }
        false
    }

    fn tail_events(&mut self) -> Vec<String> {
        let meta = match fs::metadata(&self.events_path) {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        let size = meta.len();
        if size < self.events_size {
            self.events_pos = 0;
        }
        self.events_size = size;
        if size == self.events_pos {
            return Vec::new();
        }
        let mut f = match fs::File::open(&self.events_path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        if f.seek(SeekFrom::Start(self.events_pos)).is_err() {
            return Vec::new();
        }
        let mut buf = Vec::new();
        if f.read_to_end(&mut buf).is_err() {
            return Vec::new();
        }
        self.events_pos = size;
        let txt = String::from_utf8_lossy(&buf);
        txt.lines().map(|l| l.to_string()).collect()
    }

    fn process_event_line(&mut self, line: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }
        // Try strict JSON parse first; fall back to log line on failure.
        match serde_json::from_str::<EngineEvent>(trimmed) {
            Ok(ev) => self.apply_event(ev),
            Err(_) => self.state.push_log(trimmed.to_string()),
        }
    }

    fn apply_event(&mut self, ev: EngineEvent) {
        // Build a one-line log entry summarising the event.
        let summary = format_event_summary(&ev);
        self.state.push_log(summary);
        if let Some(msg) = ev.message.as_deref() {
            self.state.last_message = Some(msg.to_string());
        }

        match ev.event.as_str() {
            "input_start" => {
                self.state.anim_state = "running";
            }
            "stage_complete" => {
                if let Some(s) = ev.stage.clone() {
                    if !self.state.stages_complete.iter().any(|x| x == &s) {
                        self.state.stages_complete.push(s.clone());
                    }
                    self.state.current_stage = Some(s.clone());
                    // F.2 voice-consistency telemetry: the engine emits
                    // per-batch aggregates in the `details` payload of
                    // the voice_consistency stage_complete event. Pluck
                    // them out so the Running screen can show them in
                    // the live telemetry panel.
                    if s == "voice_consistency" {
                        if let Some(d) = ev.details.as_ref() {
                            self.state.voice_mean_deviation =
                                d.get("mean_deviation").and_then(|v| v.as_f64()).map(|v| v as f32);
                            self.state.voice_p95_deviation =
                                d.get("p95_deviation").and_then(|v| v.as_f64()).map(|v| v as f32);
                            self.state.voice_max_deviation =
                                d.get("max_deviation").and_then(|v| v.as_f64()).map(|v| v as f32);
                            self.state.voice_speakers_observed =
                                d.get("speakers_observed").and_then(|v| v.as_u64()).map(|v| v as u32);
                            self.state.voice_cues_scored =
                                d.get("cues_scored").and_then(|v| v.as_u64()).map(|v| v as u32);
                        }
                    }
                }
            }
            "chunk_start" | "chunk_started" => {
                if let Some(c) = ev.chunk_index { self.state.chunk_current = c.saturating_add(1); }
                if let Some(t) = ev.chunk_total { self.state.chunk_total = t; }
                self.state.current_stage = Some("transcribe".into());
            }
            "asr_complete" => {
                self.state.current_stage = Some("translate".into());
            }
            "mt_complete" => {
                self.state.current_stage = Some("stitch".into());
            }
            "vad_segment" | "vad_complete" => {
                self.state.current_stage = Some("vad".into());
            }
            "replan" => {
                self.state.current_stage = Some("replan".into());
                self.state.flash("warning", Duration::from_millis(700));
            }
            "chunk_complete" => {
                if let Some(c) = ev.chunk_index { self.state.chunk_current = c.saturating_add(1); }
                if let Some(t) = ev.chunk_total { self.state.chunk_total = t; }
                if let Some(q) = ev.health_score { self.state.quality = Some(q); }
                // Record completion timestamp for the telemetry
                // throughput sparkline. Keep the queue bounded so it
                // stays cheap to scan even across long runs.
                const MAX_COMPLETIONS: usize = 256;
                if self.state.chunk_completions.len() == MAX_COMPLETIONS {
                    self.state.chunk_completions.pop_front();
                }
                self.state.chunk_completions.push_back(Instant::now());
                // Predict ETA from per-chunk wall-time when total chunks are known.
                if let (Some(elapsed), Some(total)) = (ev.elapsed_secs, ev.chunk_total) {
                    let done = self.state.chunk_current.max(1) as f64;
                    let remaining = (total as f64 - done).max(0.0);
                    self.state.eta_secs = Some(elapsed * remaining);
                }
                self.state.flash("victory", Duration::from_millis(800));
            }
            "chunk_timeout" | "chunk_failure" => {
                self.state.flash("warning", Duration::from_millis(1500));
            }
            "input_complete" => {
                if let Some(out) = ev.output {
                    self.state.output_files.push(PathBuf::from(out));
                }
                self.state.success = ev.success.or(Some(true));
            }
            _ => {}
        }
    }

    /// Best-effort abort of the running engine. Sends a kill so the
    /// shutdown is hard; future iterations can swap in a control file.
    pub fn abort(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
            self.child = None;
            self.state.finished = true;
            self.state.success = Some(false);
            self.state.anim_state = "warning";
            self.state.last_message = Some("aborted by user".into());
            self.state.flash("warning", Duration::from_secs(3));
        }
    }

    #[allow(dead_code)]
    pub fn events_path(&self) -> &Path {
        &self.events_path
    }
}

impl Drop for EngineRunner {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn events_temp_path(input: &Path) -> PathBuf {
    let mut name = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("run")
        .to_string();
    name.push_str(".sub-zero-tui.events.jsonl");
    std::env::temp_dir().join(name)
}

/// Compact one-line summary used for the log tail. Keeps each event
/// readable without dumping the whole JSON payload.
fn format_event_summary(ev: &EngineEvent) -> String {
    match ev.event.as_str() {
        "input_start" => format!("input_start: {}",
            ev.input.clone().unwrap_or_default()),
        "input_complete" => format!("input_complete: {} ({:.1}s)",
            ev.output.clone().unwrap_or_default(),
            ev.elapsed_secs.unwrap_or(0.0)),
        "stage_complete" => format!("stage_complete: {} ({:.1}s)",
            ev.stage.clone().unwrap_or_default(),
            ev.elapsed_secs.unwrap_or(0.0)),
        "chunk_start" | "chunk_started" => format!("chunk_start: {}/{}",
            ev.chunk_index.unwrap_or(0).saturating_add(1),
            ev.chunk_total.unwrap_or(0)),
        "chunk_complete" => format!(
            "chunk_complete: {}/{}  q={:.2}  {:.1}s",
            ev.chunk_index.unwrap_or(0).saturating_add(1),
            ev.chunk_total.unwrap_or(0),
            ev.health_score.unwrap_or(0.0),
            ev.elapsed_secs.unwrap_or(0.0)),
        "chunk_timeout" => format!("chunk_timeout: chunk {}",
            ev.chunk_index.unwrap_or(0).saturating_add(1)),
        "chunk_failure" => format!("chunk_failure: chunk {} attempt {}",
            ev.chunk_index.unwrap_or(0).saturating_add(1),
            ev.attempt.unwrap_or(0)),
        "asr_complete" => format!("asr_complete: chunk {}",
            ev.chunk_index.unwrap_or(0).saturating_add(1)),
        "mt_complete" => format!("mt_complete: chunk {}",
            ev.chunk_index.unwrap_or(0).saturating_add(1)),
        "vad_segment" | "vad_complete" => ev.event.clone(),
        "replan" => format!("replan: {}",
            ev.message.clone().unwrap_or_default()),
        other => other.to_string(),
    }
}

/// Locate the Sub-Zero engine binary. Looks for it next to the TUI
/// executable first (the install-time invariant), then falls back to
/// the in-repo release path so dev builds work without setup.
pub fn locate_engine_binary() -> Option<PathBuf> {
    let exe_name = if cfg!(windows) { "sub-zero.exe" } else { "sub-zero" };
    if let Ok(self_exe) = std::env::current_exe() {
        if let Some(dir) = self_exe.parent() {
            let candidate = dir.join(exe_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let cwd = std::env::current_dir().ok()?;
    let candidates = [
        cwd.join("target").join("release").join(exe_name),
        cwd.join("target").join("debug").join(exe_name),
    ];
    candidates.into_iter().find(|p| p.is_file())
}
