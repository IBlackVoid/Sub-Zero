use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    WhisperCpp,
    PyWhisper,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityProfile {
    Fast,
    Balanced,
    Strict,
}

impl QualityProfile {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "fast" => Ok(Self::Fast),
            "balanced" => Ok(Self::Balanced),
            "strict" => Ok(Self::Strict),
            other => Err(format!(
                "invalid profile: {other} (expected one of: fast, balanced, strict)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Balanced => "balanced",
            Self::Strict => "strict",
        }
    }

    pub fn max_chunk_retries(self) -> usize {
        match self {
            Self::Fast => 0,
            Self::Balanced => 1,
            Self::Strict => 2,
        }
    }
}

// Prefer `python3`, but fall back to `python` on Windows where `python3.exe` may be a Store stub.
const PYTHON_CANDIDATES: &[&str] = &["python3", "python"];

#[derive(Debug, Clone)]
pub struct TranscribeConfig {
    pub enabled: bool,
    pub whisper_bin: Option<PathBuf>,
    pub whisper_model: Option<PathBuf>,
    pub source_lang: String,
    pub target_lang: String,
    pub whisper_args: Vec<String>,
    pub vad: bool,
    pub vad_threshold_db: f64,
    pub vad_min_silence: f64,
    pub vad_pad: f64,
    pub gpu: bool,
    pub require_gpu: bool,
    pub quality_profile: QualityProfile,
}

#[derive(Debug)]
pub struct Transcriber {
    backend: Backend,
    config: TranscribeConfig,
    whisper_cpp_bin: Option<PathBuf>,
    whisper_cpp_model: Option<PathBuf>,
    python_bin: Option<String>,
    py_model_name: String,
    py_model_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub srt_path: PathBuf,
    pub audio_wav_path: PathBuf,
}

impl Transcriber {
    pub fn new(config: TranscribeConfig) -> Result<Option<Self>, String> {
        if !config.enabled {
            return Ok(None);
        }

        let mut config = config;

        if let Some(explicit) = &config.whisper_bin {
            if !explicit.is_file() {
                return Err(format!(
                    "whisper binary does not exist: {}",
                    explicit.display()
                ));
            }
        }

        let whisper_cpp_bin = resolve_whisper_cpp_bin(&config.whisper_bin);
        if let Some(bin) = whisper_cpp_bin {
            if config.whisper_model.is_none() {
                config.whisper_model =
                    discover_default_whisper_cpp_model_path_for_profile(config.quality_profile);
            }
            let Some(model) = &config.whisper_model else {
                return Err(
                    "missing --whisper-model for --transcribe (whisper.cpp backend).".to_string(),
                );
            };
            if !model.is_file() {
                return Err(format!(
                    "whisper model file does not exist: {}",
                    model.display()
                ));
            }

            return Ok(Some(Self {
                backend: Backend::WhisperCpp,
                whisper_cpp_bin: Some(bin),
                whisper_cpp_model: Some(model.to_path_buf()),
                config,
                python_bin: None,
                py_model_name: String::new(),
                py_model_dir: None,
            }));
        }

        let python_bin = python_whisper_available()?;
        let Some(python_bin) = python_bin else {
            return Err(
                "no whisper.cpp binary found, and python whisper is not available. Install whisper.cpp or install python whisper (pip install openai-whisper).".to_string(),
            );
        };

        // Python whisper backend: pick the strongest locally-available model
        // for the selected quality profile.
        let py_model_name = discover_default_py_whisper_model_name(config.quality_profile);
        let py_model_dir = discover_default_py_whisper_model_dir(&py_model_name);
        if py_model_dir.is_none() {
            return Err(format!(
                "python whisper model not found for {py_model_name}. Put it at ./models/whisper/{py_model_name}.pt or set SUB_ZERO_PYWHISPER_MODEL_DIR."
            ));
        }

        Ok(Some(Self {
            backend: Backend::PyWhisper,
            config,
            whisper_cpp_bin: None,
            whisper_cpp_model: None,
            python_bin: Some(python_bin),
            py_model_name,
            py_model_dir,
        }))
    }

    pub fn transcribe_video_to_srt(&self, video: &Path) -> Result<TranscriptionResult, String> {
        let temp_dir = create_temp_dir(video)?;
        let wav_path = temp_dir.join("audio.wav");
        extract_audio_to_wav(video, &wav_path)?;

        let srt_path = self.transcribe_wav_to_srt_internal(&wav_path, &temp_dir, None)?;

        Ok(TranscriptionResult {
            srt_path,
            audio_wav_path: wav_path,
        })
    }

    /// Transcribe an already-extracted mono 16k WAV.
    /// This avoids redundant ffmpeg extraction in chunked/parallel pipelines.
    #[allow(dead_code)]
    pub fn transcribe_wav_to_srt(&self, wav_path: &Path) -> Result<PathBuf, String> {
        if !wav_path.is_file() {
            return Err(format!("wav input does not exist: {}", wav_path.display()));
        }
        let out_dir = wav_path
            .parent()
            .ok_or_else(|| format!("wav path has no parent directory: {}", wav_path.display()))?;
        self.transcribe_wav_to_srt_internal(wav_path, out_dir, None)
    }

    pub fn transcribe_wav_to_srt_with_timeout(
        &self,
        wav_path: &Path,
        timeout_secs: f64,
    ) -> Result<PathBuf, String> {
        if !wav_path.is_file() {
            return Err(format!("wav input does not exist: {}", wav_path.display()));
        }
        let out_dir = wav_path
            .parent()
            .ok_or_else(|| format!("wav path has no parent directory: {}", wav_path.display()))?;
        let timeout = if timeout_secs.is_finite() && timeout_secs > 0.0 {
            Some(Duration::from_secs_f64(timeout_secs))
        } else {
            None
        };
        self.transcribe_wav_to_srt_internal(wav_path, out_dir, timeout)
    }

    fn transcribe_wav_to_srt_internal(
        &self,
        wav_path: &Path,
        out_dir: &Path,
        timeout: Option<Duration>,
    ) -> Result<PathBuf, String> {
        match self.backend {
            Backend::WhisperCpp => {
                let whisper_bin = self
                    .whisper_cpp_bin
                    .as_ref()
                    .expect("validated in constructor");
                let whisper_model = self
                    .whisper_cpp_model
                    .as_ref()
                    .expect("validated in constructor");
                let stem = wav_path
                    .file_stem()
                    .and_then(OsStr::to_str)
                    .unwrap_or("audio");
                let out_prefix = out_dir.join(format!("{stem}_whisper"));

                run_whisper_cpp(
                    whisper_bin,
                    whisper_model,
                    wav_path,
                    &out_prefix,
                    &self.config.source_lang,
                    self.config.target_lang.eq_ignore_ascii_case("en"),
                    &self.config.whisper_args,
                    timeout,
                )?;

                let srt_path = out_prefix.with_extension("srt");
                if !srt_path.is_file() {
                    return Err(format!(
                        "transcription completed but SRT was not produced: {}",
                        srt_path.display()
                    ));
                }
                Ok(srt_path)
            }
            Backend::PyWhisper => {
                let model_dir = self
                    .py_model_dir
                    .as_ref()
                    .expect("validated in constructor");
                let result = run_python_whisper(
                    wav_path,
                    out_dir,
                    &self.config,
                    &self.py_model_name,
                    model_dir,
                    self.python_bin
                        .as_deref()
                        .expect("validated in constructor"),
                    timeout,
                );
                match result {
                    Ok(path) => Ok(path),
                    Err(error)
                        if self.config.gpu
                            && !self.config.require_gpu
                            && is_cuda_oom_transcribe_error(&error) =>
                    {
                        eprintln!(
                            "warning: whisper CUDA OOM detected; retrying transcription on CPU with the same model."
                        );
                        let mut cpu_retry = self.config.clone();
                        cpu_retry.gpu = false;
                        cpu_retry.whisper_args =
                            sanitize_whisper_args_for_cpu_retry(&cpu_retry.whisper_args);
                        run_python_whisper(
                            wav_path,
                            out_dir,
                            &cpu_retry,
                            &self.py_model_name,
                            model_dir,
                            self.python_bin
                                .as_deref()
                                .expect("validated in constructor"),
                            timeout,
                        )
                    }
                    Err(error) => Err(error),
                }
            }
        }
    }
}

fn resolve_whisper_cpp_bin(explicit: &Option<PathBuf>) -> Option<PathBuf> {
    if let Some(explicit) = explicit {
        if explicit.is_file() {
            return Some(explicit.to_path_buf());
        }
        return None;
    }

    // Common whisper.cpp executable names.
    // Prefer the unambiguous "whisper-cli" first; bare "whisper" is checked
    // last because it collides with the Python `openai-whisper` CLI wrapper.
    let candidates = [
        "whisper-cli",
        "whisper-cli.exe",
        "whisper",
        "whisper.exe",
        "main",
        "main.exe",
    ];

    let path = find_in_path(&candidates)?;

    // Guard: reject Python scripts masquerading as whisper.cpp.
    // The openai-whisper pip package installs a `whisper` entry-point that is
    // a Python script, not the native whisper.cpp binary.  Reading the first
    // two bytes is enough — ELF starts with 0x7F 'E', PE with 'MZ', and
    // Python scripts with '#!'.
    if is_python_script(&path) {
        return None;
    }

    Some(path)
}

/// Returns `true` when the file starts with a `#!` shebang (i.e. it is an
/// interpreted script, not a compiled binary).
fn is_python_script(path: &Path) -> bool {
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    use std::io::Read;
    let mut header = [0u8; 2];
    if std::io::BufReader::new(file)
        .read_exact(&mut header)
        .is_err()
    {
        return false;
    }
    // Shebang: '#!'
    header == [b'#', b'!']
}

fn discover_default_whisper_cpp_model_path_for_profile(profile: QualityProfile) -> Option<PathBuf> {
    // Explicit env var wins.
    if let Some(path) = std::env::var_os("SUB_ZERO_WHISPER_MODEL") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }

    let candidates: &[&str] = match profile {
        QualityProfile::Strict => &[
            "models/ggml-large-v3.bin",
            "models/ggml-large-v2.bin",
            "models/ggml-large.bin",
            "models/ggml-medium.bin",
            "models/ggml-small.bin",
            "models/ggml-base.bin",
            "ggml-large-v3.bin",
            "ggml-large-v2.bin",
            "ggml-large.bin",
            "ggml-medium.bin",
            "ggml-small.bin",
            "ggml-base.bin",
        ],
        QualityProfile::Balanced => &[
            "models/ggml-medium.bin",
            "models/ggml-small.bin",
            "models/ggml-base.bin",
            "models/ggml-large-v3.bin",
            "models/ggml-large-v2.bin",
            "models/ggml-large.bin",
            "ggml-medium.bin",
            "ggml-small.bin",
            "ggml-base.bin",
            "ggml-large-v3.bin",
            "ggml-large-v2.bin",
            "ggml-large.bin",
        ],
        QualityProfile::Fast => &[
            "models/ggml-small.bin",
            "models/ggml-base.bin",
            "models/ggml-medium.bin",
            "models/ggml-large-v3.bin",
            "models/ggml-large-v2.bin",
            "models/ggml-large.bin",
            "ggml-small.bin",
            "ggml-base.bin",
            "ggml-medium.bin",
            "ggml-large-v3.bin",
            "ggml-large-v2.bin",
            "ggml-large.bin",
        ],
    };
    for candidate in candidates {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }

    None
}

fn python_whisper_available() -> Result<Option<String>, String> {
    let mut candidates = Vec::<String>::new();

    // Optional override to force a specific Python interpreter.
    if let Ok(explicit) = std::env::var("SUB_ZERO_PYTHON_BIN") {
        let explicit = explicit.trim();
        if !explicit.is_empty() {
            candidates.push(explicit.to_string());
        }
    }

    // Prefer a repo-local venv if present (keeps installs self-contained).
    // On WSL a Windows .venv/Scripts/python.exe may exist as a file but fail
    // to execute, so we only add candidates that are native to the current OS.
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cfg!(windows) {
        let venv_win = cwd.join(".venv").join("Scripts").join("python.exe");
        if venv_win.is_file() {
            candidates.push(venv_win.to_string_lossy().to_string());
        }
    } else {
        let venv_unix_py3 = cwd.join(".venv").join("bin").join("python3");
        if venv_unix_py3.is_file() {
            candidates.push(venv_unix_py3.to_string_lossy().to_string());
        }
        let venv_unix_py = cwd.join(".venv").join("bin").join("python");
        if venv_unix_py.is_file() {
            candidates.push(venv_unix_py.to_string_lossy().to_string());
        }
    }

    candidates.extend(PYTHON_CANDIDATES.iter().map(|c| c.to_string()));

    // De-dup while preserving order.
    let mut unique = Vec::<String>::new();
    for c in candidates {
        if !unique.iter().any(|u| u == &c) {
            unique.push(c);
        }
    }

    let mut spawned_any = false;
    let mut spawn_errors = Vec::<String>::new();

    for candidate in &unique {
        let status = Command::new(candidate)
            .arg("-c")
            .arg("import whisper")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        match status {
            Ok(status) => {
                spawned_any = true;
                if status.success() {
                    return Ok(Some(candidate.to_string()));
                }
            }
            Err(e) => spawn_errors.push(format!("{candidate}: {e}")),
        }
    }

    if !spawned_any {
        return Err(format!(
            "failed to spawn a Python interpreter (tried: {}): {}",
            unique.join(", "),
            spawn_errors.join("; ")
        ));
    }

    Ok(None)
}

fn discover_default_py_whisper_model_name(profile: QualityProfile) -> String {
    if let Ok(name) = std::env::var("SUB_ZERO_PYWHISPER_MODEL") {
        if !name.trim().is_empty() {
            return name.trim().to_string();
        }
    }

    let candidates: &[&str] = match profile {
        QualityProfile::Strict => &["large-v3", "large-v2", "large", "medium", "small", "base"],
        QualityProfile::Balanced => &["medium", "small", "base", "large-v3", "large-v2", "large"],
        QualityProfile::Fast => &["small", "base", "medium", "large-v3", "large-v2", "large"],
    };

    for candidate in candidates {
        if discover_default_py_whisper_model_dir(candidate).is_some() {
            return (*candidate).to_string();
        }
    }

    "base".to_string()
}

fn discover_default_py_whisper_model_dir(model_name: &str) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("SUB_ZERO_PYWHISPER_MODEL_DIR") {
        let dir = PathBuf::from(dir);
        if dir.join(format!("{model_name}.pt")).is_file() {
            return Some(dir);
        }
    }

    let local = PathBuf::from("models/whisper");
    if local.join(format!("{model_name}.pt")).is_file() {
        return Some(local);
    }

    // Common caches for this environment (WSL + Windows user cache).
    if let Ok(user_profile) = std::env::var("USERPROFILE") {
        let cache = PathBuf::from(user_profile).join(".cache/whisper");
        if cache.join(format!("{model_name}.pt")).is_file() {
            return Some(cache);
        }
    }

    let home_cache = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/whisper");
    if home_cache.join(format!("{model_name}.pt")).is_file() {
        return Some(home_cache);
    }

    None
}

fn create_temp_dir(video: &Path) -> Result<PathBuf, String> {
    let stem = video.file_stem().and_then(OsStr::to_str).unwrap_or("video");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let path = std::env::temp_dir().join(format!("sub_zero_transcribe_{stem}_{stamp}"));
    std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
    Ok(path)
}

fn extract_audio_to_wav(video: &Path, wav_out: &Path) -> Result<(), String> {
    let ffmpeg = find_in_path(&["ffmpeg", "ffmpeg.exe"])
        .ok_or_else(|| "ffmpeg not found in PATH (required for --transcribe)".to_string())?;

    let status = Command::new(ffmpeg)
        .arg("-hide_banner")
        .arg("-nostdin")
        .arg("-v")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(video)
        .arg("-vn")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-f")
        .arg("wav")
        .arg(wav_out)
        .status()
        .map_err(|e| format!("failed to spawn ffmpeg: {e}"))?;

    if !status.success() {
        return Err(format!("ffmpeg failed with status: {status}"));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct VadInterval {
    pub start: f64,
    pub end: f64,
}

pub(crate) fn detect_speech_intervals_from_wav(
    wav_path: &Path,
    threshold_db: f64,
    min_silence: f64,
    pad: f64,
    max_seconds: Option<f64>,
) -> Result<Vec<VadInterval>, String> {
    let mut duration = ffprobe_duration_seconds(wav_path)?;
    if let Some(limit) = max_seconds {
        if limit.is_finite() && limit > 0.0 {
            duration = duration.min(limit);
        }
    }
    let silence = ffmpeg_silencedetect(wav_path, threshold_db, min_silence, duration)?;
    let mut speech = invert_intervals(duration, &silence);
    if pad > 0.0 {
        speech = pad_and_merge_intervals(duration, &speech, pad);
    }
    Ok(speech)
}

/// Public wrapper so the pipeline module can probe audio duration.
pub fn ffprobe_duration_seconds_pub(path: &Path) -> Result<f64, String> {
    ffprobe_duration_seconds(path)
}

fn ffprobe_duration_seconds(path: &Path) -> Result<f64, String> {
    let ffprobe = find_in_path(&["ffprobe", "ffprobe.exe"])
        .ok_or_else(|| "ffprobe not found in PATH".to_string())?;

    let output = Command::new(ffprobe)
        .arg("-hide_banner")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(path)
        .output()
        .map_err(|e| format!("failed to spawn ffprobe: {e}"))?;

    if !output.status.success() {
        return Err(format!("ffprobe failed with status: {}", output.status));
    }

    let s = String::from_utf8_lossy(&output.stdout);
    let v = s
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("ffprobe returned invalid duration: {}", s.trim()))?;
    Ok(v)
}

fn ffmpeg_silencedetect(
    path: &Path,
    threshold_db: f64,
    min_silence: f64,
    duration_seconds: f64,
) -> Result<Vec<VadInterval>, String> {
    let ffmpeg = find_in_path(&["ffmpeg", "ffmpeg.exe"])
        .ok_or_else(|| "ffmpeg not found in PATH".to_string())?;
    let af = format!("silencedetect=n={}dB:d={}", threshold_db, min_silence);

    let output = Command::new(ffmpeg)
        .arg("-hide_banner")
        .arg("-nostdin")
        .arg("-v")
        .arg("info")
        .arg("-i")
        .arg(path)
        .arg("-vn")
        .arg("-sn")
        .arg("-t")
        .arg(format!("{:.3}", duration_seconds.max(0.0)))
        .arg("-af")
        .arg(af)
        .arg("-f")
        .arg("null")
        .arg("-")
        .output()
        .map_err(|e| format!("failed to spawn ffmpeg: {e}"))?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut intervals = Vec::<VadInterval>::new();
    let mut current_start = Option::<f64>::None;

    for line in stderr.lines() {
        if let Some(pos) = line.find("silence_start:") {
            let rest = line[pos + "silence_start:".len()..].trim();
            if let Some(value) = rest.split_whitespace().next() {
                if let Ok(v) = value.parse::<f64>() {
                    current_start = Some(v);
                }
            }
        } else if let Some(pos) = line.find("silence_end:") {
            let rest = line[pos + "silence_end:".len()..].trim();
            let end_str = rest
                .split(|c: char| c == '|' || c.is_whitespace())
                .next()
                .unwrap_or("");
            if let Ok(end) = end_str.parse::<f64>() {
                if let Some(start) = current_start.take() {
                    if end > start {
                        intervals.push(VadInterval { start, end });
                    }
                }
            }
        }
    }

    if let Some(start) = current_start.take() {
        if duration_seconds > start {
            intervals.push(VadInterval {
                start,
                end: duration_seconds,
            });
        }
    }

    intervals.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(merge_intervals(&intervals))
}

fn merge_intervals(intervals: &[VadInterval]) -> Vec<VadInterval> {
    let mut merged = Vec::<VadInterval>::new();
    for interval in intervals {
        if let Some(last) = merged.last_mut() {
            if interval.start <= last.end {
                last.end = last.end.max(interval.end);
                continue;
            }
        }
        merged.push(*interval);
    }
    merged
}

fn invert_intervals(duration: f64, silence: &[VadInterval]) -> Vec<VadInterval> {
    let mut speech = Vec::<VadInterval>::new();
    let mut cursor = 0.0f64;
    for s in silence {
        if cursor < s.start {
            speech.push(VadInterval {
                start: cursor,
                end: s.start,
            });
        }
        cursor = cursor.max(s.end);
    }
    if cursor < duration {
        speech.push(VadInterval {
            start: cursor,
            end: duration,
        });
    }
    speech
}

fn pad_and_merge_intervals(duration: f64, speech: &[VadInterval], pad: f64) -> Vec<VadInterval> {
    let mut padded = speech
        .iter()
        .map(|s| VadInterval {
            start: (s.start - pad).max(0.0),
            end: (s.end + pad).min(duration),
        })
        .collect::<Vec<_>>();
    padded.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merge_intervals(&padded)
}

#[allow(clippy::too_many_arguments)]
fn run_whisper_cpp(
    whisper_bin: &Path,
    whisper_model: &Path,
    wav_path: &Path,
    out_prefix: &Path,
    source_lang: &str,
    translate_to_english: bool,
    whisper_args: &[String],
    timeout: Option<Duration>,
) -> Result<(), String> {
    // whisper.cpp expects -of as an output *prefix* (without extension).
    // With -osrt it will write "<prefix>.srt".
    let mut cmd = Command::new(whisper_bin);
    cmd.arg("-m")
        .arg(whisper_model)
        .arg("-f")
        .arg(wav_path)
        .arg("-l")
        .arg(source_lang)
        .arg("-osrt")
        .arg("-of")
        .arg(out_prefix);

    if translate_to_english {
        cmd.arg("-tr");
    }

    // Default to max CPU parallelism unless caller overrides.
    if !has_thread_override(whisper_args) {
        if let Ok(threads) = std::thread::available_parallelism() {
            cmd.arg("-t").arg(threads.get().to_string());
        }
    }
    for arg in whisper_args {
        cmd.arg(arg);
    }

    run_command_with_timeout(cmd, "whisper", timeout)
}

fn has_thread_override(args: &[String]) -> bool {
    args.iter()
        .any(|arg| arg == "-t" || arg == "--threads" || arg == "--thread")
}

fn run_python_whisper(
    wav_path: &Path,
    out_dir: &Path,
    config: &TranscribeConfig,
    model_name: &str,
    model_dir: &Path,
    python_bin: &str,
    timeout: Option<Duration>,
) -> Result<PathBuf, String> {
    let translate_to_english = config.target_lang.eq_ignore_ascii_case("en");

    let requested_device = py_device_from_args(&config.whisper_args);
    let requested_cuda = requested_device
        .as_deref()
        .map(is_cuda_device)
        .unwrap_or(false);
    let cuda_available = if config.gpu || config.require_gpu || requested_cuda {
        python_cuda_available(python_bin)?
    } else {
        false
    };
    if config.require_gpu {
        match requested_device.as_deref() {
            Some(device) if !is_cuda_device(device) => {
                return Err(format!(
                    "--require-gpu was set but whisper args request --device {device}. Remove the override or set --device cuda."
                ));
            }
            _ => {}
        }
        if !cuda_available {
            return Err(
                "--require-gpu was set but CUDA is unavailable (torch.cuda.is_available() == False)."
                    .to_string(),
            );
        }
    } else if let Some(device) = requested_device.as_deref() {
        if is_cuda_device(device) && !cuda_available {
            eprintln!(
                "warning: python whisper requested --device {device} but CUDA is unavailable; this will likely fail."
            );
        }
    }

    let mut cmd = Command::new(python_bin);
    cmd.arg("-m")
        .arg("whisper")
        .arg(wav_path)
        .arg("--model")
        .arg(model_name)
        .arg("--model_dir")
        .arg(model_dir)
        .arg("--output_dir")
        .arg(out_dir)
        .arg("--output_format")
        .arg(py_output_format_for_profile(config))
        .arg("--language")
        .arg(&config.source_lang)
        .arg("--task")
        .arg(if translate_to_english {
            "translate"
        } else {
            "transcribe"
        })
        .arg("--verbose")
        .arg("False");

    // Device/fp16 defaults (override-able via --whisper-arg).
    let mut device_is_cuda = false;
    if let Some(device) = requested_device.as_deref() {
        device_is_cuda = is_cuda_device(device);
    } else if config.gpu {
        if cuda_available {
            device_is_cuda = true;
            cmd.arg("--device").arg("cuda");
        } else {
            eprintln!("warning: --gpu was requested but CUDA is unavailable; falling back to CPU.");
            cmd.arg("--device").arg("cpu");
        }
    } else {
        cmd.arg("--device").arg("cpu");
    }

    if !has_py_fp16_override(&config.whisper_args) {
        if device_is_cuda {
            cmd.arg("--fp16").arg("True");
        } else {
            cmd.arg("--fp16").arg("False");
        }
    }

    // VAD-based silence skipping (python backend only).
    let auto_clip_enabled = config.vad
        && config.quality_profile != QualityProfile::Strict
        && !has_py_clip_override(&config.whisper_args);
    if auto_clip_enabled {
        let speech = detect_speech_intervals_from_wav(
            wav_path,
            config.vad_threshold_db,
            config.vad_min_silence,
            config.vad_pad,
            None,
        )?;
        if !speech.is_empty() {
            let clip = speech
                .iter()
                .flat_map(|s| [format!("{:.3}", s.start), format!("{:.3}", s.end)])
                .collect::<Vec<_>>()
                .join(",");
            cmd.arg("--clip_timestamps").arg(clip);
        }
    }

    if !has_py_threads_override(&config.whisper_args) {
        if let Ok(threads) = std::thread::available_parallelism() {
            cmd.arg("--threads").arg(threads.get().to_string());
        }
    }
    apply_py_profile_defaults(&mut cmd, config);
    for arg in &config.whisper_args {
        cmd.arg(arg);
    }

    run_command_with_timeout(cmd, "python whisper", timeout)?;

    // python -m whisper names outputs after the input file stem.
    let stem = wav_path
        .file_stem()
        .and_then(OsStr::to_str)
        .unwrap_or("audio");
    let srt_path = out_dir.join(format!("{stem}.srt"));
    if !srt_path.is_file() {
        return Err(format!(
            "python whisper completed but SRT was not produced: {}",
            srt_path.display()
        ));
    }

    Ok(srt_path)
}

fn py_output_format_for_profile(config: &TranscribeConfig) -> &'static str {
    // Strict mode keeps JSON alongside SRT so the pipeline can consume
    // confidence signals (avg_logprob/no_speech/compression) for targeted rescue.
    match config.quality_profile {
        QualityProfile::Strict => "all",
        _ => "srt",
    }
}

fn run_command_with_timeout(
    mut cmd: Command,
    label: &str,
    timeout: Option<Duration>,
) -> Result<(), String> {
    if let Some(limit) = timeout {
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn {label}: {e}"))?;
        let start = Instant::now();
        loop {
            if let Some(status) = child
                .try_wait()
                .map_err(|e| format!("failed to wait for {label}: {e}"))?
            {
                if status.success() {
                    return Ok(());
                }
                return Err(format!("{label} failed with status: {status}"));
            }
            if start.elapsed() >= limit {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "{label} timed out after {:.1}s",
                    limit.as_secs_f64()
                ));
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    let output = cmd
        .output()
        .map_err(|e| format!("failed to run {label}: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            return Err(format!("{label} failed with status: {}", output.status));
        }
        return Err(format!(
            "{label} failed with status {}: {stderr}",
            output.status
        ));
    }
    Ok(())
}

fn python_cuda_available(python_bin: &str) -> Result<bool, String> {
    let output = Command::new(python_bin)
        .arg("-c")
        .arg(
            r#"
import sys, warnings
warnings.filterwarnings('ignore')
try:
    import torch
    ok = bool(torch.cuda.is_available())
except Exception as e:
    print(f'CUDA check failed: {e}', file=sys.stderr)
    ok = False
sys.exit(0 if ok else 1)
"#,
        )
        .output()
        .map_err(|e| format!("failed to spawn {python_bin} for CUDA check: {e}"))?;
    if !output.status.success() {
        let err_str = String::from_utf8_lossy(&output.stderr);
        if !err_str.trim().is_empty() {
            eprintln!("CUDA check stderr: {}", err_str.trim());
        }
    }
    Ok(output.status.success())
}

fn is_cuda_device(device: &str) -> bool {
    let d = device.trim().to_lowercase();
    d == "cuda" || d.starts_with("cuda:")
}

fn py_device_from_args(args: &[String]) -> Option<String> {
    let mut i = 0usize;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--device" {
            if let Some(v) = args.get(i + 1) {
                return Some(v.to_string());
            }
            return Some(String::new());
        }
        if let Some(v) = arg.strip_prefix("--device=") {
            return Some(v.to_string());
        }
        i += 1;
    }
    None
}

fn has_py_fp16_override(args: &[String]) -> bool {
    args.iter()
        .any(|arg| arg == "--fp16" || arg.starts_with("--fp16="))
}

fn has_py_clip_override(args: &[String]) -> bool {
    args.iter().any(|arg| {
        arg == "--clip_timestamps"
            || arg.starts_with("--clip_timestamps=")
            || arg == "--clip-timestamps"
            || arg.starts_with("--clip-timestamps=")
    })
}

fn has_py_threads_override(args: &[String]) -> bool {
    args.iter().any(|arg| {
        arg == "--threads"
            || arg.starts_with("--threads=")
            || arg == "--thread"
            || arg.starts_with("--thread=")
    })
}

fn apply_py_profile_defaults(cmd: &mut Command, config: &TranscribeConfig) {
    let args = &config.whisper_args;
    match config.quality_profile {
        QualityProfile::Fast => {
            if !has_py_beam_override(args) {
                cmd.arg("--beam_size").arg("3");
            }
            if !has_py_best_of_override(args) {
                cmd.arg("--best_of").arg("3");
            }
            if !has_py_temperature_override(args) {
                cmd.arg("--temperature").arg("0");
            }
        }
        QualityProfile::Balanced => {
            if !has_py_beam_override(args) {
                cmd.arg("--beam_size").arg("5");
            }
            if !has_py_best_of_override(args) {
                cmd.arg("--best_of").arg("5");
            }
            if !has_py_temperature_override(args) {
                cmd.arg("--temperature").arg("0");
            }
            if !has_py_condition_on_previous_text_override(args) {
                cmd.arg("--condition_on_previous_text").arg("True");
            }
        }
        QualityProfile::Strict => {
            if !has_py_beam_override(args) {
                cmd.arg("--beam_size").arg("8");
            }
            if !has_py_best_of_override(args) {
                cmd.arg("--best_of").arg("8");
            }
            if !has_py_patience_override(args) {
                cmd.arg("--patience").arg("1.2");
            }
            if !has_py_temperature_override(args) {
                cmd.arg("--temperature").arg("0");
            }
            if !has_py_condition_on_previous_text_override(args) {
                cmd.arg("--condition_on_previous_text").arg("True");
            }
            if !has_py_compression_ratio_threshold_override(args) {
                cmd.arg("--compression_ratio_threshold").arg("2.2");
            }
            if !has_py_logprob_threshold_override(args) {
                cmd.arg("--logprob_threshold").arg("-1.0");
            }
            if !has_py_no_speech_threshold_override(args) {
                cmd.arg("--no_speech_threshold").arg("0.45");
            }
            if !has_py_word_timestamps_override(args) {
                cmd.arg("--word_timestamps").arg("True");
            }
        }
    }
}

fn has_py_named_option(args: &[String], names: &[&str]) -> bool {
    args.iter().any(|arg| {
        names
            .iter()
            .any(|name| arg == *name || arg.starts_with(&format!("{name}=")))
    })
}

fn has_py_beam_override(args: &[String]) -> bool {
    has_py_named_option(args, &["--beam_size", "--beam-size"])
}

fn has_py_best_of_override(args: &[String]) -> bool {
    has_py_named_option(args, &["--best_of", "--best-of"])
}

fn has_py_temperature_override(args: &[String]) -> bool {
    has_py_named_option(args, &["--temperature"])
}

fn has_py_patience_override(args: &[String]) -> bool {
    has_py_named_option(args, &["--patience"])
}

fn has_py_condition_on_previous_text_override(args: &[String]) -> bool {
    has_py_named_option(
        args,
        &[
            "--condition_on_previous_text",
            "--condition-on-previous-text",
        ],
    )
}

fn has_py_compression_ratio_threshold_override(args: &[String]) -> bool {
    has_py_named_option(
        args,
        &[
            "--compression_ratio_threshold",
            "--compression-ratio-threshold",
        ],
    )
}

fn has_py_logprob_threshold_override(args: &[String]) -> bool {
    has_py_named_option(args, &["--logprob_threshold", "--logprob-threshold"])
}

fn has_py_no_speech_threshold_override(args: &[String]) -> bool {
    has_py_named_option(args, &["--no_speech_threshold", "--no-speech-threshold"])
}

fn has_py_word_timestamps_override(args: &[String]) -> bool {
    has_py_named_option(args, &["--word_timestamps", "--word-timestamps"])
}

fn sanitize_whisper_args_for_cpu_retry(args: &[String]) -> Vec<String> {
    let mut cleaned = Vec::<String>::new();
    let mut index = 0usize;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--device" || arg == "--fp16" {
            index += 2;
            continue;
        }
        if arg.starts_with("--device=") || arg.starts_with("--fp16=") {
            index += 1;
            continue;
        }
        cleaned.push(arg.clone());
        index += 1;
    }
    cleaned
}

fn is_cuda_oom_transcribe_error(error: &str) -> bool {
    let lowered = error.to_ascii_lowercase();
    lowered.contains("cuda out of memory")
        || (lowered.contains("cuda") && lowered.contains("out of memory"))
        || lowered.contains("torch.outofmemoryerror")
}

fn find_in_path(candidates: &[&str]) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        for name in candidates {
            let path = dir.join(name);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{TranscribeConfig, Transcriber};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn transcriber_disabled_is_none() {
        let transcriber = Transcriber::new(TranscribeConfig {
            enabled: false,
            whisper_bin: None,
            whisper_model: None,
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            whisper_args: Vec::new(),
            vad: false,
            vad_threshold_db: -35.0,
            vad_min_silence: 0.35,
            vad_pad: 0.20,
            gpu: false,
            require_gpu: false,
            quality_profile: super::QualityProfile::Balanced,
        })
        .expect("constructor should succeed");
        assert!(transcriber.is_none());
    }

    #[test]
    fn transcriber_requires_model_when_enabled_if_whisper_cpp_is_forced() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let fake_bin = std::env::temp_dir().join(format!("sub_zero_fake_whisper_bin_{stamp}"));
        fs::write(&fake_bin, "").expect("temp file should be writable");

        let error = Transcriber::new(TranscribeConfig {
            enabled: true,
            whisper_bin: Some(fake_bin),
            whisper_model: None,
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            whisper_args: Vec::new(),
            vad: false,
            vad_threshold_db: -35.0,
            vad_min_silence: 0.35,
            vad_pad: 0.20,
            gpu: false,
            require_gpu: false,
            quality_profile: super::QualityProfile::Balanced,
        })
        .expect_err("should fail without model");
        assert!(error.contains("missing --whisper-model"));
    }
}
