// Neural Machine Translation Engine — wraps CTranslate2 + NLLB-200 via Python.
//
// Sends batched context-windowed cues to a Python subprocess, receives
// translations back via JSON-lines protocol.  Falls back to the old phrase
// table when the neural backend is unavailable.

use crate::engine::context::{
    build_context_windows, build_context_windows_with_tags, ContextualCue,
};
use crate::engine::srt::SubtitleCue;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const PYTHON_CANDIDATES: &[&str] = &["python3", "python"];

#[derive(Debug, Clone)]
pub struct NeuralMTConfig {
    /// Path to the translate_batch.py helper script.
    pub script_path: PathBuf,
    /// NLLB-200 model size identifier (e.g. "nllb-200-distilled-600M").
    pub model_name: String,
    /// Optional directory where the model is cached.
    pub model_dir: Option<PathBuf>,
    /// Source language code (BCP-47 or NLLB format, e.g. "jpn_Jpan").
    pub source_lang: String,
    /// Target language code (e.g. "eng_Latn").
    pub target_lang: String,
    /// Use GPU if available.
    pub gpu: bool,
    /// Context window radius.
    pub context_radius: usize,
    /// Batch size for MT inference.
    pub batch_size: usize,
    /// Maximum token budget per translation batch in CTranslate2.
    pub max_batch_tokens: usize,
    /// Number of CUDA OOM retries before giving up.
    pub oom_retries: usize,
    /// Allow automatic CPU fallback when CUDA OOM persists.
    pub allow_cpu_fallback_on_oom: bool,
    /// Beam size used by CTranslate2.
    pub beam_size: usize,
    /// Repetition penalty used by CTranslate2.
    pub repetition_penalty: f32,
    /// Block repeated n-grams up to this size (0 disables).
    pub no_repeat_ngram_size: usize,
    /// Whether to prepend previous cue text as a context hint.
    pub prepend_prev_context: bool,
    /// Reuse a persistent Python subprocess for MT across batches/inputs.
    pub use_daemon: bool,
}

impl Default for NeuralMTConfig {
    fn default() -> Self {
        Self {
            script_path: PathBuf::from("scripts/translate_batch.py"),
            model_name: "nllb-200-distilled-600M".to_string(),
            model_dir: None,
            source_lang: "jpn_Jpan".to_string(),
            target_lang: "eng_Latn".to_string(),
            gpu: false,
            context_radius: 3,
            batch_size: 32,
            max_batch_tokens: 8192,
            oom_retries: 2,
            allow_cpu_fallback_on_oom: true,
            beam_size: 4,
            repetition_penalty: 1.1,
            no_repeat_ngram_size: 3,
            // F.2 §6.2 / win-condition W1 (discourse coherence):
            // prepend the prior cues' source text into NLLB's input
            // so anaphora and named-entity coreference resolve across
            // cue boundaries. The Python side (translate_batch.py)
            // strips the corresponding leading segments from the
            // decoded output. Default ON since this strictly improves
            // the Lagrangian objective from F.2 (E5).
            prepend_prev_context: true,
            use_daemon: false,
        }
    }
}

#[derive(Debug)]
pub enum NeuralMtError {
    CudaProbe { source: String },
    Translate { source: String },
}

pub type NeuralMtResult<T> = Result<T, NeuralMtError>;

impl std::fmt::Display for NeuralMtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CudaProbe { source } => {
                write!(f, "failed to probe neural MT CUDA devices: {source}")
            }
            Self::Translate { source } => write!(f, "neural MT failed: {source}"),
        }
    }
}

impl std::error::Error for NeuralMtError {}

/// Input record sent to the Python script.
#[derive(Debug, Serialize)]
struct MTRequest {
    index: usize,
    text: String,
    prev_context: Vec<String>,
    next_context: Vec<String>,
    context_tags: Vec<String>,
}

/// Output record received from the Python script.
#[derive(Debug, Deserialize)]
struct MTResponse {
    index: usize,
    translation: String,
}

/// Check whether the neural MT backend is available.
pub fn neural_mt_available(script_path: &Path) -> bool {
    if !script_path.is_file() {
        return false;
    }
    // Check that ctranslate2 is importable.
    let python = find_python();
    let Some(python) = python else {
        return false;
    };
    let status = Command::new(&python)
        .arg("-c")
        .arg("import ctranslate2, sentencepiece")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    matches!(status, Ok(s) if s.success())
}

/// Probe how many CUDA devices CTranslate2 can use from the active Python env.
pub fn neural_mt_cuda_device_count() -> NeuralMtResult<usize> {
    neural_mt_cuda_device_count_inner().map_err(|source| NeuralMtError::CudaProbe { source })
}

fn neural_mt_cuda_device_count_inner() -> Result<usize, String> {
    let python = find_python().ok_or_else(|| "python not found for neural MT".to_string())?;
    let output = Command::new(&python)
        .arg("-c")
        .arg(
            r#"import ctranslate2, sys
try:
    print(ctranslate2.get_cuda_device_count())
except Exception as e:
    print(e, file=sys.stderr)
    raise SystemExit(2)
"#,
        )
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("failed to probe ctranslate2 CUDA devices: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ctranslate2 CUDA probe failed with status {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().parse::<usize>().map_err(|e| {
        format!(
            "invalid ctranslate2 CUDA probe output {:?}: {e}",
            stdout.trim()
        )
    })
}

/// Translate a batch of contextual cues using the neural MT pipeline.
pub fn translate_batch(
    cues: &[ContextualCue],
    config: &NeuralMTConfig,
) -> NeuralMtResult<Vec<String>> {
    translate_batch_inner(cues, config).map_err(|source| NeuralMtError::Translate { source })
}

fn translate_batch_inner(
    cues: &[ContextualCue],
    config: &NeuralMTConfig,
) -> Result<Vec<String>, String> {
    let python = find_python().ok_or_else(|| "python not found for neural MT".to_string())?;

    if !config.script_path.is_file() {
        return Err(format!(
            "translation script not found: {}",
            config.script_path.display()
        ));
    }

    if config.use_daemon {
        return translate_batch_via_daemon(&python, cues, config);
    }

    // Build requests.
    let requests: Vec<MTRequest> = cues
        .iter()
        .map(|c| MTRequest {
            index: c.index,
            text: c.current_line.clone(),
            prev_context: c.prev_lines.clone(),
            next_context: c.next_lines.clone(),
            context_tags: c.context_tags.clone(),
        })
        .collect();

    let input_json = serde_json::to_string(&requests)
        .map_err(|e| format!("failed to serialize MT batch: {e}"))?;

    let mut cmd = Command::new(&python);
    cmd.arg(&config.script_path)
        .arg("--model")
        .arg(&config.model_name)
        .arg("--source-lang")
        .arg(&config.source_lang)
        .arg("--target-lang")
        .arg(&config.target_lang)
        .arg("--batch-size")
        .arg(config.batch_size.to_string())
        .arg("--max-batch-tokens")
        .arg(config.max_batch_tokens.to_string())
        .arg("--beam-size")
        .arg(config.beam_size.to_string())
        .arg("--repetition-penalty")
        .arg(config.repetition_penalty.to_string())
        .arg("--no-repeat-ngram-size")
        .arg(config.no_repeat_ngram_size.to_string())
        .arg("--oom-retries")
        .arg(config.oom_retries.to_string());

    if config.prepend_prev_context {
        cmd.arg("--prepend-prev-context");
    }
    if config.allow_cpu_fallback_on_oom {
        cmd.arg("--allow-cpu-fallback");
    }

    if let Some(dir) = &config.model_dir {
        cmd.arg("--model-dir").arg(dir);
    }
    if config.gpu {
        cmd.arg("--device").arg("cuda");
    } else {
        cmd.arg("--device").arg("cpu");
    }

    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn neural MT: {e}"))?;

    // Write the entire batch to stdin.
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(input_json.as_bytes())
            .map_err(|e| format!("failed to write to MT stdin: {e}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to wait for MT process: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if is_cuda_oom_message(&stderr) {
            eprintln!(
                "warning: neural MT reported CUDA OOM after retries; consider lower MT batch settings or CPU fallback."
            );
        }
        return Err(format!("neural MT failed: {stderr}"));
    }

    // Forward explicit backend diagnostics emitted by the Python helper.
    let stderr = String::from_utf8_lossy(&output.stderr);
    for line in stderr.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("sub-zero:") {
            eprintln!("{trimmed}");
        }
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let responses: Vec<MTResponse> =
        serde_json::from_str(&stdout).map_err(|e| format!("failed to parse MT output: {e}"))?;

    // Map responses back by index in O(n).
    let index_to_pos: HashMap<usize, usize> = cues
        .iter()
        .enumerate()
        .map(|(pos, cue)| (cue.index, pos))
        .collect();
    let mut translations = vec![String::new(); cues.len()];
    for resp in responses {
        if let Some(pos) = index_to_pos.get(&resp.index).copied() {
            translations[pos] = resp.translation;
        }
    }

    Ok(translations)
}

thread_local! {
    static MT_DAEMON: RefCell<Option<MtDaemon>> = const { RefCell::new(None) };
}

#[derive(Debug)]
struct MtDaemon {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::io::BufReader<std::process::ChildStdout>,
}

#[derive(Debug, Deserialize)]
struct MtDaemonResponse {
    ok: bool,
    error: Option<String>,
    responses: Option<Vec<MTResponse>>,
}

impl MtDaemon {
    fn spawn(python: &str, script_path: &Path) -> Result<Self, String> {
        let mut cmd = Command::new(python);
        cmd.arg("-u")
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn MT daemon: {e}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "MT daemon stdin pipe unavailable".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "MT daemon stdout pipe unavailable".to_string())?;

        Ok(Self {
            child,
            stdin,
            stdout: std::io::BufReader::new(stdout),
        })
    }

    fn translate(
        &mut self,
        config: &NeuralMTConfig,
        requests: &[MTRequest],
    ) -> Result<Vec<MTResponse>, String> {
        let payload = serde_json::json!({
            "cmd": "translate",
            "config": {
                "model": config.model_name,
                "model_dir": config.model_dir.as_ref().map(|p| p.display().to_string()),
                "source_lang": config.source_lang,
                "target_lang": config.target_lang,
                "device": if config.gpu { "cuda" } else { "cpu" },
                "batch_size": config.batch_size,
                "max_batch_tokens": config.max_batch_tokens,
                "beam_size": config.beam_size,
                "repetition_penalty": config.repetition_penalty,
                "no_repeat_ngram_size": config.no_repeat_ngram_size,
                "oom_retries": config.oom_retries,
                "allow_cpu_fallback": config.allow_cpu_fallback_on_oom,
                "prepend_prev_context": config.prepend_prev_context,
            },
            "requests": requests,
        });

        let encoded = serde_json::to_vec(&payload)
            .map_err(|e| format!("failed to serialize MT daemon request: {e}"))?;
        write_frame(&mut self.stdin, &encoded)
            .map_err(|e| format!("failed to write MT daemon request: {e}"))?;

        let response_bytes = read_frame(&mut self.stdout)
            .map_err(|e| format!("failed to read MT daemon response: {e}"))?;
        let response: MtDaemonResponse = serde_json::from_slice(&response_bytes)
            .map_err(|e| format!("failed to parse MT daemon response: {e}"))?;
        if !response.ok {
            return Err(response
                .error
                .unwrap_or_else(|| "MT daemon reported failure without error text".to_string()));
        }

        response
            .responses
            .ok_or_else(|| "MT daemon returned ok=true but missing responses".to_string())
    }
}

impl Drop for MtDaemon {
    fn drop(&mut self) {
        if let Ok(encoded) = serde_json::to_vec(&serde_json::json!({ "cmd": "shutdown" })) {
            let _ = write_frame(&mut self.stdin, &encoded);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn write_frame<W: Write>(writer: &mut W, payload: &[u8]) -> Result<(), std::io::Error> {
    let len: u32 = payload
        .len()
        .try_into()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "frame too large"))?;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(payload)?;
    writer.flush()
}

fn read_frame<R: Read>(reader: &mut R) -> Result<Vec<u8>, std::io::Error> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn translate_batch_via_daemon(
    python: &str,
    cues: &[ContextualCue],
    config: &NeuralMTConfig,
) -> Result<Vec<String>, String> {
    // Build requests.
    let requests: Vec<MTRequest> = cues
        .iter()
        .map(|c| MTRequest {
            index: c.index,
            text: c.current_line.clone(),
            prev_context: c.prev_lines.clone(),
            next_context: c.next_lines.clone(),
            context_tags: c.context_tags.clone(),
        })
        .collect();

    let mut responses = MT_DAEMON.with(|cell| -> Result<Vec<MTResponse>, String> {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(MtDaemon::spawn(python, &config.script_path)?);
        }
        let Some(daemon) = slot.as_mut() else {
            return Err("MT daemon missing after spawn".to_string());
        };

        match daemon.translate(config, &requests) {
            Ok(responses) => Ok(responses),
            Err(error) => {
                // One restart attempt for robustness if the daemon crashed or got wedged.
                *slot = None;
                *slot = Some(MtDaemon::spawn(python, &config.script_path)?);
                slot.as_mut()
                    .ok_or_else(|| "MT daemon missing after restart".to_string())?
                    .translate(config, &requests)
                    .map_err(|e| format!("MT daemon retry failed: {e} (original={error})"))
            }
        }
    })?;

    // Map responses back by index in O(n).
    let index_to_pos: HashMap<usize, usize> = cues
        .iter()
        .enumerate()
        .map(|(pos, cue)| (cue.index, pos))
        .collect();
    let mut translations = vec![String::new(); cues.len()];
    for resp in responses.drain(..) {
        if let Some(pos) = index_to_pos.get(&resp.index).copied() {
            translations[pos] = resp.translation;
        }
    }

    Ok(translations)
}

pub(crate) fn is_cuda_oom_message(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("cuda") && lowered.contains("out of memory")
}

/// Apply neural MT to a full set of subtitle cues, replacing their text
/// with translated versions.  Falls back to passthrough on failure.
pub fn translate_cues_neural(
    cues: &[SubtitleCue],
    config: &NeuralMTConfig,
) -> NeuralMtResult<Vec<SubtitleCue>> {
    let context_windows = build_context_windows(cues, config.context_radius);
    translate_cues_neural_from_windows(cues, &context_windows, config)
        .map_err(|source| NeuralMtError::Translate { source })
}

/// Same as `translate_cues_neural`, but with optional per-cue tags to inform
/// adaptive decode policy in the Python backend.
pub fn translate_cues_neural_with_tags(
    cues: &[SubtitleCue],
    cue_tags: &[Vec<String>],
    config: &NeuralMTConfig,
) -> NeuralMtResult<Vec<SubtitleCue>> {
    let context_windows = build_context_windows_with_tags(cues, config.context_radius, cue_tags);
    translate_cues_neural_from_windows(cues, &context_windows, config)
        .map_err(|source| NeuralMtError::Translate { source })
}

fn translate_cues_neural_from_windows(
    cues: &[SubtitleCue],
    context_windows: &[ContextualCue],
    config: &NeuralMTConfig,
) -> Result<Vec<SubtitleCue>, String> {
    // Run a single subprocess for the full cue set. The Python backend still
    // respects `--batch-size` internally for model inference, but this avoids
    // repeated model load/unload overhead per Rust-side batch.
    let translated_texts =
        translate_batch(context_windows, config).map_err(|error| error.to_string())?;
    if translated_texts.len() != cues.len() {
        return Err(format!(
            "neural MT returned mismatched item count: expected {}, got {}",
            cues.len(),
            translated_texts.len()
        ));
    }

    let result: Vec<SubtitleCue> = cues
        .iter()
        .zip(translated_texts.iter())
        .map(|(cue, translation)| SubtitleCue {
            index: cue.index,
            timing: cue.timing.clone(),
            text: if translation.is_empty() {
                cue.text.clone()
            } else {
                translation.clone()
            },
        })
        .collect();

    Ok(result)
}

/// Map common language codes to NLLB format.
pub fn to_nllb_lang(code: &str) -> String {
    match code.to_lowercase().as_str() {
        "ja" | "jpn" => "jpn_Jpan".to_string(),
        "en" | "eng" => "eng_Latn".to_string(),
        "zh" | "zho" | "cmn" => "zho_Hans".to_string(),
        "ko" | "kor" => "kor_Hang".to_string(),
        "es" | "spa" => "spa_Latn".to_string(),
        "fr" | "fra" => "fra_Latn".to_string(),
        "de" | "deu" => "deu_Latn".to_string(),
        "pt" | "por" => "por_Latn".to_string(),
        "ru" | "rus" => "rus_Cyrl".to_string(),
        "ar" | "ara" => "arb_Arab".to_string(),
        "hi" | "hin" => "hin_Deva".to_string(),
        "it" | "ita" => "ita_Latn".to_string(),
        "th" | "tha" => "tha_Thai".to_string(),
        "vi" | "vie" => "vie_Latn".to_string(),
        "tr" | "tur" => "tur_Latn".to_string(),
        "pl" | "pol" => "pol_Latn".to_string(),
        "nl" | "nld" => "nld_Latn".to_string(),
        "sv" | "swe" => "swe_Latn".to_string(),
        _ => code.to_string(),
    }
}

fn find_python() -> Option<String> {
    // Check for a repo-local venv first (native to the current OS only).
    if cfg!(windows) {
        let venv_win = PathBuf::from(".venv").join("Scripts").join("python.exe");
        if venv_win.is_file() {
            return Some(venv_win.to_string_lossy().to_string());
        }
    } else {
        for name in &["python3", "python"] {
            let venv_unix = PathBuf::from(".venv").join("bin").join(name);
            if venv_unix.is_file() {
                return Some(venv_unix.to_string_lossy().to_string());
            }
        }
    }

    for candidate in PYTHON_CANDIDATES {
        let status = Command::new(candidate)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if matches!(status, Ok(s) if s.success()) {
            return Some(candidate.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nllb_lang_mapping() {
        assert_eq!(to_nllb_lang("ja"), "jpn_Jpan");
        assert_eq!(to_nllb_lang("en"), "eng_Latn");
        assert_eq!(to_nllb_lang("fr"), "fra_Latn");
        assert_eq!(to_nllb_lang("unknown"), "unknown");
    }

    #[test]
    fn detect_cuda_oom_message() {
        let msg = "RuntimeError: CUDA failed with error out of memory";
        assert!(is_cuda_oom_message(msg));
        assert!(!is_cuda_oom_message(
            "RuntimeError: some other backend error"
        ));
    }
}
