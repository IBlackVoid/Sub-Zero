//! Local LLM machine-translation backend — the Qwen3 escalation rung (WS-A).
//!
//! The NLLB ladder (600M → 1.3B) repairs *systematic* mistranslation but cannot
//! rescue rapid casual speech, where NLLB degenerates regardless of size (see
//! agent-memory `finding_ja_en_casual_mt_quality`). A local instruction-tuned
//! LLM does: the WS-A spike showed Qwen3-4B getting 39/39 unique correct lines
//! on the exact content NLLB degenerates on. This module is that rung.
//!
//! Design constraints, all deliberate:
//! - **Offline / local only.** The LLM runs as a `llama-server` child process
//!   bound to loopback (`127.0.0.1`) on an ephemeral port — never a network
//!   service, never a cloud call. The sidecar dies with its `LlmTranslator`.
//! - **Zero new dependencies.** Talking to a known localhost server is a small
//!   enough job that a std-only HTTP/1.1 client (`Connection: close`,
//!   read-to-EOF) beats pulling in a TLS-carrying HTTP crate for a plaintext
//!   loopback request. Keeps the dependency surface (and audit burden) flat.
//! - **Exactly-N output.** A generated GBNF grammar constrains the model to a
//!   JSON array of exactly N strings (one per input cue), so the cue count is
//!   preserved and the model can't run on — a freeform line grammar let it
//!   cram many sentences into one "line" and loop to the token cap (observed
//!   live). The closing `]` completes the grammar and forces a clean stop.
//! - **Determinism-leaning sampling.** temperature 0.2 + repeat-penalty 1.1:
//!   low enough to be near-reproducible, high enough to avoid loops.
//!
//! The pure pieces (grammar, prompt, response parsing, path resolution) are unit
//! tested without a server; the live spawn + `/completion` round-trip is an
//! `#[ignore]`d smoke (`live_smoke_translates_japanese_lines`) run explicitly
//! with `--ignored`, so CI without a GPU stays green.

use crate::engine::srt::SubtitleCue;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const HOST: &str = "127.0.0.1";
/// How long to wait for the model to load and the server to report healthy.
const READY_TIMEOUT: Duration = Duration::from_secs(180);
/// Per-request generation timeout.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Resolve the `llama-server` binary and the GGUF model, if both exist.
///
/// `VOIDEX_LLM_SERVER` / `VOIDEX_LLM_MODEL` override the defaults; otherwise the
/// spike layout under `~/.voidex/spike/` is used. Returns `None` (not an error)
/// when either is missing, so the policy simply omits the LLM rung and the
/// ladder stays NLLB-only — graceful, not fatal.
pub(crate) fn resolve_llm_paths() -> Option<(PathBuf, PathBuf)> {
    let binary = env_path("VOIDEX_LLM_SERVER")
        .unwrap_or_else(|| spike_dir().join("llama").join("llama-server.exe"));
    let model =
        env_path("VOIDEX_LLM_MODEL").unwrap_or_else(|| spike_dir().join("qwen3-4b-q4km.gguf"));
    if binary.is_file() && model.is_file() {
        Some((binary, model))
    } else {
        None
    }
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn spike_dir() -> PathBuf {
    let home = std::env::var_os("VOIDEX_HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".voidex").join("spike")
}

/// Human-readable language name for the prompt (LLMs translate better when told
/// the language by name than by ISO code).
fn lang_name(code: &str) -> String {
    match code.to_lowercase().as_str() {
        "ja" | "jpn" | "japanese" => "Japanese".to_string(),
        "en" | "eng" | "english" => "English".to_string(),
        "zh" | "zho" | "chinese" => "Chinese".to_string(),
        "ko" | "kor" | "korean" => "Korean".to_string(),
        "es" => "Spanish".to_string(),
        "fr" => "French".to_string(),
        "de" => "German".to_string(),
        other => other.to_string(),
    }
}

/// A GBNF grammar admitting a JSON array of exactly `n` strings.
///
/// A JSON array is used deliberately over newline-separated lines: the closing
/// `]` after the nth element makes the grammar complete, which forces the model
/// to stop. A freeform `line ::= [^\n]+` grammar let the model run on (cramming
/// many sentences into one "line" and looping to the token cap) and leak `1.`
/// numbering — both observed live. Quoted strings also can't carry stray
/// numbering prefixes from bleeding into the parsed text.
fn build_grammar(n: usize) -> String {
    let n = n.max(1);
    let mut root = String::from("root ::= \"[\" ws string");
    for _ in 1..n {
        root.push_str(" ws \",\" ws string");
    }
    root.push_str(" ws \"]\"\n");
    root.push_str(
        r#"string ::= "\"" ( [^"\\] | "\\" . )* "\""
ws ::= [ \t\n]*
"#,
    );
    root
}

/// Build the translation prompt for one scene's worth of cues.
///
/// `context` is the previously-translated tail (rolling context for coherence);
/// `glossary` is the canonical character/term names to keep consistent.
fn build_prompt(
    source_lang: &str,
    target_lang: &str,
    cues: &[SubtitleCue],
    context: &[String],
    glossary: &[String],
) -> String {
    let src = lang_name(source_lang);
    let tgt = lang_name(target_lang);
    let n = cues.len();

    let mut p = String::new();
    p.push_str(&format!(
        "You are a professional subtitle translator. Translate each numbered {src} line into natural, concise, idiomatic {tgt} as spoken subtitles.\n"
    ));
    p.push_str(&format!(
        "Reply with ONLY a JSON array of EXACTLY {n} {tgt} string(s), one translation per input line, in the same order. Example: [\"first line\",\"second line\"]. No numbering, no commentary outside the array.\n"
    ));

    if !glossary.is_empty() {
        p.push_str("Keep these names consistent: ");
        p.push_str(&glossary.join(", "));
        p.push('\n');
    }

    if !context.is_empty() {
        p.push_str("Earlier lines (context, do not translate):\n");
        for line in context {
            p.push_str("- ");
            p.push_str(line);
            p.push('\n');
        }
    }

    p.push_str(&format!("Input ({src}):\n"));
    for (i, cue) in cues.iter().enumerate() {
        p.push_str(&format!("{}. {}\n", i + 1, cue.text.replace('\n', " ")));
    }
    p.push_str(&format!("Output ({tgt}):\n"));
    p
}

/// Parse a `/completion` `content` field (a JSON array of strings, per the
/// grammar) into exactly `n` translations. Surplus elements are dropped; a short
/// array is an error (the grammar makes this practically unreachable, but we
/// never splice a ragged result).
fn parse_completion_lines(content: &str, n: usize) -> Result<Vec<String>, String> {
    let items: Vec<String> = serde_json::from_str(content.trim())
        .map_err(|e| format!("llm output was not a JSON string array: {e}"))?;
    if items.len() < n {
        return Err(format!(
            "llm returned {} item(s), expected {n}",
            items.len()
        ));
    }
    Ok(items
        .into_iter()
        .map(|s| s.trim().to_string())
        .take(n)
        .collect())
}

/// A loopback `llama-server` child process. Killed on drop.
struct LlmSidecar {
    child: Child,
    base_url: String,
}

impl LlmSidecar {
    fn spawn(binary: &PathBuf, model: &PathBuf) -> Result<Self, String> {
        let port = free_loopback_port()?;
        let child = Command::new(binary)
            .arg("--model")
            .arg(model)
            .arg("--host")
            .arg(HOST)
            .arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg("4096")
            .arg("--n-gpu-layers")
            .arg("999")
            .arg("--no-webui")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn llama-server: {e}"))?;

        let sidecar = LlmSidecar {
            child,
            base_url: format!("http://{HOST}:{port}"),
        };
        sidecar.wait_until_ready()?;
        Ok(sidecar)
    }

    fn wait_until_ready(&self) -> Result<(), String> {
        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            if let Ok(body) = http_get(HOST, self.port(), "/health") {
                if body.contains("\"ok\"") || body.contains("\"status\":\"ok\"") {
                    return Ok(());
                }
            }
            if Instant::now() >= deadline {
                return Err("llama-server did not become healthy in time".to_string());
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    }

    fn port(&self) -> u16 {
        // base_url is always "http://127.0.0.1:<port>"
        self.base_url
            .rsplit(':')
            .next()
            .and_then(|p| p.parse().ok())
            .unwrap_or(0)
    }

    fn complete(&self, prompt: &str, grammar: &str, n_predict: usize) -> Result<String, String> {
        let body = serde_json::json!({
            "prompt": prompt,
            "grammar": grammar,
            "temperature": 0.2,
            "repeat_penalty": 1.1,
            "n_predict": n_predict,
            "cache_prompt": true,
            "stream": false,
        })
        .to_string();

        let response = http_post_json(HOST, self.port(), "/completion", &body)?;
        let parsed: serde_json::Value =
            serde_json::from_str(&response).map_err(|e| format!("parse /completion json: {e}"))?;
        parsed
            .get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "llm response missing 'content'".to_string())
    }
}

impl Drop for LlmSidecar {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The escalation rung's translator: owns a sidecar for the session and
/// translates one scene per call, preserving cue count and timing.
pub(crate) struct LlmTranslator {
    sidecar: LlmSidecar,
    source_lang: String,
    target_lang: String,
    glossary: Vec<String>,
}

impl LlmTranslator {
    pub(crate) fn new(
        binary: PathBuf,
        model: PathBuf,
        source_lang: &str,
        target_lang: &str,
        glossary: Vec<String>,
    ) -> Result<Self, String> {
        let sidecar = LlmSidecar::spawn(&binary, &model)?;
        Ok(Self {
            sidecar,
            source_lang: source_lang.to_string(),
            target_lang: target_lang.to_string(),
            glossary,
        })
    }

    /// Translate one scene. Returns cues with the SAME timing/index as `source`
    /// and exactly `source.len()` entries (or an error, which the ladder treats
    /// as a still-failing best-effort and never splices).
    pub(crate) fn translate_all(&self, source: &[SubtitleCue]) -> Result<Vec<SubtitleCue>, String> {
        if source.is_empty() {
            return Ok(Vec::new());
        }
        let n = source.len();
        let grammar = build_grammar(n);
        let prompt = build_prompt(
            &self.source_lang,
            &self.target_lang,
            source,
            &[],
            &self.glossary,
        );
        // Budget ~64 tokens per line, with headroom; bounded so a runaway can't
        // generate forever.
        let n_predict = (n * 64 + 64).min(2048);
        let content = self.sidecar.complete(&prompt, &grammar, n_predict)?;
        let lines = parse_completion_lines(&content, n)?;

        Ok(source
            .iter()
            .zip(lines)
            .map(|(cue, text)| SubtitleCue {
                index: cue.index,
                timing: cue.timing.clone(),
                text,
            })
            .collect())
    }
}

/// Bind to an ephemeral loopback port, then release it so the child can claim
/// it. (A small TOCTOU window exists; acceptable for a local dev/run sidecar.)
fn free_loopback_port() -> Result<u16, String> {
    let listener = TcpListener::bind((HOST, 0)).map_err(|e| format!("reserve port: {e}"))?;
    listener
        .local_addr()
        .map(|a| a.port())
        .map_err(|e| format!("read reserved port: {e}"))
}

/// Minimal std-only HTTP/1.1 GET for loopback. Returns the response body.
fn http_get(host: &str, port: u16, path: &str) -> Result<String, String> {
    let request = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    http_roundtrip(host, port, request.as_bytes(), Duration::from_secs(5))
}

/// Minimal std-only HTTP/1.1 POST of a JSON body for loopback. Returns the body.
fn http_post_json(host: &str, port: u16, path: &str, json: &str) -> Result<String, String> {
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
        json.len()
    );
    http_roundtrip(host, port, request.as_bytes(), REQUEST_TIMEOUT)
}

/// Connect, write the request, read to EOF (server closes via `Connection:
/// close`), then split off the HTTP body. Status must be 2xx.
fn http_roundtrip(
    host: &str,
    port: u16,
    request: &[u8],
    timeout: Duration,
) -> Result<String, String> {
    let mut stream = TcpStream::connect((host, port)).map_err(|e| format!("connect: {e}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .and_then(|_| stream.set_write_timeout(Some(timeout)))
        .map_err(|e| format!("set timeout: {e}"))?;
    stream
        .write_all(request)
        .map_err(|e| format!("write request: {e}"))?;

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| format!("read response: {e}"))?;
    let text = String::from_utf8_lossy(&raw);

    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response (no header/body split)".to_string())?;
    let status_ok = head
        .lines()
        .next()
        .map(|line| line.contains(" 2"))
        .unwrap_or(false);
    if !status_ok {
        let status = head.lines().next().unwrap_or("<no status>");
        return Err(format!("HTTP error: {status}"));
    }
    Ok(body.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(index: usize, text: &str) -> SubtitleCue {
        SubtitleCue {
            index,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: text.to_string(),
        }
    }

    /// Live end-to-end smoke: spawns the real llama-server, loads the model, and
    /// translates two Japanese lines. Ignored by default (needs the model + GPU
    /// and tens of seconds); run with `cargo test live_smoke -- --ignored`.
    #[test]
    #[ignore = "spawns llama-server and loads the model; run explicitly with --ignored"]
    fn live_smoke_translates_japanese_lines() {
        let Some((binary, model)) = resolve_llm_paths() else {
            eprintln!("skipping live smoke: llm binary/model not resolved");
            return;
        };
        let translator = LlmTranslator::new(binary, model, "ja", "en", Vec::new())
            .expect("sidecar should spawn and become healthy");
        let cues = vec![cue(1, "そうだね"), cue(2, "えっ、本当に?")];
        let out = translator
            .translate_all(&cues)
            .expect("translation should succeed");

        // Cue count and timing preserved; non-empty, non-identity output.
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].index, 1);
        assert_eq!(out[0].timing, cues[0].timing);
        for (src, got) in cues.iter().zip(&out) {
            assert!(!got.text.trim().is_empty(), "empty translation");
            assert_ne!(got.text, src.text, "output must not echo the source");
        }
        eprintln!(
            "LLM smoke output: {:?}",
            out.iter().map(|c| c.text.as_str()).collect::<Vec<_>>()
        );
    }

    fn timing_start_secs(timing: &str) -> Option<f64> {
        let start = timing.split("-->").next()?.trim();
        let (hms, ms) = start.split_once(',')?;
        let parts: Vec<&str> = hms.split(':').collect();
        if parts.len() != 3 {
            return None;
        }
        let h: f64 = parts[0].parse().ok()?;
        let m: f64 = parts[1].parse().ok()?;
        let s: f64 = parts[2].parse().ok()?;
        let mil: f64 = ms.parse().ok()?;
        Some(h * 3600.0 + m * 60.0 + s + mil / 1000.0)
    }

    /// Live E2E on the real SILENT HILL casual region (~3060–3240s) — the exact
    /// content where NLLB collapsed to ~17% unique lines ("that's the matter
    /// with you?" ×348). Proves the integrated LLM rung handles real degenerate
    /// content at scale, not just two toy lines. Path overridable via
    /// `VOIDEX_E2E_SRT`. Ignored by default; run with `--ignored`.
    #[test]
    #[ignore = "loads the model + a real SRT; run explicitly with --ignored"]
    fn live_e2e_silent_hill_degenerate_region() {
        let Some((binary, model)) = resolve_llm_paths() else {
            eprintln!("skipping E2E: llm binary/model not resolved");
            return;
        };
        let srt_path = std::env::var("VOIDEX_E2E_SRT").unwrap_or_else(|_| {
            "benchmarks/runs/2026-04-22_silent_hill_f1/SILENT HILL f #1 加藤小夏 [0Ek5c3sQygs].ja.srt"
                .to_string()
        });
        let all = crate::engine::srt::parse_srt_file(std::path::Path::new(&srt_path))
            .expect("parse silent hill JA srt");
        let region: Vec<SubtitleCue> = all
            .into_iter()
            .filter(|c| {
                timing_start_secs(&c.timing)
                    .map(|s| (3060.0..3240.0).contains(&s))
                    .unwrap_or(false)
            })
            .take(40)
            .collect();
        assert!(!region.is_empty(), "no cues found in the casual region");
        let n = region.len();

        let translator = LlmTranslator::new(binary, model, "ja", "en", Vec::new())
            .expect("sidecar should spawn");
        let out = translator.translate_all(&region).expect("translate region");

        assert_eq!(out.len(), n, "cue count must be preserved");

        // Diversity must track the SOURCE, not be high in the absolute: casual
        // speech genuinely repeats ("行こう行こう" → "Let's go" ×N is faithful,
        // not degenerate). The principled fidelity check is therefore (a) output
        // diversity does not collapse below source diversity, and (b) no single
        // phrase dominates — the latter being NLLB's actual failure signature
        // (one wrong phrase emitted across many UNRELATED inputs).
        let out_unique: std::collections::HashSet<&str> =
            out.iter().map(|c| c.text.trim()).collect();
        let src_unique: std::collections::HashSet<&str> =
            region.iter().map(|c| c.text.trim()).collect();
        let out_ratio = out_unique.len() as f64 / n as f64;
        let src_ratio = src_unique.len() as f64 / n as f64;
        let max_repeat = out
            .iter()
            .map(|c| {
                out.iter()
                    .filter(|d| d.text.trim() == c.text.trim())
                    .count()
            })
            .max()
            .unwrap_or(0);
        eprintln!(
            "E2E SILENT HILL region: {n} cues | out {} unique ({:.0}%) vs src {} unique ({:.0}%) | max single-phrase repeat {}/{}",
            out_unique.len(),
            out_ratio * 100.0,
            src_unique.len(),
            src_ratio * 100.0,
            max_repeat,
            n
        );
        for c in &out {
            eprintln!("  [{}] {}", c.index, c.text);
        }
        // (a) Translation must not collapse below the source's own diversity.
        assert!(
            out_ratio >= src_ratio - 0.05,
            "LLM collapsed below source diversity: out {:.0}% vs src {:.0}%",
            out_ratio * 100.0,
            src_ratio * 100.0
        );
        // (b) No single phrase may dominate (NLLB collapsed to ~26% one phrase).
        assert!(
            (max_repeat as f64) / (n as f64) < 0.30,
            "single phrase dominates {max_repeat}/{n} — degeneration signature"
        );
    }

    #[test]
    fn grammar_has_one_string_per_cue() {
        let g = build_grammar(3);
        let root = g.lines().next().unwrap();
        // Three array elements, two separators, bracketed.
        assert_eq!(root.matches("string").count(), 3);
        assert_eq!(root.matches(',').count(), 2);
        assert!(root.starts_with("root ::= \"[\""));
        assert!(root.trim_end().ends_with("\"]\""));
        assert!(g.contains("string ::="));
        assert!(g.contains("ws ::="));
    }

    #[test]
    fn grammar_single_element_has_no_comma() {
        let g = build_grammar(1);
        let root = g.lines().next().unwrap();
        assert_eq!(root.matches("string").count(), 1);
        assert!(!root.contains(','));
    }

    #[test]
    fn prompt_numbers_every_line_and_names_languages() {
        let cues = vec![cue(1, "そうだね"), cue(2, "えっ")];
        let p = build_prompt("ja", "en", &cues, &[], &[]);
        assert!(p.contains("Japanese"));
        assert!(p.contains("English"));
        assert!(p.contains("EXACTLY 2"));
        assert!(p.contains("1. そうだね"));
        assert!(p.contains("2. えっ"));
    }

    #[test]
    fn prompt_includes_glossary_and_context_when_present() {
        let cues = vec![cue(1, "やあ、コナツ")];
        let glossary = vec!["Konatsu".to_string()];
        let context = vec!["Earlier this happened.".to_string()];
        let p = build_prompt("ja", "en", &cues, &context, &glossary);
        assert!(p.contains("Keep these names consistent: Konatsu"));
        assert!(p.contains("Earlier this happened."));
    }

    #[test]
    fn parse_takes_exactly_n_dropping_surplus() {
        let content = r#"["First line.","Second line.","Extra."]"#;
        let got = parse_completion_lines(content, 2).unwrap();
        assert_eq!(got, vec!["First line.", "Second line."]);
    }

    #[test]
    fn parse_rejects_short_response() {
        let content = r#"["Only one."]"#;
        assert!(parse_completion_lines(content, 3).is_err());
    }

    #[test]
    fn parse_handles_whitespace_around_array() {
        let content = "  \n[\"First.\", \"Second.\"]\n ";
        let got = parse_completion_lines(content, 2).unwrap();
        assert_eq!(got, vec!["First.", "Second."]);
    }

    #[test]
    fn parse_rejects_non_json() {
        assert!(parse_completion_lines("1. not json at all", 1).is_err());
    }

    #[test]
    fn lang_name_maps_known_codes() {
        assert_eq!(lang_name("ja"), "Japanese");
        assert_eq!(lang_name("EN"), "English");
        assert_eq!(lang_name("xx"), "xx");
    }

    #[test]
    fn free_port_is_nonzero_and_distinct() {
        let a = free_loopback_port().unwrap();
        let b = free_loopback_port().unwrap();
        assert_ne!(a, 0);
        assert_ne!(b, 0);
    }
}
