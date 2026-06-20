use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn bin_path() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_voidex") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_voidex") {
        return PathBuf::from(path);
    }

    let exe_name = if cfg!(windows) {
        "voidex.exe"
    } else {
        "voidex"
    };
    let fallback = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().and_then(|p| p.parent()).map(PathBuf::from))
        .map(|deps_parent| deps_parent.join(exe_name));
    if let Some(path) = fallback {
        if path.is_file() {
            return path;
        }
    }

    panic!("missing binary path for voidex");
}

fn temp_case_dir(name: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after UNIX_EPOCH")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("voidex_it_{name}_{stamp}"));
    fs::create_dir_all(&path).expect("temp dir should be creatable");
    path
}

fn parse_srt(path: &Path) -> Vec<(String, String)> {
    let content = fs::read_to_string(path).expect("srt should be readable");
    let normalized = content.replace("\r\n", "\n");
    let mut cues = Vec::new();
    for block in normalized.split("\n\n") {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            continue;
        }
        let lines: Vec<&str> = trimmed.lines().collect();
        if lines.len() < 3 {
            continue;
        }
        let timing = lines[1].trim().to_string();
        let text = lines[2..]
            .iter()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        cues.push((timing, text));
    }
    cues
}

fn japanese_char_ratio(text: &str) -> f64 {
    let mut total = 0usize;
    let mut jp = 0usize;
    for ch in text.chars() {
        if ch.is_whitespace() {
            continue;
        }
        total += 1;
        let is_jp = ('\u{3040}'..='\u{309F}').contains(&ch)
            || ('\u{30A0}'..='\u{30FF}').contains(&ch)
            || ('\u{4E00}'..='\u{9FFF}').contains(&ch);
        if is_jp {
            jp += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        jp as f64 / total as f64
    }
}

#[test]
fn help_flag_exits_success() {
    let output = Command::new(bin_path())
        .arg("--help")
        .output()
        .expect("help command should run");
    assert!(output.status.success(), "help should exit 0");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Usage:"),
        "help output should include usage"
    );
}

#[test]
fn phrase_table_cli_smoke() {
    let dir = temp_case_dir("phrase_table_smoke");
    let source = dir.join("sample.ja.srt");
    fs::write(
        &source,
        "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n\n2\n00:00:01,000 --> 00:00:02,000\nありがとう\n",
    )
    .expect("fixture should be writable");

    let output = Command::new(bin_path())
        .arg("-i")
        .arg(&source)
        .arg("--source-lang")
        .arg("ja")
        .arg("--lang")
        .arg("en")
        .arg("--offline")
        .arg("--phrase-table")
        .output()
        .expect("phrase-table CLI smoke should run");

    assert!(
        output.status.success(),
        "phrase-table run failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let translated = dir.join("sample.ja.en.srt");
    assert!(translated.is_file(), "expected translated file to exist");
    let translated_content =
        fs::read_to_string(&translated).expect("translated SRT should be readable");
    // postprocess() capitalizes sentence starts, so the phrase-table output
    // is "Hello" / "Thank you" (see fix_capitalization). Matches the
    // pipeline unit test process_file_translates_and_writes.
    assert!(translated_content.contains("Hello"));
    assert!(translated_content.contains("Thank you"));
}

#[test]
fn ffmpeg_ffprobe_smoke() {
    if std::env::var("VOIDEX_RUN_FFMPEG_SMOKE").ok().as_deref() != Some("1") {
        return;
    }

    let input = PathBuf::from("clip_10s.wav");
    assert!(
        input.is_file(),
        "expected clip_10s.wav in repository root for ffmpeg smoke test"
    );

    let probe = Command::new("ffprobe")
        .arg("-hide_banner")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(&input)
        .output()
        .expect("ffprobe should be runnable");
    assert!(
        probe.status.success(),
        "ffprobe failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&probe.stdout),
        String::from_utf8_lossy(&probe.stderr)
    );

    let duration = String::from_utf8_lossy(&probe.stdout)
        .trim()
        .parse::<f64>()
        .expect("ffprobe should output a numeric duration");
    assert!(duration > 0.5, "ffprobe duration should be > 0.5s");

    let dir = temp_case_dir("ffmpeg_smoke");
    let out = dir.join("resampled.wav");
    let ffmpeg = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-nostdin")
        .arg("-v")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(&input)
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg(&out)
        .output()
        .expect("ffmpeg should be runnable");
    assert!(
        ffmpeg.status.success(),
        "ffmpeg failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&ffmpeg.stdout),
        String::from_utf8_lossy(&ffmpeg.stderr)
    );
    assert!(out.is_file(), "ffmpeg output file should exist");
}

#[test]
fn neural_mt_subtitle_quality_smoke() {
    if std::env::var("VOIDEX_RUN_NEURAL_SMOKE").ok().as_deref() != Some("1") {
        return;
    }

    let source = PathBuf::from("clip_10s.ja.srt");
    let reference = PathBuf::from("clip_10s.en.srt");
    assert!(
        source.is_file(),
        "missing source fixture: {}",
        source.display()
    );
    assert!(
        reference.is_file(),
        "missing reference fixture: {}",
        reference.display()
    );

    let dir = temp_case_dir("neural_smoke");
    let source_copy = dir.join("clip_10s.ja.srt");
    fs::copy(&source, &source_copy).expect("source fixture should copy");

    let output = Command::new(bin_path())
        .arg("-i")
        .arg(&source_copy)
        .arg("--source-lang")
        .arg("ja")
        .arg("--lang")
        .arg("en")
        .arg("--offline")
        .output()
        .expect("neural MT smoke should run");

    assert!(
        output.status.success(),
        "neural MT smoke failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let translated = dir.join("clip_10s.ja.en.srt");
    assert!(
        translated.is_file(),
        "expected neural translated file: {}",
        translated.display()
    );

    let ref_cues = parse_srt(&reference);
    let hyp_cues = parse_srt(&translated);
    assert_eq!(
        ref_cues.len(),
        hyp_cues.len(),
        "translated cue count should match reference cue count"
    );

    let timing_matches = ref_cues
        .iter()
        .zip(&hyp_cues)
        .filter(|((ref_timing, _), (hyp_timing, _))| ref_timing == hyp_timing)
        .count();
    assert_eq!(
        timing_matches,
        ref_cues.len(),
        "translated cues should preserve all timings"
    );

    let hyp_text_joined = hyp_cues
        .iter()
        .map(|(_, text)| text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let jp_ratio = japanese_char_ratio(&hyp_text_joined);
    assert!(
        jp_ratio <= 0.20,
        "translated subtitles still contain too many Japanese characters (ratio={jp_ratio:.3})"
    );

    let avg_line_similarity = ref_cues
        .iter()
        .zip(&hyp_cues)
        .map(|((_, ref_text), (_, hyp_text))| {
            strsim::normalized_levenshtein(&ref_text.to_lowercase(), &hyp_text.to_lowercase())
        })
        .sum::<f64>()
        / (ref_cues.len() as f64).max(1.0);
    assert!(
        avg_line_similarity >= 0.20,
        "translated subtitles are too far from reference (avg line similarity={avg_line_similarity:.3})"
    );
}
