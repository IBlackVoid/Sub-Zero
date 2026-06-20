use crate::engine::cache::{cached, ContentCache};
use memchr::memchr_iter;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtitleCue {
    pub index: usize,
    pub timing: String,
    pub text: String,
}

#[derive(Debug, thiserror::Error)]
pub enum SrtError {
    #[error("failed to read SRT file {}: {message}", path.display())]
    ReadFile { path: PathBuf, message: String },
    #[error("invalid SRT: {message}")]
    Parse { message: String },
    #[error("failed to parse SRT file {}: {message}", path.display())]
    ParseFile { path: PathBuf, message: String },
    #[error("failed to write SRT file {}: {message}", path.display())]
    WriteFile { path: PathBuf, message: String },
    #[error("SRT input exceeds size cap: {bytes} > {max_bytes}")]
    SizeCap { bytes: u64, max_bytes: u64 },
}

pub type SrtResult<T> = Result<T, SrtError>;

/// Upper bound on the byte length of any SRT input `parse_srt` will
/// accept. SRT is a plain-text cue stream; any single file larger than
/// this is either an upload-DoS or a misidentified payload. The cap is
/// intentionally generous (50 MiB is many hours of dialogue) so genuine
/// content always parses, but a malicious 4 GiB input still bounces.
pub const MAX_SRT_INPUT_BYTES: u64 = 50 * 1024 * 1024;

pub fn parse_srt_file(path: &Path) -> SrtResult<Vec<SubtitleCue>> {
    // Stat the file before reading: if it exceeds the cap, fail with a
    // typed error instead of OOMing on `read_to_string`.
    let len = fs::metadata(path)
        .map_err(|e| SrtError::ReadFile {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?
        .len();
    if len > MAX_SRT_INPUT_BYTES {
        return Err(SrtError::SizeCap {
            bytes: len,
            max_bytes: MAX_SRT_INPUT_BYTES,
        });
    }

    let content = fs::read_to_string(path).map_err(|e| SrtError::ReadFile {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    parse_srt(&content).map_err(|error| match error {
        SrtError::Parse { message } => SrtError::ParseFile {
            path: path.to_path_buf(),
            message,
        },
        other => other,
    })
}

/// Parse an SRT cue stream from `content`. The parser:
///
/// - Streams over `&str` directly — no `replace("\r\n", "\n")` copy.
/// - Uses `memchr` to scan line breaks (SIMD where available).
/// - Tolerates LF, CRLF, and mixed line endings (per-line `\r` strip).
/// - Synthesises a sequential index if the first line of a block is
///   already a timing line.
/// - Caps input at `MAX_SRT_INPUT_BYTES` so a hostile or corrupt input
///   is rejected before allocating cues.
pub fn parse_srt(content: &str) -> SrtResult<Vec<SubtitleCue>> {
    if (content.len() as u64) > MAX_SRT_INPUT_BYTES {
        return Err(SrtError::SizeCap {
            bytes: content.len() as u64,
            max_bytes: MAX_SRT_INPUT_BYTES,
        });
    }

    let bytes = content.as_bytes();
    let total = bytes.len();
    let mut cues = Vec::<SubtitleCue>::new();
    let mut block: Vec<&str> = Vec::with_capacity(4);

    // Walk line starts via memchr; each line is `content[cursor..end)`
    // with a trailing `\r` stripped to handle CRLF without a global
    // normalisation pass.
    let mut cursor = 0usize;
    let mut newlines = memchr_iter(b'\n', bytes);
    loop {
        let line_end = newlines.next().unwrap_or(total);
        let raw = &content[cursor..line_end];
        let line = raw.strip_suffix('\r').unwrap_or(raw);

        if line.trim().is_empty() {
            if !block.is_empty() {
                flush_block(&block, &mut cues)?;
                block.clear();
            }
        } else {
            block.push(line);
        }

        if line_end >= total {
            break;
        }
        cursor = line_end + 1; // past the '\n'
    }
    if !block.is_empty() {
        flush_block(&block, &mut cues)?;
    }

    if cues.is_empty() {
        return Err(SrtError::Parse {
            message: "no cues found in SRT".to_string(),
        });
    }
    Ok(cues)
}

/// Function-id namespace under which `parse_srt` results are cached.
/// Bump the version suffix whenever `parse_srt`'s output schema or
/// semantics change so the cache invalidates instead of returning
/// stale results. The Rust loader of cached blobs decodes JSON and
/// will refuse a value whose shape no longer matches.
pub const PARSE_SRT_CACHE_ID: &str = "engine::srt::parse_srt:v1";

/// Content-addressed variant of [`parse_srt`]. On cache hit, the
/// pre-parsed cue list is JSON-decoded from the cache. On miss,
/// [`parse_srt`] runs, the result is JSON-encoded and stored, then
/// returned. Errors from a cache *miss* propagate; errors from
/// deserialising a corrupted cache entry fall back to recompute.
///
/// This is the ADR-0001 Phase A pilot. The point is to prove
/// content-addressing works for a real engine function before
/// committing to the full Phase A–E plan. See
/// `docs/adr/0001-content-addressed-build-graph.md` for the rationale.
pub fn parse_srt_cached(content: &str, cache: &dyn ContentCache) -> SrtResult<Vec<SubtitleCue>> {
    // The cache key is `(function_id, content_bytes)`. The
    // function-id is the version-stamped namespace; the value is the
    // JSON-encoded cue list. We do *not* check the size cap here —
    // the cap is enforced by `parse_srt` on miss. A cache hit that
    // returns a cached parse of an oversize input would only happen
    // if a prior version of this code was lax; we treat the cached
    // blob as authoritative once accepted.
    let bytes = cached(cache, PARSE_SRT_CACHE_ID, content.as_bytes(), || {
        // On miss: run the real parser and JSON-encode the result.
        // If parse fails, store an empty marker so we don't thrash;
        // the caller will see the error via the second call below.
        match parse_srt(content) {
            Ok(cues) => serde_json::to_vec(&cues).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    });
    if bytes.is_empty() {
        // Either a cache-side decode failure, or the underlying
        // parse_srt errored. Run the parser inline so the caller
        // sees the typed error.
        return parse_srt(content);
    }
    // Cache hit (or fresh miss-with-success): decode.
    match serde_json::from_slice::<Vec<SubtitleCue>>(&bytes) {
        Ok(cues) => Ok(cues),
        Err(_) => {
            // Corrupted cache entry. Fall through to recompute and
            // let the caller observe the canonical error.
            parse_srt(content)
        }
    }
}

fn flush_block(lines: &[&str], cues: &mut Vec<SubtitleCue>) -> SrtResult<()> {
    debug_assert!(!lines.is_empty());
    // First-line rule: a numeric leading line is the cue index; anything
    // else is taken as the timing line and the index is synthesised.
    let first = lines[0].trim();
    let (index, timing_idx) = match first.parse::<usize>() {
        Ok(parsed) => (parsed, 1usize),
        Err(_) => (cues.len() + 1, 0usize),
    };
    let timing_line = lines.get(timing_idx).ok_or_else(|| SrtError::Parse {
        message: format!("missing timing line for cue {index}"),
    })?;
    // Remaining lines are body text joined with `\n` to preserve the
    // pre-memchr-refactor output format byte-for-byte.
    let text = if lines.len() > timing_idx + 1 {
        lines[timing_idx + 1..].join("\n")
    } else {
        String::new()
    };
    cues.push(SubtitleCue {
        index,
        timing: (*timing_line).to_string(),
        text,
    });
    Ok(())
}

pub fn write_srt_file(path: &Path, cues: &[SubtitleCue]) -> SrtResult<()> {
    let mut file = fs::File::create(path).map_err(|e| SrtError::WriteFile {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    for (position, cue) in cues.iter().enumerate() {
        writeln!(file, "{}", position + 1).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        writeln!(file, "{}", cue.timing).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        writeln!(file, "{}", cue.text.trim_end()).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        writeln!(file).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
    }

    Ok(())
}

/// Append cues to an SRT file without renumbering previously-written cues.
///
/// Returns the next index after the last written cue.
pub fn append_srt_file(
    path: &Path,
    cues: &[SubtitleCue],
    starting_index: usize,
) -> SrtResult<usize> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

    for (position, cue) in cues.iter().enumerate() {
        let index = starting_index + position;
        writeln!(file, "{index}").map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        writeln!(file, "{}", cue.timing).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        writeln!(file, "{}", cue.text.trim_end()).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        writeln!(file).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
    }

    file.flush().map_err(|e| SrtError::WriteFile {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    Ok(starting_index + cues.len())
}

#[cfg(test)]
mod tests {
    use super::{
        append_srt_file, parse_srt, parse_srt_cached, parse_srt_file, write_srt_file, SubtitleCue,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        std::env::temp_dir().join(format!("voidex_{name}_{stamp}.srt"))
    }

    #[test]
    fn parse_standard_srt() {
        let input = "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n2\n00:00:01,000 --> 00:00:03,000\nline 1\nline 2\n";
        let cues = parse_srt(input).expect("parse should succeed");

        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].index, 1);
        assert_eq!(cues[0].timing, "00:00:00,000 --> 00:00:01,000");
        assert_eq!(cues[1].text, "line 1\nline 2");
    }

    #[test]
    fn parse_without_numeric_index() {
        let input = "00:00:00,000 --> 00:00:01,000\nhello";
        let cues = parse_srt(input).expect("parse should succeed");

        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].index, 1);
        assert_eq!(cues[0].timing, "00:00:00,000 --> 00:00:01,000");
        assert_eq!(cues[0].text, "hello");
    }

    #[test]
    fn write_srt_file_renumbers_and_writes() {
        let path = temp_file("write");
        let cues = vec![
            SubtitleCue {
                index: 12,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "hello".to_string(),
            },
            SubtitleCue {
                index: 24,
                timing: "00:00:01,000 --> 00:00:02,000".to_string(),
                text: "world".to_string(),
            },
        ];

        write_srt_file(&path, &cues).expect("write should succeed");
        let output = fs::read_to_string(&path).expect("output should be readable");

        assert!(output.contains("1\n00:00:00,000 --> 00:00:01,000\nhello"));
        assert!(output.contains("2\n00:00:01,000 --> 00:00:02,000\nworld"));
    }

    #[test]
    fn append_srt_file_appends_with_custom_starting_index() {
        let path = temp_file("append");
        let cues_first = vec![SubtitleCue {
            index: 999,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: "A".to_string(),
        }];
        let cues_second = vec![
            SubtitleCue {
                index: 999,
                timing: "00:00:02,000 --> 00:00:03,000".to_string(),
                text: "B".to_string(),
            },
            SubtitleCue {
                index: 999,
                timing: "00:00:04,000 --> 00:00:05,000".to_string(),
                text: "C".to_string(),
            },
        ];

        let next = append_srt_file(&path, &cues_first, 1).expect("append should succeed");
        assert_eq!(next, 2);
        let next = append_srt_file(&path, &cues_second, next).expect("append should succeed");
        assert_eq!(next, 4);

        let parsed = parse_srt_file(&path).expect("parse should succeed");
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].index, 1);
        assert_eq!(parsed[0].text, "A");
        assert_eq!(parsed[1].index, 2);
        assert_eq!(parsed[1].text, "B");
        assert_eq!(parsed[2].index, 3);
        assert_eq!(parsed[2].text, "C");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn generated_roundtrip_preserves_timing_text_and_sequential_indices() {
        for cue_count in 1..=6usize {
            for multiline in [false, true] {
                let path = temp_file(&format!("roundtrip_{cue_count}_{multiline}"));
                let cues = (0..cue_count)
                    .map(|i| SubtitleCue {
                        index: i + 10,
                        timing: format!("00:00:{:02},000 --> 00:00:{:02},500", i * 2, i * 2 + 1),
                        text: if multiline {
                            format!("line {i}\ncontinuation {i}")
                        } else {
                            format!("line {i}")
                        },
                    })
                    .collect::<Vec<_>>();

                write_srt_file(&path, &cues).expect("write should succeed");
                let reparsed = parse_srt_file(&path).expect("roundtrip parse should succeed");

                assert_eq!(reparsed.len(), cues.len());
                for (index, cue) in reparsed.iter().enumerate() {
                    assert_eq!(cue.index, index + 1);
                    assert_eq!(cue.timing, cues[index].timing);
                    assert_eq!(cue.text, cues[index].text.trim_end());
                }

                let _ = fs::remove_file(path);
            }
        }
    }

    #[test]
    fn parse_srt_cached_matches_uncached_on_hit_and_miss() {
        // ADR-0001 Phase A pilot invariant: cached parse and uncached
        // parse must return byte-identical Vec<SubtitleCue> on both
        // miss (first call) and hit (subsequent call). If this ever
        // diverges, the cache has corrupted the function's output
        // contract — which would be a Phase A kill condition.
        use crate::engine::cache::MemoryContentCache;
        let input = "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n2\n00:00:01,000 --> 00:00:03,000\nline 1\nline 2\n";
        let uncached = parse_srt(input).expect("uncached parse");
        let cache = MemoryContentCache::new();
        let miss = parse_srt_cached(input, &cache).expect("cache miss parse");
        assert_eq!(miss, uncached, "miss must equal uncached parse");
        let hit = parse_srt_cached(input, &cache).expect("cache hit parse");
        assert_eq!(hit, uncached, "hit must equal uncached parse");
    }

    #[test]
    fn parse_handles_crlf_line_endings_without_normalisation() {
        let input = "1\r\n00:00:00,000 --> 00:00:01,000\r\nhello\r\n\r\n2\r\n00:00:01,000 --> 00:00:03,000\r\nline 1\r\nline 2\r\n";
        let cues = parse_srt(input).expect("CRLF SRT should parse");
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].timing, "00:00:00,000 --> 00:00:01,000");
        assert_eq!(cues[1].text, "line 1\nline 2");
    }

    #[test]
    fn parse_handles_mixed_line_endings() {
        let input =
            "1\n00:00:00,000 --> 00:00:01,000\nfoo\n\n2\r\n00:00:02,000 --> 00:00:03,000\r\nbar";
        let cues = parse_srt(input).expect("mixed-ending SRT should parse");
        assert_eq!(cues.len(), 2);
        assert_eq!(cues[0].text, "foo");
        assert_eq!(cues[1].text, "bar");
    }

    #[test]
    fn parse_rejects_input_above_size_cap() {
        use crate::engine::srt::{SrtError, MAX_SRT_INPUT_BYTES};
        // Build a string just one byte over the cap. We don't need it to
        // be a valid SRT — the cap check runs before any parsing.
        let oversize = "x".repeat(MAX_SRT_INPUT_BYTES as usize + 1);
        match parse_srt(&oversize) {
            Err(SrtError::SizeCap { bytes, max_bytes }) => {
                assert!(bytes > max_bytes);
            }
            other => panic!("expected SizeCap, got {other:?}"),
        }
    }

    #[test]
    fn generated_srt_without_numeric_indices_is_reindexed_in_order() {
        for cue_count in 1..=5usize {
            let mut input = String::new();
            for i in 0..cue_count {
                input.push_str(&format!(
                    "00:00:{:02},000 --> 00:00:{:02},250\nbody {}\nextra {}\n\n",
                    i * 3,
                    i * 3 + 1,
                    i,
                    i
                ));
            }

            let cues = parse_srt(&input).expect("parse should succeed");
            assert_eq!(cues.len(), cue_count);
            for (index, cue) in cues.iter().enumerate() {
                assert_eq!(cue.index, index + 1);
                assert_eq!(
                    cue.timing,
                    format!(
                        "00:00:{:02},000 --> 00:00:{:02},250",
                        index * 3,
                        index * 3 + 1
                    )
                );
                assert_eq!(cue.text, format!("body {}\nextra {}", index, index));
            }
        }
    }
}
