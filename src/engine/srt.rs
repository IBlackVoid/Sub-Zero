use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtitleCue {
    pub index: usize,
    pub timing: String,
    pub text: String,
}

#[derive(Debug)]
pub enum SrtError {
    ReadFile { path: PathBuf, source: String },
    Parse { source: String },
    ParseFile { path: PathBuf, source: String },
    WriteFile { path: PathBuf, source: String },
}

pub type SrtResult<T> = Result<T, SrtError>;

impl std::fmt::Display for SrtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadFile { path, source } => {
                write!(f, "failed to read SRT file {}: {source}", path.display())
            }
            Self::Parse { source } => write!(f, "invalid SRT: {source}"),
            Self::ParseFile { path, source } => {
                write!(f, "failed to parse SRT file {}: {source}", path.display())
            }
            Self::WriteFile { path, source } => {
                write!(f, "failed to write SRT file {}: {source}", path.display())
            }
        }
    }
}

impl std::error::Error for SrtError {}

pub fn parse_srt_file(path: &Path) -> SrtResult<Vec<SubtitleCue>> {
    let content = fs::read_to_string(path).map_err(|e| SrtError::ReadFile {
        path: path.to_path_buf(),
        source: e.to_string(),
    })?;
    parse_srt(&content).map_err(|error| match error {
        SrtError::Parse { source } => SrtError::ParseFile {
            path: path.to_path_buf(),
            source,
        },
        other => other,
    })
}

pub fn parse_srt(content: &str) -> SrtResult<Vec<SubtitleCue>> {
    let normalized = content.replace("\r\n", "\n");
    let mut cues = Vec::new();

    for block in normalized.split("\n\n") {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut lines = trimmed.lines();
        let first_line = lines.next().ok_or_else(|| SrtError::Parse {
            source: "invalid SRT block with no lines".to_string(),
        })?;

        let (index, timing) = match first_line.trim().parse::<usize>() {
            Ok(parsed_index) => {
                let timing_line = lines.next().ok_or_else(|| SrtError::Parse {
                    source: format!("missing timing line for cue {parsed_index}"),
                })?;
                (parsed_index, timing_line.to_string())
            }
            Err(_) => (cues.len() + 1, first_line.to_string()),
        };

        let text = lines.collect::<Vec<_>>().join("\n");
        cues.push(SubtitleCue {
            index,
            timing,
            text,
        });
    }

    if cues.is_empty() {
        return Err(SrtError::Parse {
            source: "no cues found in SRT".to_string(),
        });
    }
    Ok(cues)
}

pub fn write_srt_file(path: &Path, cues: &[SubtitleCue]) -> SrtResult<()> {
    let mut file = fs::File::create(path).map_err(|e| SrtError::WriteFile {
        path: path.to_path_buf(),
        source: e.to_string(),
    })?;

    for (position, cue) in cues.iter().enumerate() {
        writeln!(file, "{}", position + 1).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            source: e.to_string(),
        })?;
        writeln!(file, "{}", cue.timing).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            source: e.to_string(),
        })?;
        writeln!(file, "{}", cue.text.trim_end()).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            source: e.to_string(),
        })?;
        writeln!(file).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            source: e.to_string(),
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
            source: e.to_string(),
        })?;

    for (position, cue) in cues.iter().enumerate() {
        let index = starting_index + position;
        writeln!(file, "{index}").map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            source: e.to_string(),
        })?;
        writeln!(file, "{}", cue.timing).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            source: e.to_string(),
        })?;
        writeln!(file, "{}", cue.text.trim_end()).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            source: e.to_string(),
        })?;
        writeln!(file).map_err(|e| SrtError::WriteFile {
            path: path.to_path_buf(),
            source: e.to_string(),
        })?;
    }

    file.flush().map_err(|e| SrtError::WriteFile {
        path: path.to_path_buf(),
        source: e.to_string(),
    })?;

    Ok(starting_index + cues.len())
}

#[cfg(test)]
mod tests {
    use super::{append_srt_file, parse_srt, parse_srt_file, write_srt_file, SubtitleCue};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        std::env::temp_dir().join(format!("sub_zero_{name}_{stamp}.srt"))
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
