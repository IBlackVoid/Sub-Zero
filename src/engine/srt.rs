use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtitleCue {
    pub index: usize,
    pub timing: String,
    pub text: String,
}

pub fn parse_srt_file(path: &Path) -> Result<Vec<SubtitleCue>, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("failed to read SRT file {}: {e}", path.display()))?;
    parse_srt(&content)
}

pub fn parse_srt(content: &str) -> Result<Vec<SubtitleCue>, String> {
    let normalized = content.replace("\r\n", "\n");
    let mut cues = Vec::new();

    for block in normalized.split("\n\n") {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut lines = trimmed.lines();
        let first_line = lines
            .next()
            .ok_or_else(|| "invalid SRT block with no lines".to_string())?;

        let (index, timing) = match first_line.trim().parse::<usize>() {
            Ok(parsed_index) => {
                let timing_line = lines
                    .next()
                    .ok_or_else(|| format!("missing timing line for cue {parsed_index}"))?;
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
        return Err("no cues found in SRT".to_string());
    }
    Ok(cues)
}

pub fn write_srt_file(path: &Path, cues: &[SubtitleCue]) -> Result<(), String> {
    let mut file = fs::File::create(path)
        .map_err(|e| format!("failed to create output {}: {e}", path.display()))?;

    for (position, cue) in cues.iter().enumerate() {
        writeln!(file, "{}", position + 1).map_err(|e| e.to_string())?;
        writeln!(file, "{}", cue.timing).map_err(|e| e.to_string())?;
        writeln!(file, "{}", cue.text.trim_end()).map_err(|e| e.to_string())?;
        writeln!(file).map_err(|e| e.to_string())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_srt, write_srt_file, SubtitleCue};
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
}
