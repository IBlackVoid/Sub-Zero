use crate::engine::srt::parse_srt_file;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub(super) struct OutputHealthSnapshot {
    pub(super) cue_count: usize,
    pub(super) non_empty_ratio: f64,
    pub(super) top_line_ratio: f64,
}

pub(super) fn assess_output_health(path: &Path) -> Result<OutputHealthSnapshot, String> {
    let cues = parse_srt_file(path).map_err(|error| error.to_string())?;
    if cues.is_empty() {
        return Ok(OutputHealthSnapshot {
            cue_count: 0,
            non_empty_ratio: 0.0,
            top_line_ratio: 0.0,
        });
    }

    let mut freq = HashMap::<String, usize>::new();
    let mut non_empty = 0usize;
    for cue in &cues {
        let normalized = cue
            .text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        if normalized.is_empty() {
            continue;
        }
        non_empty += 1;
        *freq.entry(normalized).or_insert(0) += 1;
    }

    let top_count = freq.values().copied().max().unwrap_or(0);
    let total = cues.len() as f64;
    Ok(OutputHealthSnapshot {
        cue_count: cues.len(),
        non_empty_ratio: (non_empty as f64) / total,
        top_line_ratio: (top_count as f64) / total,
    })
}
