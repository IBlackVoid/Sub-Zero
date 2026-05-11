use super::{
    cue_has_adjacent_repeat, cue_has_low_function_word_coverage, format_srt_timing_line,
    has_repeated_ngram, parse_srt_timing_line, tokenize_ascii_words,
};
use crate::engine::srt::SubtitleCue;

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CueCompactionStats {
    pub(super) merged_pairs: usize,
    pub(super) dropped_duplicates: usize,
}

pub(super) fn compact_adjacent_cues(
    cues: &[SubtitleCue],
    max_gap_s: f64,
    max_chars_per_line: usize,
    max_lines: usize,
    max_cps: f64,
    max_duration_s: f64,
) -> Result<(Vec<SubtitleCue>, CueCompactionStats), String> {
    if cues.is_empty() {
        return Ok((Vec::new(), CueCompactionStats::default()));
    }

    let mut out = Vec::<SubtitleCue>::new();
    let mut stats = CueCompactionStats::default();
    let max_merge_pairs = (cues.len() / 16).max(20);
    let mut idx = 0usize;

    while idx < cues.len() {
        let mut current = cues[idx].clone();
        current.text = normalize_compaction_text(&current.text);
        let (start, mut end) = parse_srt_timing_line(&current.timing)?;
        let mut current_text = current.text.clone();
        let mut lookahead = idx + 1;

        while lookahead < cues.len() {
            let next = &cues[lookahead];
            let (next_start, next_end) = parse_srt_timing_line(&next.timing)?;
            let next_text = normalize_compaction_text(&next.text);
            if next_text.is_empty() {
                lookahead += 1;
                continue;
            }

            let gap = (next_start - end).max(0.0);
            if gap > max_gap_s {
                break;
            }

            if normalized_text_key(&current_text) == normalized_text_key(&next_text)
                && gap <= 0.12
                && current_text.chars().count() <= 48
            {
                end = next_end.max(end);
                stats.dropped_duplicates += 1;
                lookahead += 1;
                continue;
            }

            if !can_merge_cue_texts(&current_text, &next_text) {
                break;
            }
            if stats.merged_pairs >= max_merge_pairs {
                break;
            }

            let merged_text = merge_cue_texts(&current_text, &next_text);
            let merged_duration = (next_end - start).max(0.001);
            let merged_chars = merged_text.chars().count() as f64;
            let merged_cps = merged_chars / merged_duration;
            let merged_line_budget = max_chars_per_line * max_lines;
            let merged_tokens = tokenize_ascii_words(&merged_text);
            if merged_duration > max_duration_s
                || merged_cps > max_cps
                || merged_text.chars().count() > merged_line_budget
                || cue_has_adjacent_repeat(&merged_tokens)
                || has_repeated_ngram(&merged_tokens, 3)
                || cue_has_low_function_word_coverage(&merged_tokens)
            {
                break;
            }

            current_text = merged_text;
            end = next_end.max(end);
            stats.merged_pairs += 1;
            lookahead += 1;
        }

        out.push(SubtitleCue {
            index: current.index,
            timing: format_srt_timing_line(start, end),
            text: current_text,
        });
        idx = lookahead.max(idx + 1);
    }

    Ok((out, stats))
}

fn normalize_compaction_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalized_text_key(text: &str) -> String {
    normalize_compaction_text(text).to_ascii_lowercase()
}

fn can_merge_cue_texts(left: &str, right: &str) -> bool {
    if left.is_empty() || right.is_empty() {
        return false;
    }

    if left.contains('[') || left.contains(']') || right.contains('[') || right.contains(']') {
        return false;
    }

    let left_trim = left.trim_end();
    let right_trim = right.trim_start();
    let left_terminal = left_trim.chars().last().unwrap_or(' ');
    let right_initial = right_trim.chars().next().unwrap_or(' ');
    let left_words = left_trim.split_whitespace().count();
    let right_words = right_trim.split_whitespace().count();

    if left_words > 8 || right_words > 8 {
        return false;
    }

    if right_initial.is_ascii_uppercase() {
        return false;
    }
    if matches!(left_terminal, '.' | '?' | '!' | ':') {
        return false;
    }
    if left_trim.ends_with("...") && right_initial.is_ascii_alphabetic() {
        return false;
    }
    true
}

fn merge_cue_texts(left: &str, right: &str) -> String {
    let left = left.trim_end();
    let right = right.trim_start();
    if left.is_empty() {
        return right.to_string();
    }
    if right.is_empty() {
        return left.to_string();
    }
    format!("{left} {right}")
}
