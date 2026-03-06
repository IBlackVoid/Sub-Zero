// Context Window Builder — Theorem 3 of SRT².
//
// Constructs sliding-window context for each cue so the neural MT model
// can resolve zero-anaphora, register shifts, and maintain name consistency.

use crate::engine::srt::SubtitleCue;
use serde::{Deserialize, Serialize};

/// A cue bundled with its surrounding context, ready for batched MT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualCue {
    /// Index of this cue in the original sequence.
    pub index: usize,
    /// Lines of preceding context (up to `radius` cues).
    pub prev_lines: Vec<String>,
    /// The actual line to translate.
    pub current_line: String,
    /// Lines of following context (up to `radius` cues).
    pub next_lines: Vec<String>,
    /// The original timing string (passed through).
    pub timing: String,
    /// Optional context tags used by downstream adaptive decoders.
    pub context_tags: Vec<String>,
}

/// Build context windows of radius `r` around each cue.
///
/// For a cue at position `i`, the window includes
///   `cues[i-r..i]` as prev context and `cues[i+1..i+1+r]` as next context.
pub fn build_context_windows(cues: &[SubtitleCue], radius: usize) -> Vec<ContextualCue> {
    build_context_windows_with_tags(cues, radius, &[])
}

/// Build context windows with optional per-cue tags.
pub fn build_context_windows_with_tags(
    cues: &[SubtitleCue],
    radius: usize,
    tags: &[Vec<String>],
) -> Vec<ContextualCue> {
    let n = cues.len();
    cues.iter()
        .enumerate()
        .map(|(i, cue)| {
            let prev_start = i.saturating_sub(radius);
            let prev_lines: Vec<String> =
                cues[prev_start..i].iter().map(|c| c.text.clone()).collect();

            let next_end = (i + 1 + radius).min(n);
            let next_lines: Vec<String> = cues[i + 1..next_end]
                .iter()
                .map(|c| c.text.clone())
                .collect();

            ContextualCue {
                index: cue.index,
                prev_lines,
                current_line: cue.text.clone(),
                next_lines,
                timing: cue.timing.clone(),
                context_tags: tags.get(i).cloned().unwrap_or_default(),
            }
        })
        .collect()
}

/// Format a contextual cue into a structured prompt string for the MT model.
/// The model sees surrounding lines as context and translates the [CUR] line.
#[allow(dead_code)]
pub fn format_mt_prompt(cue: &ContextualCue) -> String {
    let mut prompt = String::new();
    for line in &cue.prev_lines {
        prompt.push_str("[PREV] ");
        prompt.push_str(line);
        prompt.push('\n');
    }
    prompt.push_str("[CUR] ");
    prompt.push_str(&cue.current_line);
    prompt.push('\n');
    for line in &cue.next_lines {
        prompt.push_str("[NEXT] ");
        prompt.push_str(line);
        prompt.push('\n');
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::srt::SubtitleCue;

    fn make_cues(texts: &[&str]) -> Vec<SubtitleCue> {
        texts
            .iter()
            .enumerate()
            .map(|(i, t)| SubtitleCue {
                index: i + 1,
                timing: format!("00:00:{:02},000 --> 00:00:{:02},000", i * 3, i * 3 + 2),
                text: t.to_string(),
            })
            .collect()
    }

    #[test]
    fn window_radius_zero_gives_no_context() {
        let cues = make_cues(&["AAA", "BBB", "CCC"]);
        let windows = build_context_windows(&cues, 0);
        assert_eq!(windows.len(), 3);
        assert!(windows[1].prev_lines.is_empty());
        assert!(windows[1].next_lines.is_empty());
        assert_eq!(windows[1].current_line, "BBB");
    }

    #[test]
    fn window_radius_one() {
        let cues = make_cues(&["AAA", "BBB", "CCC", "DDD"]);
        let windows = build_context_windows(&cues, 1);
        // Middle cue should have 1 prev + 1 next.
        assert_eq!(windows[1].prev_lines, vec!["AAA"]);
        assert_eq!(windows[1].current_line, "BBB");
        assert_eq!(windows[1].next_lines, vec!["CCC"]);
    }

    #[test]
    fn window_clamps_at_edges() {
        let cues = make_cues(&["AAA", "BBB", "CCC"]);
        let windows = build_context_windows(&cues, 3);
        // First cue: no prev context.
        assert!(windows[0].prev_lines.is_empty());
        assert_eq!(windows[0].next_lines.len(), 2);
        // Last cue: no next context.
        assert_eq!(windows[2].prev_lines.len(), 2);
        assert!(windows[2].next_lines.is_empty());
    }

    #[test]
    fn format_prompt_structure() {
        let cue = ContextualCue {
            index: 2,
            prev_lines: vec!["Line one".to_string()],
            current_line: "Line two".to_string(),
            next_lines: vec!["Line three".to_string()],
            timing: "00:00:03,000 --> 00:00:06,000".to_string(),
            context_tags: vec!["scene_medium".to_string()],
        };
        let prompt = format_mt_prompt(&cue);
        assert!(prompt.contains("[PREV] Line one"));
        assert!(prompt.contains("[CUR] Line two"));
        assert!(prompt.contains("[NEXT] Line three"));
    }

    #[test]
    fn window_tags_follow_input_tags() {
        let cues = make_cues(&["AAA", "BBB"]);
        let tags = vec![
            vec!["scene_easy".to_string()],
            vec!["scene_hard".to_string()],
        ];
        let windows = build_context_windows_with_tags(&cues, 1, &tags);
        assert_eq!(windows[0].context_tags, vec!["scene_easy"]);
        assert_eq!(windows[1].context_tags, vec!["scene_hard"]);
    }
}
