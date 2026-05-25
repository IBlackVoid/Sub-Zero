use crate::engine::srt::SubtitleCue;
use std::collections::{HashMap, HashSet};

use super::time::parse_srt_timing_line;

#[derive(Debug, Default, Clone)]
pub(super) struct SpeakerInferenceStats {
    pub(super) labeled_cues: usize,
    pub(super) unique_speakers: usize,
}

pub(super) fn infer_speakers(cues: &[SubtitleCue]) -> (Vec<Option<String>>, SpeakerInferenceStats) {
    if cues.is_empty() {
        return (Vec::new(), SpeakerInferenceStats::default());
    }

    let mut raw = Vec::<Option<String>>::with_capacity(cues.len());
    let mut counts = HashMap::<String, usize>::new();

    for cue in cues {
        let speaker = infer_speaker_from_text(&cue.text);
        if let Some(speaker) = speaker.as_ref() {
            *counts.entry(speaker.clone()).or_insert(0) += 1;
        }
        raw.push(speaker);
    }

    // Filter out singletons to reduce false positives on noisy prefixes.
    for slot in raw.iter_mut() {
        let Some(candidate) = slot.as_ref() else {
            continue;
        };
        let seen = counts.get(candidate).copied().unwrap_or(0);
        if seen < 2 {
            *slot = None;
        }
    }

    let labeled_cues = raw.iter().filter(|s| s.is_some()).count();
    let unique_speakers = raw
        .iter()
        .filter_map(|s| s.as_ref())
        .collect::<HashSet<_>>()
        .len();

    (
        raw,
        SpeakerInferenceStats {
            labeled_cues,
            unique_speakers,
        },
    )
}

pub(super) fn build_speaker_tags(speakers: &[Option<String>]) -> Vec<Vec<String>> {
    let mut tags = vec![Vec::<String>::new(); speakers.len()];
    for (idx, speaker) in speakers.iter().enumerate() {
        let Some(speaker) = speaker.as_ref() else {
            continue;
        };
        let normalized = normalize_speaker_tag(speaker);
        if normalized.is_empty() {
            continue;
        }
        // Keep the tag compact and ASCII so it survives all toolchains.
        tags[idx] = vec![format!("speaker={normalized}")];
    }
    tags
}

pub(super) fn build_speaker_tags_with_context(
    cues: &[SubtitleCue],
    speakers: &[Option<String>],
) -> Vec<Vec<String>> {
    let mut tags = build_speaker_tags(speakers);
    if cues.len() != speakers.len() || cues.is_empty() {
        return tags;
    }

    let mut last_labeled = None::<(usize, String, f64)>;
    for (idx, cue) in cues.iter().enumerate() {
        let Some(raw) = speakers[idx].as_ref() else {
            continue;
        };
        let normalized = normalize_speaker_tag(raw);
        if normalized.is_empty() {
            continue;
        }

        let Ok((start, end)) = parse_srt_timing_line(&cue.timing) else {
            continue;
        };

        if let Some((_, prev_speaker, prev_end)) = last_labeled.as_ref() {
            let gap = start - *prev_end;
            let switched = prev_speaker != &normalized;
            if switched && gap.is_finite() && (0.0..0.25).contains(&gap) {
                tags[idx].push("rapid_dialogue".to_string());
            }
            if switched && gap.is_finite() && gap < 0.0 {
                tags[idx].push("overlap_risk".to_string());
            }
        }

        last_labeled = Some((idx, normalized, end));
    }

    tags
}

fn infer_speaker_from_text(text: &str) -> Option<String> {
    let first_line = text.lines().next()?.trim();
    if first_line.is_empty() {
        return None;
    }

    if let Some(label) = strip_wrapped_label(first_line, '[', ']')
        .or_else(|| strip_wrapped_label(first_line, '(', ')'))
        .or_else(|| strip_wrapped_label(first_line, '【', '】'))
        .or_else(|| strip_wrapped_label(first_line, '「', '」'))
    {
        let label = label.trim();
        if is_reasonable_speaker_label(label) {
            return Some(label.to_string());
        }
    }

    if let Some((label, rest)) = first_line
        .split_once(':')
        .or_else(|| first_line.split_once('：'))
    {
        if rest.trim().is_empty() {
            return None;
        }
        let label = label.trim();
        if is_reasonable_speaker_label(label) {
            return Some(label.to_string());
        }
    }

    None
}

fn strip_wrapped_label(text: &str, open: char, close: char) -> Option<&str> {
    let trimmed = text.trim();
    let stripped = trimmed.strip_prefix(open)?;
    let (label, suffix) = stripped.split_once(close)?;
    if !suffix.trim().is_empty() {
        return None;
    }
    Some(label)
}

fn is_reasonable_speaker_label(label: &str) -> bool {
    let label = label.trim();
    if label.len() < 2 || label.len() > 28 {
        return false;
    }
    if label.chars().all(|ch| !ch.is_alphanumeric()) {
        return false;
    }

    // Avoid tagging common non-speaker prefixes.
    let lowered = label.to_ascii_lowercase();
    if matches!(
        lowered.as_str(),
        "sfx" | "bgm" | "music" | "narration" | "crowd" | "all" | "everyone"
    ) {
        return false;
    }

    // Reject labels that look like timing artifacts.
    if lowered.contains("-->") || lowered.contains("00:") || lowered.contains(",") {
        return false;
    }

    true
}

fn normalize_speaker_tag(label: &str) -> String {
    let mut out = String::with_capacity(label.len());
    let mut last_was_sep = false;
    for ch in label.chars() {
        let lowered = ch.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() {
            out.push(lowered);
            last_was_sep = false;
            continue;
        }
        if matches!(lowered, '_' | '-' | ' ') && !last_was_sep && !out.is_empty() {
            out.push('_');
            last_was_sep = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_speaker_from_colon_prefix() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "ALICE: hello".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,000 --> 00:00:02,000".to_string(),
                text: "ALICE: again".to_string(),
            },
            SubtitleCue {
                index: 3,
                timing: "00:00:02,000 --> 00:00:03,000".to_string(),
                text: "BOB: hi".to_string(),
            },
        ];
        let (speakers, stats) = infer_speakers(&cues);
        assert_eq!(stats.unique_speakers, 1);
        assert_eq!(stats.labeled_cues, 2);
        assert_eq!(speakers[0].as_deref(), Some("ALICE"));
        assert_eq!(speakers[1].as_deref(), Some("ALICE"));
        assert_eq!(speakers[2], None);
    }

    #[test]
    fn normalize_speaker_tag_scrubs_punctuation() {
        assert_eq!(normalize_speaker_tag("Dr. Alice"), "dr_alice");
        assert_eq!(normalize_speaker_tag("ALICE-01"), "alice_01");
    }

    #[test]
    fn build_speaker_tags_with_context_marks_rapid_dialogue() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "ALICE: hi".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,050 --> 00:00:02,000".to_string(),
                text: "BOB: yo".to_string(),
            },
        ];
        let speakers = vec![Some("ALICE".to_string()), Some("BOB".to_string())];
        let tags = build_speaker_tags_with_context(&cues, &speakers);
        assert!(tags[1].iter().any(|t| t == "rapid_dialogue"));
    }
}
