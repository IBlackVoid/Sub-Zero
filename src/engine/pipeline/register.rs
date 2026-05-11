use crate::engine::postprocess;
use crate::engine::srt::SubtitleCue;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum SpeakerRegister {
    Formal,
    Casual,
}

#[derive(Debug, Default, Clone)]
pub(super) struct SpeakerRegisterStats {
    pub(super) speakers_observed: usize,
    pub(super) speakers_formal: usize,
    pub(super) cues_labeled: usize,
}

pub(super) fn apply_register_post_edit(cues: &mut [SubtitleCue], tags: &[Vec<String>]) {
    let count = cues.len().min(tags.len());
    for idx in 0..count {
        if !tags[idx].iter().any(|t| t == "register_formal") {
            continue;
        }
        cues[idx].text = postprocess::expand_english_contractions_formal(&cues[idx].text);
    }
}

pub(super) fn infer_speaker_register_tags(
    cues: &[SubtitleCue],
    speakers: &[Option<String>],
) -> (
    Vec<Vec<String>>,
    HashMap<String, SpeakerRegister>,
    SpeakerRegisterStats,
) {
    if cues.is_empty() || speakers.is_empty() || cues.len() != speakers.len() {
        return (
            vec![Vec::new(); cues.len()],
            HashMap::new(),
            SpeakerRegisterStats::default(),
        );
    }

    let mut per_speaker = HashMap::<String, SpeakerCounts>::new();
    let mut cues_labeled = 0usize;

    for (cue, speaker) in cues.iter().zip(speakers.iter()) {
        let Some(speaker) = speaker.as_ref() else {
            continue;
        };
        let key = normalize_speaker_id(speaker);
        if key.is_empty() {
            continue;
        }
        cues_labeled += 1;
        per_speaker.entry(key).or_default().observe_text(&cue.text);
    }

    let mut register = HashMap::<String, SpeakerRegister>::new();
    let mut formal_count = 0usize;
    for (speaker, counts) in &per_speaker {
        let Some(class) = classify_register(counts) else {
            continue;
        };
        if class == SpeakerRegister::Formal {
            formal_count += 1;
        }
        register.insert(speaker.clone(), class);
    }

    let mut tags = vec![Vec::<String>::new(); cues.len()];
    for (idx, speaker) in speakers.iter().enumerate() {
        let Some(speaker) = speaker.as_ref() else {
            continue;
        };
        let key = normalize_speaker_id(speaker);
        let Some(class) = register.get(&key).copied() else {
            continue;
        };
        match class {
            SpeakerRegister::Formal => tags[idx].push("register_formal".to_string()),
            SpeakerRegister::Casual => tags[idx].push("register_casual".to_string()),
        }
    }

    let stats = SpeakerRegisterStats {
        speakers_observed: per_speaker.len(),
        speakers_formal: formal_count,
        cues_labeled,
    };

    (tags, register, stats)
}

#[derive(Debug, Default, Clone, Copy)]
struct SpeakerCounts {
    cues: usize,
    polite_markers: usize,
}

impl SpeakerCounts {
    fn observe_text(&mut self, text: &str) {
        self.cues += 1;

        // Lightweight Japanese register heuristic.
        // Mirrors relationship graph features (Phase E), but stays local so we can
        // make MT/post-edit decisions without reading sidecars.
        if text.contains("です") || text.contains("ます") || text.contains("でございます")
        {
            self.polite_markers += 1;
        }
    }
}

fn classify_register(counts: &SpeakerCounts) -> Option<SpeakerRegister> {
    // Require a minimum signal so we don't random-walk on tiny sample sizes.
    if counts.cues < 4 {
        return None;
    }
    let ratio = counts.polite_markers as f64 / counts.cues.max(1) as f64;
    if ratio >= 0.12 {
        Some(SpeakerRegister::Formal)
    } else {
        Some(SpeakerRegister::Casual)
    }
}

fn normalize_speaker_id(label: &str) -> String {
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
    fn register_formal_tags_when_polite_ratio_high() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "ALICE: そうです".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,000 --> 00:00:02,000".to_string(),
                text: "ALICE: そうです".to_string(),
            },
            SubtitleCue {
                index: 3,
                timing: "00:00:02,000 --> 00:00:03,000".to_string(),
                text: "ALICE: そうです".to_string(),
            },
            SubtitleCue {
                index: 4,
                timing: "00:00:03,000 --> 00:00:04,000".to_string(),
                text: "ALICE: そうです".to_string(),
            },
        ];
        let speakers = vec![
            Some("ALICE".to_string()),
            Some("ALICE".to_string()),
            Some("ALICE".to_string()),
            Some("ALICE".to_string()),
        ];
        let (tags, register, stats) = infer_speaker_register_tags(&cues, &speakers);
        assert_eq!(stats.speakers_observed, 1);
        assert_eq!(stats.speakers_formal, 1);
        assert_eq!(stats.cues_labeled, 4);
        assert!(register.values().any(|v| *v == SpeakerRegister::Formal));
        assert!(tags
            .iter()
            .all(|t| t.iter().any(|x| x == "register_formal")));
    }
}
