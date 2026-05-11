use crate::engine::srt::SubtitleCue;
use std::collections::HashMap;

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct DiscourseConsistencyStats {
    pub(super) source_clusters: usize,
    pub(super) rewritten_cues: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct SpeakerConsistencyStats {
    pub(super) speakers: usize,
    pub(super) source_clusters: usize,
    pub(super) rewritten_cues: usize,
}

pub(super) fn apply_source_phrase_consistency(
    source_cues: &[SubtitleCue],
    translated_cues: &mut [SubtitleCue],
) -> DiscourseConsistencyStats {
    let mut source_to_target = HashMap::<String, HashMap<String, usize>>::new();
    let mut source_target_display = HashMap::<String, HashMap<String, String>>::new();
    for (source, target) in source_cues.iter().zip(translated_cues.iter()) {
        let source_key = normalize_source_phrase_for_consistency(&source.text);
        if source_key.len() < 2 || source_key.len() > 64 {
            continue;
        }
        let target_display = normalize_display_phrase(&target.text);
        let target_key = normalize_target_phrase_for_consistency(&target.text);
        if target_key.is_empty() || target_key.split_whitespace().count() > 10 {
            continue;
        }
        *source_to_target
            .entry(source_key.clone())
            .or_default()
            .entry(target_key.clone())
            .or_insert(0) += 1;
        source_target_display
            .entry(source_key)
            .or_default()
            .entry(target_key)
            .or_insert(target_display);
    }

    let mut canonical = HashMap::<String, String>::new();
    let mut stats = DiscourseConsistencyStats::default();
    for (source_key, variants) in source_to_target {
        let total = variants.values().sum::<usize>();
        if total < 3 || variants.len() < 2 {
            continue;
        }
        let Some((best_key, best_count)) = variants.iter().max_by_key(|(_, count)| **count) else {
            continue;
        };
        if (*best_count as f64) / (total as f64) < 0.50 {
            continue;
        }
        let display = source_target_display
            .get(&source_key)
            .and_then(|variant_map| variant_map.get(best_key))
            .cloned()
            .unwrap_or_else(|| best_key.clone());
        canonical.insert(source_key, display);
        stats.source_clusters += 1;
    }

    if canonical.is_empty() {
        return stats;
    }

    for (index, source) in source_cues.iter().enumerate() {
        let source_key = normalize_source_phrase_for_consistency(&source.text);
        let Some(canonical_display) = canonical.get(&source_key) else {
            continue;
        };
        let canonical_key = normalize_target_phrase_for_consistency(canonical_display);
        let current_key = normalize_target_phrase_for_consistency(&translated_cues[index].text);
        if current_key.is_empty() || current_key == canonical_key {
            continue;
        }
        if !should_replace_consistency_variant(&current_key, &canonical_key) {
            continue;
        }
        let rewritten =
            rewrite_with_canonical_phrase(&translated_cues[index].text, canonical_display);
        if normalize_target_phrase_for_consistency(&rewritten) == current_key {
            continue;
        }
        translated_cues[index].text = rewritten;
        stats.rewritten_cues += 1;
    }

    stats
}

pub(super) fn apply_source_phrase_consistency_by_speaker(
    source_cues: &[SubtitleCue],
    translated_cues: &mut [SubtitleCue],
    speakers: &[Option<String>],
) -> SpeakerConsistencyStats {
    if speakers.len() != source_cues.len() || speakers.len() != translated_cues.len() {
        return SpeakerConsistencyStats::default();
    }

    let mut speaker_source_to_target =
        HashMap::<String, HashMap<String, HashMap<String, usize>>>::new();
    let mut speaker_source_target_display =
        HashMap::<String, HashMap<String, HashMap<String, String>>>::new();

    for ((speaker, source), target) in speakers
        .iter()
        .zip(source_cues.iter())
        .zip(translated_cues.iter())
    {
        let Some(speaker) = speaker.as_ref() else {
            continue;
        };
        let source_key = normalize_source_phrase_for_consistency(&source.text);
        if source_key.len() < 2 || source_key.len() > 64 {
            continue;
        }
        let target_display = normalize_display_phrase(&target.text);
        let target_key = normalize_target_phrase_for_consistency(&target.text);
        if target_key.is_empty() || target_key.split_whitespace().count() > 10 {
            continue;
        }

        *speaker_source_to_target
            .entry(speaker.clone())
            .or_default()
            .entry(source_key.clone())
            .or_default()
            .entry(target_key.clone())
            .or_insert(0) += 1;

        speaker_source_target_display
            .entry(speaker.clone())
            .or_default()
            .entry(source_key)
            .or_default()
            .entry(target_key)
            .or_insert(target_display);
    }

    if speaker_source_to_target.is_empty() {
        return SpeakerConsistencyStats::default();
    }

    let mut canonical = HashMap::<(String, String), String>::new();
    let mut stats = SpeakerConsistencyStats {
        speakers: speaker_source_to_target.len(),
        ..Default::default()
    };

    for (speaker, per_source) in speaker_source_to_target {
        for (source_key, variants) in per_source {
            let total = variants.values().sum::<usize>();
            if total < 3 || variants.len() < 2 {
                continue;
            }
            let Some((best_key, best_count)) = variants.iter().max_by_key(|(_, count)| **count)
            else {
                continue;
            };
            if (*best_count as f64) / (total as f64) < 0.50 {
                continue;
            }
            let display = speaker_source_target_display
                .get(&speaker)
                .and_then(|source_map| source_map.get(&source_key))
                .and_then(|variant_map| variant_map.get(best_key))
                .cloned()
                .unwrap_or_else(|| best_key.clone());
            canonical.insert((speaker.clone(), source_key), display);
            stats.source_clusters += 1;
        }
    }

    if canonical.is_empty() {
        return stats;
    }

    for (index, source) in source_cues.iter().enumerate() {
        let Some(speaker) = speakers[index].as_ref() else {
            continue;
        };
        let source_key = normalize_source_phrase_for_consistency(&source.text);
        let Some(canonical_display) = canonical.get(&(speaker.clone(), source_key)) else {
            continue;
        };
        let canonical_key = normalize_target_phrase_for_consistency(canonical_display);
        let current_key = normalize_target_phrase_for_consistency(&translated_cues[index].text);
        if current_key.is_empty() || current_key == canonical_key {
            continue;
        }
        if !should_replace_consistency_variant(&current_key, &canonical_key) {
            continue;
        }
        let rewritten =
            rewrite_with_canonical_phrase(&translated_cues[index].text, canonical_display);
        if normalize_target_phrase_for_consistency(&rewritten) == current_key {
            continue;
        }
        translated_cues[index].text = rewritten;
        stats.rewritten_cues += 1;
    }

    stats
}

fn normalize_display_phrase(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_source_phrase_for_consistency(text: &str) -> String {
    text.chars()
        .filter(|ch| {
            ch.is_ascii_alphanumeric()
                || ('\u{3040}'..='\u{30FF}').contains(ch)
                || ('\u{4E00}'..='\u{9FFF}').contains(ch)
        })
        .collect::<String>()
}

fn normalize_target_phrase_for_consistency(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() || ch == '\'' {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn should_replace_consistency_variant(current: &str, canonical: &str) -> bool {
    if current.is_empty() || canonical.is_empty() {
        return false;
    }
    let similarity = strsim::normalized_levenshtein(current, canonical);
    if similarity >= 0.60 {
        return true;
    }

    let current_words = current.split_whitespace().count();
    let canonical_words = canonical.split_whitespace().count();
    current_words <= 4
        && canonical_words <= 4
        && (current.starts_with(canonical) || canonical.starts_with(current))
}

fn rewrite_with_canonical_phrase(original: &str, canonical: &str) -> String {
    let mut rewritten = canonical.to_string();
    if original
        .trim_start()
        .chars()
        .next()
        .map(|ch| ch.is_ascii_uppercase())
        .unwrap_or(false)
    {
        rewritten = capitalize_ascii_first(&rewritten);
    }
    let punctuation_suffix = original
        .trim_end()
        .chars()
        .rev()
        .take_while(|ch| matches!(ch, '.' | '!' | '?'))
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    if !punctuation_suffix.is_empty() && !rewritten.ends_with(&punctuation_suffix) {
        rewritten.push_str(&punctuation_suffix);
    }
    rewritten
}

fn capitalize_ascii_first(text: &str) -> String {
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    if !first.is_ascii_lowercase() {
        return text.to_string();
    }
    format!("{}{}", first.to_ascii_uppercase(), chars.as_str())
}
