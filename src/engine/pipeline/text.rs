use crate::engine::srt::SubtitleCue;
use std::collections::HashMap;

pub(super) fn assess_name_inconsistency(cues: &[SubtitleCue]) -> f64 {
    let mut names = Vec::<String>::new();
    for cue in cues {
        names.extend(extract_titlecase_tokens(&cue.text));
    }
    if names.len() < 4 {
        return 0.0;
    }

    let mut freq = HashMap::<String, usize>::new();
    for name in &names {
        *freq.entry(name.to_string()).or_insert(0) += 1;
    }

    let mut inconsistent_mentions = 0usize;
    let total_mentions = names.len();
    let keys: Vec<String> = freq.keys().cloned().collect();
    for i in 0..keys.len() {
        for j in (i + 1)..keys.len() {
            let a = &keys[i];
            let b = &keys[j];
            if a.chars().next() != b.chars().next() {
                continue;
            }
            if (a.len() as isize - b.len() as isize).abs() > 2 {
                continue;
            }
            let similarity =
                strsim::normalized_levenshtein(&a.to_ascii_lowercase(), &b.to_ascii_lowercase());
            if (0.78..1.0).contains(&similarity) {
                let count_a = freq.get(a).copied().unwrap_or(0);
                let count_b = freq.get(b).copied().unwrap_or(0);
                inconsistent_mentions += count_a.min(count_b);
            }
        }
    }
    (inconsistent_mentions as f64 / total_mentions as f64).clamp(0.0, 1.0)
}

pub(super) fn extract_titlecase_tokens(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_ascii_alphabetic())
        .filter(|token| token.len() >= 3)
        .filter(|token| {
            token
                .chars()
                .next()
                .map(|ch| ch.is_ascii_uppercase())
                .unwrap_or(false)
                && token.chars().skip(1).all(|ch| ch.is_ascii_lowercase())
        })
        .map(|token| token.to_string())
        .collect()
}

pub(super) fn tokenize_ascii_words(text: &str) -> Vec<String> {
    let mut tokens = Vec::<String>::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || ch == '\'' {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

pub(super) fn token_has_double_apostrophe(tokens: &[String]) -> bool {
    tokens.iter().any(|token| token.matches('\'').count() >= 2)
}

pub(super) fn cue_has_malformed_contraction(text: &str) -> bool {
    let lowered = text.to_ascii_lowercase();
    [
        "i'm's",
        "you're's",
        "we're's",
        "they're's",
        "he's's",
        "she's's",
        "it's's",
        "let's's",
        "i'm be",
        "i'm let",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

pub(super) fn cue_has_low_function_word_coverage(tokens: &[String]) -> bool {
    if tokens.len() < 6 {
        return false;
    }

    const FUNCTION_WORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "for", "with", "is",
        "are", "was", "were", "be", "been", "being", "i", "you", "he", "she", "it", "we", "they",
        "me", "my", "your", "our", "their", "this", "that", "these", "those", "not", "do", "did",
        "does", "have", "has", "had", "can", "could", "will", "would", "should", "as", "at",
        "from",
    ];

    let function_words = tokens
        .iter()
        .map(|token| token.to_ascii_lowercase())
        .filter(|token| FUNCTION_WORDS.contains(&token.as_str()))
        .count();
    let ratio = (function_words as f64) / (tokens.len() as f64);
    ratio < 0.10
}

pub(super) fn cue_has_adjacent_repeat(tokens: &[String]) -> bool {
    if tokens.len() < 4 {
        return false;
    }
    let lowered: Vec<String> = tokens
        .iter()
        .map(|token| token.to_ascii_lowercase())
        .collect();

    // Repeated token: "no no no"
    for pair in lowered.windows(2) {
        if pair[0] == pair[1] {
            return true;
        }
    }

    // Repeated bigram: "wait a wait a"
    for i in 0..(lowered.len().saturating_sub(3)) {
        if lowered[i..i + 2] == lowered[i + 2..i + 4] {
            return true;
        }
    }
    false
}
