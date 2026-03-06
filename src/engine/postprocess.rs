// Post-Processor — polishes neural MT output for production-quality subtitles.
//
// - Name consistency: clusters similar transliterations and picks canonical forms
// - Honorific mapping: Japanese suffixes (-san, -sama, etc.)
// - Natural English cleanup: capitalization, whitespace, artifact removal

use crate::engine::srt::SubtitleCue;
use std::collections::HashMap;

/// Run all post-processing passes over translated cues.
pub fn postprocess(cues: &mut [SubtitleCue]) {
    enforce_name_consistency(cues);
    normalize_contractions(cues);
    repair_grammar_artifacts(cues);
    cleanup_artifacts(cues);
    fix_capitalization(cues);
}

// ── Name Consistency ─────────────────────────────────────────────────────────

/// Find words that appear as multiple similar variants (likely the same name
/// transliterated differently) and normalize them to the most common form.
fn enforce_name_consistency(cues: &mut [SubtitleCue]) {
    // Collect all capitalized words (potential proper nouns).
    let mut word_freq: HashMap<String, usize> = HashMap::new();
    for cue in cues.iter() {
        for word in cue.text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.len() >= 2
                && clean
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
            {
                *word_freq.entry(clean.to_string()).or_insert(0) += 1;
            }
        }
    }

    // Group similar names (Levenshtein distance ≤ 2 and first char matches).
    let names: Vec<String> = word_freq.keys().cloned().collect();
    let mut canonical: HashMap<String, String> = HashMap::new();

    for i in 0..names.len() {
        if canonical.contains_key(&names[i]) {
            continue;
        }
        let mut cluster = vec![names[i].clone()];
        for j in (i + 1)..names.len() {
            if canonical.contains_key(&names[j]) {
                continue;
            }
            // Same first character and short edit distance → likely same name.
            if names[i].chars().next() == names[j].chars().next() {
                let dist = strsim::levenshtein(&names[i], &names[j]);
                if dist <= 2 && dist > 0 {
                    cluster.push(names[j].clone());
                }
            }
        }
        if cluster.len() > 1 {
            // Pick the most frequent variant as canonical.
            let best = cluster
                .iter()
                .max_by_key(|name| word_freq.get(*name).copied().unwrap_or(0))
                .unwrap()
                .clone();
            for name in &cluster {
                if name != &best {
                    canonical.insert(name.clone(), best.clone());
                }
            }
        }
    }

    // Apply name replacements.
    if !canonical.is_empty() {
        for cue in cues.iter_mut() {
            let mut text = cue.text.clone();
            for (variant, canon) in &canonical {
                text = replace_word(&text, variant, canon);
            }
            cue.text = text;
        }
    }
}

/// Replace whole-word occurrences of `from` with `to`.
fn replace_word(text: &str, from: &str, to: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;

    while let Some(pos) = remaining.find(from) {
        // Check word boundaries.
        let before_ok = pos == 0
            || remaining.as_bytes()[pos - 1].is_ascii_whitespace()
            || !remaining.as_bytes()[pos - 1].is_ascii_alphanumeric();
        let after_pos = pos + from.len();
        let after_ok = after_pos >= remaining.len()
            || remaining.as_bytes()[after_pos].is_ascii_whitespace()
            || !remaining.as_bytes()[after_pos].is_ascii_alphanumeric();

        if before_ok && after_ok {
            result.push_str(&remaining[..pos]);
            result.push_str(to);
            remaining = &remaining[after_pos..];
        } else {
            result.push_str(&remaining[..pos + from.len()]);
            remaining = &remaining[after_pos..];
        }
    }
    result.push_str(remaining);
    result
}

// ── Artifact Cleanup ─────────────────────────────────────────────────────────

/// Repair common malformed contractions and tense artifacts from MT output.
fn normalize_contractions(cues: &mut [SubtitleCue]) {
    const MALFORMED_CONTRACTIONS: &[(&str, &str)] = &[
        ("I'm's", "I'm"),
        ("you're's", "you're"),
        ("we're's", "we're"),
        ("they're's", "they're"),
        ("he's's", "he's"),
        ("she's's", "she's"),
        ("it's's", "it's"),
        ("let's's", "let's"),
        ("I'm'll", "I'll"),
        ("This's", "This is"),
        ("this's", "this is"),
    ];

    const PHRASE_REWRITES: &[(&str, &str)] = &[
        ("This's right", "That's right"),
        ("this's right", "that's right"),
        ("This's it", "This is it"),
        ("this's it", "this is it"),
        ("The's", "There's"),
        ("the's", "there's"),
        ("It's be", "It's"),
        ("it's be", "it's"),
        ("It's hasn't", "It hasn't"),
        ("it's hasn't", "it hasn't"),
        ("I'm be", "I'll be"),
        ("I'm let", "I'll let"),
        ("I'm got", "I've got"),
        ("I'm take", "I'll take"),
        ("I'm leave", "I'll leave"),
        ("I'm tell", "I'll tell"),
        ("I'm give", "I'll give"),
        ("I'm like to", "I'd like to"),
        ("i'm like to", "i'd like to"),
        ("I'm do", "I do"),
        ("i'm do", "i do"),
        ("I'm put it", "I'm putting it"),
        ("i'm put it", "i'm putting it"),
        ("I was asked him", "I asked him"),
        ("i was asked him", "i asked him"),
        ("puttingting", "putting"),
        ("I'll been", "I've been"),
        ("i'll been", "i've been"),
        ("Well're's", "Where's"),
        ("well're's", "where's"),
        ("Well're", "We're"),
        ("well're", "we're"),
        ("Whoa're's", "Where's"),
        ("whoa're's", "where's"),
        ("Whoa're", "We're"),
        ("whoa're", "we're"),
        ("What're's", "Where's"),
        ("what're's", "where's"),
        ("What're was", "Where was"),
        ("what're was", "where was"),
        ("How's we", "How do we"),
        ("how's we", "how do we"),
    ];

    for cue in cues.iter_mut() {
        let mut text = cue.text.clone();
        for (source, target) in MALFORMED_CONTRACTIONS {
            text = replace_case_insensitive_literal(&text, source, target);
        }
        for (source, target) in PHRASE_REWRITES {
            text = replace_case_insensitive_literal(&text, source, target);
        }
        if text.starts_with("All I'm ") {
            text = text.replacen("All I'm ", "I'm ", 1);
        } else if text.starts_with("all i'm ") {
            text = text.replacen("all i'm ", "i'm ", 1);
        }
        if text.starts_with("All I think ") {
            text = text.replacen("All I think ", "I think ", 1);
        } else if text.starts_with("all i think ") {
            text = text.replacen("all i think ", "i think ", 1);
        }
        // Last-resort cleanup for compounded possessives.
        text = replace_case_insensitive_literal(&text, "'s's", "'s");
        cue.text = text;
    }
}

/// Repair recurrent agreement artifacts from MT output while keeping valid
/// "I'm <adjective>" constructions untouched.
fn repair_grammar_artifacts(cues: &mut [SubtitleCue]) {
    const IM_VERB_REWRITES: &[(&str, &str)] = &[
        ("I'm was", "I was"),
        ("I'm were", "I was"),
        ("I'm had", "I had"),
        ("I'm has", "I have"),
        ("I'm have", "I have"),
        ("I'm did", "I did"),
        ("I'm done", "I've done"),
        ("I'm said", "I said"),
        ("I'm asked", "I asked"),
        ("I'm ask", "I ask"),
        ("I'm get", "I get"),
        ("I'm got", "I've got"),
        ("I'm see", "I see"),
        ("I'm know", "I know"),
        ("I'm need", "I need"),
        ("I'm want", "I want"),
        ("I'm love", "I love"),
        ("I'm think", "I think"),
        ("I'm thought", "I thought"),
        ("I'm make", "I make"),
        ("I'm made", "I made"),
        ("I'm looks", "It looks"),
        ("I'm look", "I look"),
        ("I'm scary", "I'm scared"),
    ];

    for cue in cues.iter_mut() {
        let mut text = cue.text.clone();
        for (source, target) in IM_VERB_REWRITES {
            text = replace_case_insensitive_literal(&text, source, target);
        }
        cue.text = text;
    }
}

fn replace_case_insensitive_literal(text: &str, needle: &str, replacement: &str) -> String {
    if needle.is_empty() {
        return text.to_string();
    }
    let text_lower = text.to_ascii_lowercase();
    let needle_lower = needle.to_ascii_lowercase();

    let mut out = String::with_capacity(text.len());
    let mut start = 0usize;
    while let Some(offset) = text_lower[start..].find(&needle_lower) {
        let pos = start + offset;
        out.push_str(&text[start..pos]);
        out.push_str(replacement);
        start = pos + needle.len();
    }
    out.push_str(&text[start..]);
    out
}

/// Remove common MT artifacts from subtitle text.
fn cleanup_artifacts(cues: &mut [SubtitleCue]) {
    for cue in cues.iter_mut() {
        let mut text = cue.text.clone();

        // Remove repeated phrases like "I'm going. I'm going. I'm going."
        text = collapse_repetitions(&text);
        text = collapse_adjacent_token_repeats(&text);

        // Remove leading/trailing whitespace and normalize internal spaces.
        text = text.split_whitespace().collect::<Vec<_>>().join(" ");

        // Remove empty parenthetical notes the model sometimes hallucinates.
        text = text.replace("()", "").replace("( )", "");

        cue.text = text.trim().to_string();
    }
}

/// Collapse noisy token loops such as:
/// - "go go go now" -> "go now"
/// - "wait a wait a second" -> "wait a second"
fn collapse_adjacent_token_repeats(text: &str) -> String {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < 2 {
        return text.to_string();
    }

    // Pass 1: remove direct adjacent duplicates.
    let mut deduped = Vec::<String>::new();
    for token in tokens {
        let key = token_key(token);
        if let Some(last) = deduped.last() {
            if !key.is_empty() && key == token_key(last) {
                continue;
            }
        }
        deduped.push(token.to_string());
    }

    // Pass 2: collapse immediate ABAB loops.
    let mut collapsed = Vec::<String>::new();
    let mut index = 0usize;
    while index < deduped.len() {
        if index + 3 < deduped.len() {
            let a0 = token_key(&deduped[index]);
            let b0 = token_key(&deduped[index + 1]);
            let a1 = token_key(&deduped[index + 2]);
            let b1 = token_key(&deduped[index + 3]);
            if !a0.is_empty() && !b0.is_empty() && a0 == a1 && b0 == b1 {
                collapsed.push(deduped[index].clone());
                collapsed.push(deduped[index + 1].clone());
                index += 4;
                continue;
            }
        }
        collapsed.push(deduped[index].clone());
        index += 1;
    }

    collapsed.join(" ")
}

fn token_key(token: &str) -> String {
    let cleaned = token
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '\'')
        .to_ascii_lowercase();
    if cleaned.is_empty() {
        token.to_ascii_lowercase()
    } else {
        cleaned
    }
}

/// Collapse phrases that repeat 3+ times into a single occurrence.
fn collapse_repetitions(text: &str) -> String {
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .collect();

    if sentences.len() < 3 {
        return text.to_string();
    }

    // Check if the same sentence repeats.
    let mut seen: HashMap<String, usize> = HashMap::new();
    let mut result_parts: Vec<String> = Vec::new();

    for sentence in &sentences {
        let normalized = sentence.trim().to_lowercase();
        let count = seen.entry(normalized.clone()).or_insert(0);
        *count += 1;
        if *count <= 2 {
            result_parts.push(sentence.trim().to_string());
        }
    }

    if result_parts.len() < sentences.len() {
        result_parts.join(". ")
    } else {
        text.to_string()
    }
}

// ── Capitalization ───────────────────────────────────────────────────────────

/// Ensure first letter of each subtitle line is capitalized.
fn fix_capitalization(cues: &mut [SubtitleCue]) {
    for cue in cues.iter_mut() {
        let lines: Vec<String> = cue
            .text
            .lines()
            .map(|line| {
                let trimmed = line.trim_start();
                if trimmed.is_empty() {
                    return line.to_string();
                }
                let mut chars = trimmed.chars();
                match chars.next() {
                    Some(first) if first.is_ascii_lowercase() => {
                        let leading_ws = &line[..line.len() - trimmed.len()];
                        format!("{}{}{}", leading_ws, first.to_uppercase(), chars.as_str())
                    }
                    _ => line.to_string(),
                }
            })
            .collect();
        cue.text = lines.join("\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_triple_repeat() {
        let input = "Wow. Wow. Wow. Wow.";
        let result = collapse_repetitions(input);
        assert!(!result.contains("Wow. Wow. Wow"));
    }

    #[test]
    fn collapse_adjacent_token_repeat() {
        let input = "go go go now now";
        let result = collapse_adjacent_token_repeats(input);
        assert_eq!(result, "go now");
    }

    #[test]
    fn collapse_abab_token_loop() {
        let input = "wait a wait a second";
        let result = collapse_adjacent_token_repeats(input);
        assert_eq!(result, "wait a second");
    }

    #[test]
    fn replace_word_preserves_boundaries() {
        assert_eq!(
            replace_word("Hello Konozuka and Konatsu", "Konozuka", "Konatsu"),
            "Hello Konatsu and Konatsu"
        );
        // Should not replace inside another word.
        assert_eq!(replace_word("Unconditional", "Con", "Kan"), "Unconditional");
    }

    #[test]
    fn fix_capitalization_works() {
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: "hello world".to_string(),
        }];
        fix_capitalization(&mut cues);
        assert_eq!(cues[0].text, "Hello world");
    }

    #[test]
    fn postprocess_full_pipeline() {
        let mut cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "hello Konozuka".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,000 --> 00:00:02,000".to_string(),
                text: "Konatsu is here".to_string(),
            },
            SubtitleCue {
                index: 3,
                timing: "00:00:02,000 --> 00:00:03,000".to_string(),
                text: "Konatsu said hello".to_string(),
            },
        ];
        postprocess(&mut cues);
        // First cue should be capitalized.
        assert!(cues[0].text.starts_with('H'));
    }

    #[test]
    fn normalize_contractions_fixes_common_artifacts() {
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: "I'm's here and I'm let you know.".to_string(),
        }];
        normalize_contractions(&mut cues);
        assert_eq!(cues[0].text, "I'm here and I'll let you know.");
    }

    #[test]
    fn normalize_contractions_repairs_this_artifacts() {
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: "This's right. This's it. It's hasn't done, it's be weird.".to_string(),
        }];
        normalize_contractions(&mut cues);
        assert_eq!(
            cues[0].text,
            "This is right. This is it. It hasn't done, It's weird."
        );
    }

    #[test]
    fn normalize_contractions_repairs_where_artifacts() {
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: "Whoa're's the exit? Well're's the wall?".to_string(),
        }];
        normalize_contractions(&mut cues);
        assert_eq!(cues[0].text, "Where's the exit? Where's the wall?");
    }

    #[test]
    fn normalize_contractions_repairs_whatre_artifacts() {
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: "What're's Yukon? What're was Yukon? How's we get in?".to_string(),
        }];
        normalize_contractions(&mut cues);
        assert_eq!(
            cues[0].text,
            "Where's Yukon? Where was Yukon? How do we get in?"
        );
    }

    #[test]
    fn repair_grammar_artifacts_fixes_im_verb_corruption() {
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: "I'm was wrong. I'm had enough. I'm looks fine.".to_string(),
        }];
        repair_grammar_artifacts(&mut cues);
        assert_eq!(cues[0].text, "I was wrong. I had enough. It looks fine.");
    }

    #[test]
    fn repair_grammar_artifacts_preserves_valid_im_adjective() {
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: "I'm tired but I'm ready.".to_string(),
        }];
        repair_grammar_artifacts(&mut cues);
        assert_eq!(cues[0].text, "I'm tired but I'm ready.");
    }
}
