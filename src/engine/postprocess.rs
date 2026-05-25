// Post-Processor — polishes neural MT output for production-quality subtitles.
//
// - Name consistency: clusters similar transliterations and picks canonical forms
// - Honorific mapping: Japanese suffixes (-san, -sama, etc.)
// - Natural English cleanup: capitalization, whitespace, artifact removal

use crate::engine::srt::SubtitleCue;
use std::collections::HashMap;

/// Netflix Style Guide reading-rate ceiling for adult content. The
/// guide uses 17 cps for Latin-alphabet targets; subtitles longer
/// than `cue_duration_secs · cps_max` characters fail readability.
/// See `docs/F2_subtitle_information_bottleneck.md` § 2 (term E2)
/// for the formal motivation.
pub const DEFAULT_READING_RATE_CPS: f64 = 17.0;

/// Run all post-processing passes over translated cues. The
/// reading-rate compressor is the final pass so other normalisations
/// (contraction repair, name canonicalisation) settle first; the
/// compressor measures the *final* output length.
pub fn postprocess(cues: &mut [SubtitleCue]) {
    enforce_name_consistency(cues);
    normalize_contractions(cues);
    repair_grammar_artifacts(cues);
    detect_and_label_non_speech(cues);
    cleanup_artifacts(cues);
    fix_capitalization(cues);
    compress_reading_rate(cues, DEFAULT_READING_RATE_CPS);
}

// ── Non-speech detection (whisper hallucination suppression) ─────────────────

/// Detect whisper hallucination patterns and replace with appropriate
/// non-speech event labels like [Music], [Applause], [Laughter].
///
/// Whisper hallucinates specific patterns on non-speech audio:
/// - Repetitive syllables: "be-be-be-be-be", "doodle-doodle-doodle"
/// - Music transcription: "♪", "la la la", "do do do"
/// - Onomatopoeia chains: same word repeated 4+ times
/// - Very short repeated tokens filling a long duration
///
/// This runs BEFORE cleanup_artifacts so the repetition collapser doesn't
/// mangle the detection.
fn detect_and_label_non_speech(cues: &mut [SubtitleCue]) {
    for cue in cues.iter_mut() {
        let text = cue.text.trim();
        if text.is_empty() {
            continue;
        }

        // Check for known non-speech markers already present.
        let lower = text.to_ascii_lowercase();
        if lower.contains("[music]")
            || lower.contains("[applause]")
            || lower.contains("[laughter]")
        {
            continue;
        }

        // Detect repetitive hallucination: if a single token (or hyphenated
        // pattern) repeats 4+ times consecutively, it's almost certainly
        // whisper trying to transcribe non-speech audio.
        if is_repetitive_hallucination(text) {
            // Classify what kind of non-speech it likely is.
            cue.text = classify_non_speech_event(text).to_string();
            continue;
        }

        // Detect music symbols.
        if text.contains('♪') || text.contains('♫') {
            cue.text = "[Music]".to_string();
            continue;
        }

        // Detect asterisk-wrapped sound effects and normalize formatting.
        // e.g. "*laughs*" → "[Laughter]", "*sniff*" → keep as-is (brief)
        if let Some(label) = detect_sound_effect_label(text) {
            cue.text = label;
        }
    }
}

/// Returns true if the text is a repetitive hallucination pattern.
fn is_repetitive_hallucination(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();

    // Catch merged-word hallucinations FIRST (before word-count gate):
    // "DoodledoodleDoodledoodle" where whisper merges the repeated token
    // without spaces. Check if a 3-8 char substring repeats 3+ times.
    let lower = text.to_ascii_lowercase();
    let lower_alpha: String = lower.chars().filter(|c| c.is_alphabetic()).collect();
    if lower_alpha.len() >= 12 {
        for pat_len in 3..=8 {
            if pat_len > lower_alpha.len() / 3 {
                break;
            }
            let pattern = &lower_alpha[..pat_len];
            let repeat_count = lower_alpha.matches(pattern).count();
            if repeat_count >= 4 && repeat_count * pat_len > lower_alpha.len() / 3 {
                return true;
            }
        }
    }

    // Very short text (by word count AND character count) can't be hallucination.
    if words.len() < 4 && text.len() < 30 {
        return false;
    }

    // Check for hyphenated repetition: "be-be-be-be-be"
    if text.contains('-') {
        let parts: Vec<&str> = text.split('-').collect();
        if parts.len() >= 4 {
            let first = parts[0].trim().to_ascii_lowercase();
            if !first.is_empty() {
                let repeat_count = parts
                    .iter()
                    .filter(|p| p.trim().to_ascii_lowercase() == first)
                    .count();
                if repeat_count as f64 / parts.len() as f64 > 0.7 {
                    return true;
                }
            }
        }
    }

    // Check for word-level repetition: same word 4+ times in a row.
    let mut max_consecutive = 1usize;
    let mut current_run = 1usize;
    for window in words.windows(2) {
        let a = window[0].trim_matches(|c: char| !c.is_alphanumeric()).to_ascii_lowercase();
        let b = window[1].trim_matches(|c: char| !c.is_alphanumeric()).to_ascii_lowercase();
        if a == b && !a.is_empty() {
            current_run += 1;
            max_consecutive = max_consecutive.max(current_run);
        } else {
            current_run = 1;
        }
    }
    if max_consecutive >= 4 {
        return true;
    }

    // Check if the entire text is essentially one pattern repeated.
    // "doodle-doodle-doodle-doodle-doodle..." pattern.
    if words.len() >= 6 {
        let unique_words: std::collections::HashSet<String> = words
            .iter()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_ascii_lowercase())
            .filter(|w| !w.is_empty())
            .collect();
        // If 80%+ of words are the same token, it's hallucination.
        if unique_words.len() <= 2 && words.len() >= 6 {
            return true;
        }
    }

    false
}

/// Classify what kind of non-speech event the hallucination represents.
fn classify_non_speech_event(text: &str) -> &'static str {
    let lower = text.to_ascii_lowercase();

    // Music-like patterns.
    if lower.contains("la ")
        || lower.contains("da ")
        || lower.contains("do ")
        || lower.contains("doo")
        || lower.contains("dum")
        || lower.contains("hum")
        || lower.contains("na ")
    {
        return "[Music]";
    }

    // Applause-like.
    if lower.contains("clap") || lower.contains("applau") {
        return "[Applause]";
    }

    // Default: [Music] is the most common non-speech hallucination source.
    "[Music]"
}

/// Detect asterisk-wrapped sound effects and return a normalized label.
fn detect_sound_effect_label(text: &str) -> Option<String> {
    let trimmed = text.trim();
    // Only process lines that are entirely a sound effect.
    if !trimmed.starts_with('*') || !trimmed.ends_with('*') {
        return None;
    }
    let inner = trimmed.trim_matches('*').trim().to_ascii_lowercase();

    // Map common sound effects to standard labels.
    let label = match inner.as_str() {
        "laughs" | "laughter" | "laughing" => "[Laughter]",
        "applause" | "clapping" => "[Applause]",
        "music" | "singing" | "hums" | "humming" => "[Music]",
        "silence" | "quiet" => return Some(String::new()), // Remove silence markers
        "gasps" | "gasp" => "[Gasps]",
        "sighs" | "sigh" => "[Sighs]",
        "sniffs" | "sniff" => "[Sniffs]",
        "coughs" | "cough" => "[Coughs]",
        "screams" | "scream" | "screaming" => "[Screaming]",
        _ => return None, // Keep other effects as-is
    };
    Some(label.to_string())
}

// ── Reading-rate compressor (F.2 term E2) ────────────────────────────────────

/// Compress cue text to fit a reading-rate ceiling of `cps_max`
/// characters per second of display time. Implements term E2 of the
/// F.2 Lagrangian objective.
///
/// Strategy, in order — each step stops as soon as the cue fits:
///
/// 1. **Filler removal**: drop low-information discourse markers
///    (`um`, `uh`, `well`, `you know`, `like`, `actually`, `I mean`).
///    These contribute almost zero translation fidelity but several
///    characters per cue.
/// 2. **Contraction**: expand into high-density English contractions
///    (`I am` → `I'm`, `do not` → `don't`, `going to` → `gonna`).
///    Reduces character count without losing meaning.
/// 3. **Trim trailing punctuation noise**: collapse `…!`, `?!?` etc.
/// 4. **Last-resort ellipsis truncation**: cut to `L_max - 1` chars
///    and append `…`. Only triggered when the prior steps couldn't
///    bring the cue under budget; fidelity loss is logged in the
///    cue text by the trailing ellipsis.
///
/// Cues whose timing cannot be parsed are left unchanged.
pub fn compress_reading_rate(cues: &mut [SubtitleCue], cps_max: f64) {
    if cps_max <= 0.0 {
        return;
    }
    for cue in cues.iter_mut() {
        let Some(duration_secs) = cue_duration_secs(&cue.timing) else {
            continue;
        };
        let l_max = (duration_secs * cps_max).floor() as usize;
        if l_max == 0 {
            continue;
        }
        if cue.text.chars().count() <= l_max {
            continue;
        }
        cue.text = compress_to_budget(&cue.text, l_max);
    }
}

/// Apply the compression ladder. Returns text guaranteed to be at
/// most `l_max` Unicode scalar values long.
fn compress_to_budget(input: &str, l_max: usize) -> String {
    // Step 1 — filler removal.
    let mut text = remove_filler_phrases(input);
    if text.chars().count() <= l_max {
        return text;
    }

    // Step 2 — contraction.
    text = contract_phrases(&text);
    if text.chars().count() <= l_max {
        return text;
    }

    // Step 3 — strip duplicate trailing punctuation.
    text = collapse_terminal_punctuation(&text);
    if text.chars().count() <= l_max {
        return text;
    }

    // Step 4 — last-resort ellipsis truncation. Stop on a word
    // boundary if there is one within 8 chars of the cut.
    truncate_with_ellipsis(&text, l_max)
}

const FILLER_PHRASES: &[&str] = &[
    "you know,",
    "you know ",
    "i mean,",
    "i mean ",
    "well,",
    "well ",
    "actually,",
    "actually ",
    "literally,",
    "literally ",
    "basically,",
    "basically ",
    "uh,",
    "uh ",
    "uhm,",
    "uhm ",
    "um,",
    "um ",
    "kind of ",
    "kinda ",
    "sort of ",
    "sorta ",
    "like,",
    " like ",
];

fn remove_filler_phrases(input: &str) -> String {
    // `to_ascii_lowercase` only maps A-Z → a-z (one byte → one byte) and
    // is a no-op for every other byte, including the trailing bytes of
    // any multi-byte UTF-8 sequence. So `out` and `lowered` share byte
    // indices for the whole life of this function — we can mutate them
    // in lockstep without re-lowercasing after every edit, which the
    // previous implementation did once per match.
    let mut out = input.to_string();
    let mut lowered = out.to_ascii_lowercase();
    debug_assert_eq!(out.len(), lowered.len());

    for filler in FILLER_PHRASES {
        let mut start = 0usize;
        while let Some(pos) = lowered[start..].find(filler) {
            let abs = start + pos;
            let end = abs + filler.len();
            if end > out.len() {
                break;
            }
            out.replace_range(abs..end, "");
            lowered.replace_range(abs..end, "");
            start = abs;
        }
    }

    // Collapse runs of ASCII spaces to a single space in one O(n) pass,
    // instead of the previous `while contains("  ") { replace }` which
    // re-scans the whole string every iteration.
    let bytes = out.as_bytes();
    let mut collapsed = Vec::<u8>::with_capacity(bytes.len());
    let mut prev_space = false;
    for &b in bytes {
        let is_space = b == b' ';
        if is_space && prev_space {
            continue;
        }
        collapsed.push(b);
        prev_space = is_space;
    }
    // Safe: we only filtered b' ' bytes, which never appear inside a
    // multi-byte UTF-8 continuation. The remaining sequence is the same
    // UTF-8 as `out` minus single ASCII spaces.
    let s = String::from_utf8(collapsed).unwrap_or(out);
    s.trim().to_string()
}

const CONTRACTION_PAIRS: &[(&str, &str)] = &[
    (" I am ", " I'm "),
    (" you are ", " you're "),
    (" we are ", " we're "),
    (" they are ", " they're "),
    (" it is ", " it's "),
    (" he is ", " he's "),
    (" she is ", " she's "),
    (" do not ", " don't "),
    (" does not ", " doesn't "),
    (" did not ", " didn't "),
    (" will not ", " won't "),
    (" cannot ", " can't "),
    (" would not ", " wouldn't "),
    (" should not ", " shouldn't "),
    (" could not ", " couldn't "),
    (" has not ", " hasn't "),
    (" have not ", " haven't "),
    (" had not ", " hadn't "),
    (" is not ", " isn't "),
    (" are not ", " aren't "),
    (" was not ", " wasn't "),
    (" were not ", " weren't "),
    (" I will ", " I'll "),
    (" you will ", " you'll "),
    (" we will ", " we'll "),
    (" they will ", " they'll "),
    (" going to ", " gonna "),
    (" want to ", " wanna "),
    (" got to ", " gotta "),
    (" out of ", " outta "),
    (" let me ", " lemme "),
];

fn contract_phrases(input: &str) -> String {
    let padded = format!(" {input} ");
    let mut out = padded.clone();
    let lowered = padded.to_ascii_lowercase();
    for (from, to) in CONTRACTION_PAIRS {
        let mut start = 0usize;
        loop {
            let Some(pos) = lowered[start..].find(from) else {
                break;
            };
            let abs = start + pos;
            if abs + from.len() <= out.len() {
                out.replace_range(abs..(abs + from.len()), to);
                // The replacement string is shorter, so subsequent
                // searches against the original `lowered` would slip
                // out of sync; reset `start` past the replaced span.
                start = abs + to.len();
                if start >= out.len() {
                    break;
                }
            } else {
                break;
            }
        }
    }
    out.trim().to_string()
}

fn collapse_terminal_punctuation(input: &str) -> String {
    // Collapse runs of `!` or `?` (length ≥ 3) at the end of the cue
    // down to a single terminal mark. Leaves `...` intact.
    let trimmed = input.trim_end();
    let last = trimmed.chars().last().unwrap_or(' ');
    if last != '!' && last != '?' {
        return input.to_string();
    }
    let mut chars: Vec<char> = trimmed.chars().collect();
    let mut run = 0usize;
    while run < chars.len() && chars[chars.len() - 1 - run] == last {
        run += 1;
    }
    if run < 3 {
        return input.to_string();
    }
    chars.truncate(chars.len() - run + 1);
    chars.into_iter().collect()
}

fn truncate_with_ellipsis(input: &str, l_max: usize) -> String {
    if l_max <= 1 {
        return "…".to_string();
    }
    // Reserve one slot for the ellipsis itself.
    let target = l_max - 1;
    let mut last_space_byte: Option<usize> = None;
    let mut cut_byte: usize = 0;
    for (count, (byte_idx, ch)) in input.char_indices().enumerate() {
        if count == target {
            cut_byte = byte_idx;
            break;
        }
        if ch.is_whitespace() {
            last_space_byte = Some(byte_idx);
        }
        cut_byte = byte_idx + ch.len_utf8();
    }
    // Prefer cutting at the last word boundary if it's within 8
    // characters of the hard cut — produces "I don't…" instead of
    // "I do…" mid-word.
    if let Some(space) = last_space_byte {
        if cut_byte.saturating_sub(space) <= 8 {
            cut_byte = space;
        }
    }
    let mut out = input[..cut_byte].trim_end().to_string();
    out.push('…');
    out
}

/// Parse `"00:00:01,500 --> 00:00:04,000"` into duration in seconds.
/// Tolerates both `,` and `.` as the millisecond separator.
fn cue_duration_secs(timing: &str) -> Option<f64> {
    let arrow = timing.find("-->")?;
    let start_raw = timing[..arrow].trim();
    let end_raw = timing[arrow + 3..].trim();
    let start_secs = parse_timestamp(start_raw)?;
    let end_secs = parse_timestamp(end_raw)?;
    let dur = end_secs - start_secs;
    if dur > 0.0 {
        Some(dur)
    } else {
        None
    }
}

fn parse_timestamp(s: &str) -> Option<f64> {
    // Accepts HH:MM:SS,mmm or HH:MM:SS.mmm.
    let s = s.replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let sm: f64 = parts[2].parse().ok()?;
    Some(h * 3600.0 + m * 60.0 + sm)
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
            let Some(best) = cluster
                .iter()
                .max_by_key(|name| word_freq.get(*name).copied().unwrap_or(0))
                .cloned()
            else {
                continue;
            };
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
/// `I'm <adjective>` constructions untouched.
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

/// Expand a conservative set of English contractions for a more formal register.
///
/// This is intentionally limited to low-ambiguity expansions (e.g. "can't" -> "cannot").
pub fn expand_english_contractions_formal(text: &str) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }

    let mut out = String::with_capacity(text.len() + 8);
    let bytes = text.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        let b = bytes[i];
        if is_formal_word_byte(b) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_formal_word_byte(bytes[i]) {
                i += 1;
            }
            let token = &text[start..i];
            if let Some(repl) = expand_formal_token(token) {
                out.push_str(&repl);
            } else {
                out.push_str(token);
            }
        } else {
            out.push(b as char);
            i += 1;
        }
    }

    out
}

fn is_formal_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'\''
}

fn expand_formal_token(token: &str) -> Option<String> {
    let lower = token.to_ascii_lowercase();
    let expanded = match lower.as_str() {
        "can't" => "cannot",
        "won't" => "will not",
        "don't" => "do not",
        "doesn't" => "does not",
        "didn't" => "did not",
        "isn't" => "is not",
        "aren't" => "are not",
        "wasn't" => "was not",
        "weren't" => "were not",
        "haven't" => "have not",
        "hasn't" => "has not",
        "hadn't" => "had not",
        "i'm" => "i am",
        "you're" => "you are",
        "we're" => "we are",
        "they're" => "they are",
        "i'll" => "i will",
        "you'll" => "you will",
        "we'll" => "we will",
        "they'll" => "they will",
        "i've" => "i have",
        "you've" => "you have",
        "we've" => "we have",
        "they've" => "they have",
        _ => return None,
    };

    let mut out = expanded.to_string();
    if is_all_caps_token(token) {
        out = out.to_ascii_uppercase();
        return Some(out);
    }
    if token
        .bytes()
        .next()
        .is_some_and(|b| (b as char).is_ascii_uppercase())
    {
        if let Some(first) = out.chars().next() {
            let rest: String = out.chars().skip(1).collect();
            out = first.to_ascii_uppercase().to_string() + &rest;
        }
    }

    Some(out)
}

fn is_all_caps_token(token: &str) -> bool {
    let mut saw_letter = false;
    for ch in token.chars() {
        if ch == '\'' {
            continue;
        }
        if !ch.is_ascii_alphabetic() {
            continue;
        }
        saw_letter = true;
        if !ch.is_ascii_uppercase() {
            return false;
        }
    }
    saw_letter
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
    fn expand_english_contractions_formal_is_boundary_aware() {
        let input = "I'm here. don't go. CAN'T. rock'n'roll.";
        let out = expand_english_contractions_formal(input);
        assert!(out.contains("I am here."));
        assert!(out.contains("do not go."));
        assert!(out.contains("CANNOT."));
        // Don't touch non-matching apostrophe words.
        assert!(out.contains("rock'n'roll"));
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

    // ── Reading-rate compressor (F.2 term E2) ────────────────────────────

    #[test]
    fn cue_duration_secs_parses_standard_srt_timing() {
        let dur = super::cue_duration_secs("00:00:01,500 --> 00:00:04,000");
        assert!((dur.unwrap() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn cue_duration_secs_tolerates_dot_separator() {
        let dur = super::cue_duration_secs("00:00:00.500 --> 00:00:02.250");
        assert!((dur.unwrap() - 1.75).abs() < 1e-6);
    }

    #[test]
    fn rate_compressor_leaves_short_cues_untouched() {
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:05,000".to_string(),
            text: "Short line.".to_string(),
        }];
        super::compress_reading_rate(&mut cues, 17.0);
        assert_eq!(cues[0].text, "Short line.");
    }

    #[test]
    fn rate_compressor_drops_filler_when_over_budget() {
        // 1.0s @ 17 cps = 17 char budget. Original is 35 chars.
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: "Well, you know, I'm tired.".to_string(),
        }];
        super::compress_reading_rate(&mut cues, 17.0);
        assert!(
            cues[0].text.chars().count() <= 17,
            "expected ≤ 17 chars, got {:?}",
            cues[0].text
        );
        // Should retain core meaning even after stripping fillers.
        assert!(
            cues[0].text.to_lowercase().contains("tired") || cues[0].text.contains("…"),
            "expected core meaning, got {:?}",
            cues[0].text
        );
    }

    #[test]
    fn rate_compressor_uses_contractions_before_truncation() {
        // 2.0s @ 17 cps = 34 char budget. Source uses formal forms.
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:02,000".to_string(),
            text: "I am going to do not know what to do.".to_string(),
        }];
        super::compress_reading_rate(&mut cues, 17.0);
        // The contraction pass should bring it under budget without
        // truncation (no trailing ellipsis).
        assert!(
            !cues[0].text.ends_with('…'),
            "should not truncate, got {:?}",
            cues[0].text
        );
        assert!(
            cues[0].text.chars().count() <= 34,
            "expected ≤ 34 chars, got {:?}",
            cues[0].text
        );
    }

    #[test]
    fn rate_compressor_truncates_with_word_boundary_ellipsis() {
        // 0.5s @ 17 cps = 8 char budget. No fillers / contractions help.
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:00,500".to_string(),
            text: "supercalifragilisticexpialidocious".to_string(),
        }];
        super::compress_reading_rate(&mut cues, 17.0);
        assert!(
            cues[0].text.ends_with('…'),
            "expected ellipsis, got {:?}",
            cues[0].text
        );
        assert!(
            cues[0].text.chars().count() <= 8,
            "expected ≤ 8 chars, got {:?}",
            cues[0].text
        );
    }

    #[test]
    fn rate_compressor_skips_unparseable_timing() {
        let mut cues = vec![SubtitleCue {
            index: 1,
            timing: "(unknown timing)".to_string(),
            text: "Some really very extremely long line of dialogue here.".to_string(),
        }];
        super::compress_reading_rate(&mut cues, 17.0);
        // No timing → no compression, regardless of length.
        assert_eq!(
            cues[0].text,
            "Some really very extremely long line of dialogue here."
        );
    }
}
