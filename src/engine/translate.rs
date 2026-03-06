// Translation engine — dual-mode: neural MT (NLLB-200) with phrase-table fallback.
//
// The neural path uses CTranslate2 via Python subprocess for GOD-TIER accuracy
// across 200+ languages.  When the neural backend is unavailable, falls back
// to the original deterministic phrase table (JA→EN only).

use crate::engine::neural_mt::{
    neural_mt_available, neural_mt_cuda_device_count, to_nllb_lang, translate_cues_neural,
    translate_cues_neural_with_tags, NeuralMTConfig,
};
use crate::engine::postprocess;
use crate::engine::srt::SubtitleCue;
use crate::engine::transcribe::QualityProfile;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslatorBackend {
    /// CTranslate2 + NLLB-200 (GOD-TIER, 200+ languages).
    Neural,
    /// Hardcoded phrase table (offline fallback, JA→EN only).
    PhraseTable,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Translator {
    source_lang: String,
    target_lang: String,
    offline: bool,
    quality_profile: QualityProfile,
    backend: TranslatorBackend,
    neural_config: Option<NeuralMTConfig>,
}

#[derive(Debug, Clone)]
pub struct TranslatorConfig {
    pub source_lang: String,
    pub target_lang: String,
    pub offline: bool,
    pub force_phrase_table: bool,
    pub gpu: bool,
    pub require_gpu: bool,
    pub mt_model: Option<String>,
    pub mt_batch_size: Option<usize>,
    pub mt_max_batch_tokens: Option<usize>,
    pub mt_oom_retries: Option<usize>,
    pub mt_allow_cpu_fallback: bool,
    pub quality_profile: QualityProfile,
}

impl Translator {
    pub fn new(config: TranslatorConfig) -> Result<Self, String> {
        let source_lang = config.source_lang;
        let target_lang = config.target_lang;
        let offline = config.offline;
        let force_phrase_table = config.force_phrase_table;
        let gpu = config.gpu;
        let require_gpu = config.require_gpu;
        let mt_model = config.mt_model;
        let mt_batch_size = config.mt_batch_size;
        let mt_max_batch_tokens = config.mt_max_batch_tokens;
        let mt_oom_retries = config.mt_oom_retries;
        let mt_allow_cpu_fallback = config.mt_allow_cpu_fallback;
        let quality_profile = config.quality_profile;
        let script_path = resolve_translate_script_path();
        let can_use_neural =
            !force_phrase_table && script_path.as_deref().is_some_and(neural_mt_available);

        // Try neural backend first unless explicitly forced to phrase table.
        let (backend, neural_config) = if can_use_neural {
            let script_path = script_path.expect("checked by can_use_neural");
            let mt_device = resolve_mt_device(gpu, require_gpu)?;
            let decode = decode_profile(quality_profile);
            let batch_size = mt_batch_size.unwrap_or(decode.batch_size);
            let max_batch_tokens = mt_max_batch_tokens.unwrap_or(decode.max_batch_tokens);
            let oom_retries = mt_oom_retries.unwrap_or(decode.oom_retries);
            let allow_cpu_fallback_on_oom = !require_gpu && mt_allow_cpu_fallback;
            let config = NeuralMTConfig {
                script_path,
                model_name: mt_model
                    .unwrap_or_else(|| default_mt_model_for_profile(quality_profile).to_string()),
                model_dir: None,
                source_lang: to_nllb_lang(&source_lang),
                target_lang: to_nllb_lang(&target_lang),
                gpu: mt_device == MTDevice::Cuda,
                context_radius: decode.context_radius,
                batch_size,
                max_batch_tokens,
                oom_retries,
                allow_cpu_fallback_on_oom,
                beam_size: decode.beam_size,
                repetition_penalty: decode.repetition_penalty,
                no_repeat_ngram_size: decode.no_repeat_ngram_size,
                prepend_prev_context: decode.prepend_prev_context,
            };
            eprintln!(
                "translator: using neural MT (NLLB-200) [{}→{}] [device={}] [profile={}] [batch={}] [max_batch_tokens={}] [oom_retries={}] [cpu_fallback={}]",
                config.source_lang,
                config.target_lang,
                if config.gpu { "cuda" } else { "cpu" },
                quality_profile.as_str(),
                config.batch_size,
                config.max_batch_tokens,
                config.oom_retries,
                if config.allow_cpu_fallback_on_oom {
                    "on"
                } else {
                    "off"
                },
            );
            (TranslatorBackend::Neural, Some(config))
        } else {
            if gpu && require_gpu && !force_phrase_table {
                return Err(
                    "GPU was required but neural MT backend is unavailable (missing script/dependencies)."
                        .to_string(),
                );
            }
            if !force_phrase_table && script_path.is_none() {
                eprintln!(
                    "warning: scripts/translate_batch.py not found; falling back to phrase-table translator."
                );
            }
            if !target_lang.eq_ignore_ascii_case("en") && offline {
                eprintln!(
                    "warning: phrase-table fallback only supports JA→EN; \
                    install ctranslate2+sentencepiece for 200+ languages."
                );
            }
            eprintln!("translator: using phrase-table fallback [JA→EN only]");
            (TranslatorBackend::PhraseTable, None)
        };

        Ok(Self {
            source_lang: source_lang.to_lowercase(),
            target_lang: target_lang.to_lowercase(),
            offline,
            quality_profile,
            backend,
            neural_config,
        })
    }

    #[allow(dead_code)]
    pub fn backend(&self) -> TranslatorBackend {
        self.backend
    }

    /// Translate a single cue (phrase-table mode only).
    pub fn translate(&self, text: &str) -> String {
        if text.trim().is_empty() {
            return text.to_string();
        }
        if self.backend == TranslatorBackend::Neural {
            // For single-cue calls in neural mode, just passthrough —
            // batch translation is done via translate_all().
            return text.to_string();
        }
        if !self.target_lang.eq_ignore_ascii_case("en") {
            return text.to_string();
        }
        if !self.should_translate(text) {
            return text.to_string();
        }
        translate_ja_to_en(text)
    }

    /// Translate all cues using the active backend.
    /// Neural mode: batched context-aware translation + post-processing.
    /// Phrase-table mode: per-cue string replacement.
    pub fn translate_all(&self, cues: &[SubtitleCue]) -> Result<Vec<SubtitleCue>, String> {
        match self.backend {
            TranslatorBackend::Neural => {
                let config = self
                    .neural_config
                    .as_ref()
                    .expect("neural config must exist when backend is Neural");
                self.translate_neural_with_emergency_ladder(cues, config)
            }
            TranslatorBackend::PhraseTable => {
                let translated: Vec<SubtitleCue> = cues
                    .iter()
                    .map(|cue| {
                        let text = self.translate(&cue.text);
                        SubtitleCue {
                            index: cue.index,
                            timing: cue.timing.clone(),
                            text,
                        }
                    })
                    .collect();
                Ok(translated)
            }
        }
    }

    fn translate_neural_with_emergency_ladder(
        &self,
        cues: &[SubtitleCue],
        base: &NeuralMTConfig,
    ) -> Result<Vec<SubtitleCue>, String> {
        let plans = build_neural_emergency_plans(base, self.quality_profile);
        let min_quality = semantic_floor(self.quality_profile);
        let mut last_error = String::new();
        let cue_tags = if source_is_japanese(&self.source_lang) {
            build_adaptive_context_tags(cues, self.quality_profile)
        } else {
            Vec::new()
        };

        for (index, plan) in plans.iter().enumerate() {
            let stage = index + 1;
            eprintln!(
                "ibvoid-doom-qlock: mt stage {}/{} model={} device={} batch={} tokens={} beam={}",
                stage,
                plans.len(),
                plan.model_name,
                if plan.gpu { "cuda" } else { "cpu" },
                plan.batch_size,
                plan.max_batch_tokens,
                plan.beam_size
            );

            let mut translated = match if cue_tags.is_empty() {
                translate_cues_neural(cues, plan)
            } else {
                translate_cues_neural_with_tags(cues, &cue_tags, plan)
            } {
                Ok(translated) => translated,
                Err(error) => {
                    last_error = error.clone();
                    if is_mt_memory_error(&error) && stage < plans.len() {
                        eprintln!(
                            "warning: ibvoid-doom-qlock mt_replan reason=oom stage={stage} err={}",
                            squash_whitespace(&error)
                        );
                        continue;
                    }
                    if stage < plans.len() {
                        eprintln!(
                            "warning: ibvoid-doom-qlock mt_replan reason=backend stage={stage} err={}",
                            squash_whitespace(&error)
                        );
                        continue;
                    }
                    return Err(error);
                }
            };

            postprocess::postprocess(&mut translated);
            if source_is_japanese(&self.source_lang) && self.target_lang.eq_ignore_ascii_case("en")
            {
                apply_source_aware_short_cue_repairs(cues, &mut translated);
            }

            let quality = assess_neural_translation_quality(&translated, &self.target_lang);
            if quality < min_quality {
                let deficit = min_quality - quality;
                if self.quality_profile == QualityProfile::Strict && deficit <= 0.015 {
                    eprintln!(
                        "warning: ibvoid-doom-qlock strict borderline quality accepted score={:.3} floor={:.3} deficit={:.3}",
                        quality, min_quality, deficit
                    );
                    return Ok(translated);
                }
                if stage < plans.len() {
                    eprintln!(
                        "warning: ibvoid-doom-qlock mt_replan reason=quality stage={} score={:.3} floor={:.3}",
                        stage, quality, min_quality
                    );
                    continue;
                }
                return Err(format!(
                    "neural MT quality floor failure at final stage: score={quality:.3} floor={min_quality:.3}"
                ));
            }
            return Ok(translated);
        }

        Err(if last_error.is_empty() {
            "neural MT failed after emergency ladder".to_string()
        } else {
            format!("neural MT failed after emergency ladder: {last_error}")
        })
    }

    fn should_translate(&self, text: &str) -> bool {
        let source_is_japanese = self.source_lang == "ja" || self.source_lang == "jpn";
        if source_is_japanese {
            return true;
        }
        contains_japanese(text)
    }
}

fn semantic_floor(profile: QualityProfile) -> f64 {
    match profile {
        QualityProfile::Fast => 0.58,
        QualityProfile::Balanced => 0.68,
        QualityProfile::Strict => 0.84,
    }
}

fn default_mt_model_for_profile(profile: QualityProfile) -> &'static str {
    match profile {
        QualityProfile::Fast => "nllb-200-distilled-600M",
        QualityProfile::Balanced => "nllb-200-distilled-600M",
        QualityProfile::Strict => "nllb-200-distilled-1.3B",
    }
}

fn assess_neural_translation_quality(cues: &[SubtitleCue], target_lang: &str) -> f64 {
    if cues.is_empty() || !target_lang.eq_ignore_ascii_case("en") {
        return 1.0;
    }
    let total = cues.len() as f64;
    let mut malformed = 0usize;
    let mut repeat = 0usize;
    let mut very_low_diversity = 0usize;
    let mut duplicates = 0usize;
    let mut generic_duplicates = 0usize;
    let mut prev_line = String::new();
    let mut token_freq = std::collections::HashMap::<String, usize>::new();
    let mut line_freq = std::collections::HashMap::<String, usize>::new();
    let mut total_tokens = 0usize;

    for cue in cues {
        let normalized = cue
            .text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_ascii_lowercase();
        if normalized == prev_line && !normalized.is_empty() {
            if is_generic_interjection_line(&normalized) {
                generic_duplicates += 1;
            } else {
                duplicates += 1;
            }
        }
        if !normalized.is_empty() {
            *line_freq.entry(normalized.clone()).or_insert(0) += 1;
        }
        prev_line = normalized.clone();

        let tokens = tokenize_ascii_words(&cue.text);
        if cue_has_malformed_contraction(&cue.text) || token_has_double_apostrophe(&tokens) {
            malformed += 1;
        }
        if cue_has_adjacent_repeat(&tokens) {
            repeat += 1;
        }
        if tokens.len() >= 8 {
            let unique = tokens
                .iter()
                .map(|token| token.to_ascii_lowercase())
                .collect::<std::collections::HashSet<_>>()
                .len();
            let ratio = unique as f64 / tokens.len() as f64;
            if ratio < 0.45 {
                very_low_diversity += 1;
            }
        }

        for token in tokens {
            let lowered = token.to_ascii_lowercase();
            if lowered.len() >= 2 {
                *token_freq.entry(lowered).or_insert(0) += 1;
                total_tokens += 1;
            }
        }
    }

    let malformed_ratio = malformed as f64 / total;
    let repeat_ratio = repeat as f64 / total;
    let low_div_ratio = very_low_diversity as f64 / total;
    let duplicate_ratio =
        ((duplicates as f64) + (generic_duplicates as f64 * 0.25)) / total.max(1.0);
    let lexical_collapse_penalty = lexical_collapse_penalty(&token_freq, total_tokens, cues.len());
    let line_repetition_penalty = line_repetition_penalty(&line_freq, cues.len());

    (1.0 - (malformed_ratio * 1.5
        + repeat_ratio * 1.1
        + low_div_ratio
        + duplicate_ratio * 1.2
        + lexical_collapse_penalty
        + line_repetition_penalty))
        .clamp(0.0, 1.0)
}

fn lexical_collapse_penalty(
    token_freq: &std::collections::HashMap<String, usize>,
    total_tokens: usize,
    cue_count: usize,
) -> f64 {
    if total_tokens < 200 || cue_count < 80 {
        return 0.0;
    }

    let unique_ratio = token_freq.len() as f64 / total_tokens as f64;
    let dominant_ratio = token_freq
        .values()
        .copied()
        .max()
        .map(|count| count as f64 / total_tokens as f64)
        .unwrap_or(0.0);
    let question_loop_ratio = ["what's", "what", "matter", "mean"]
        .iter()
        .map(|token| token_freq.get(*token).copied().unwrap_or(0))
        .sum::<usize>() as f64
        / total_tokens as f64;

    let mut penalty = 0.0;
    if unique_ratio < 0.030 {
        penalty += ((0.030 - unique_ratio) / 0.030).min(1.0) * 0.8;
    }
    if dominant_ratio > 0.12 {
        penalty += ((dominant_ratio - 0.12) / 0.18).min(1.0) * 0.7;
    }
    if question_loop_ratio > 0.15 {
        penalty += ((question_loop_ratio - 0.15) / 0.25).min(1.0) * 0.5;
    }
    penalty.clamp(0.0, 1.4)
}

fn line_repetition_penalty(
    line_freq: &std::collections::HashMap<String, usize>,
    cue_count: usize,
) -> f64 {
    if cue_count < 80 || line_freq.is_empty() {
        return 0.0;
    }

    let mut top_line = "";
    let mut top_count = 0usize;
    for (line, count) in line_freq {
        if *count > top_count {
            top_count = *count;
            top_line = line;
        }
    }
    if top_count == 0 {
        return 0.0;
    }

    let top_ratio = top_count as f64 / cue_count as f64;
    let mut penalty = 0.0;
    if top_ratio > 0.04 {
        penalty += ((top_ratio - 0.04) / 0.12).min(1.0) * 1.0;
    }
    if is_generic_interjection_line(top_line) && top_ratio > 0.025 {
        penalty += ((top_ratio - 0.025) / 0.12).min(1.0) * 0.5;
    }
    penalty.clamp(0.0, 1.6)
}

fn is_generic_interjection_line(line: &str) -> bool {
    let normalized = line
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphabetic() || ch.is_ascii_whitespace() || ch == '\'' {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    matches!(
        normalized.as_str(),
        "oh my god" | "huh" | "what" | "yeah" | "wait" | "wait a minute" | "hold on a second"
    )
}

fn source_is_japanese(source_lang: &str) -> bool {
    let lowered = source_lang.to_ascii_lowercase();
    lowered == "ja" || lowered == "jpn" || lowered.starts_with("ja-")
}

fn build_adaptive_context_tags(cues: &[SubtitleCue], profile: QualityProfile) -> Vec<Vec<String>> {
    if cues.is_empty() {
        return Vec::new();
    }
    let ranges = split_scene_ranges(cues);
    let mut tags = vec![Vec::<String>::new(); cues.len()];
    for (start, end) in ranges {
        let scene_difficulty = estimate_scene_difficulty_for_tags(&cues[start..end], profile);
        let scene_tag = if scene_difficulty >= 0.72 {
            "scene_hard"
        } else if scene_difficulty >= 0.46 {
            "scene_medium"
        } else {
            "scene_easy"
        };

        for index in start..end {
            let cue = &cues[index];
            let mut cue_tags = vec![scene_tag.to_string()];
            if cue.text.contains('？') || cue.text.contains('?') {
                cue_tags.push("cue_question".to_string());
            }
            if cue.text.contains('！') || cue.text.contains('!') {
                cue_tags.push("cue_exclaim".to_string());
            }
            if let Ok((cue_start, cue_end)) = parse_srt_timing_line(&cue.timing) {
                let duration = (cue_end - cue_start).max(0.001);
                let dense_chars =
                    cue.text.chars().filter(|ch| !ch.is_whitespace()).count() as f64 / duration;
                if dense_chars >= 9.5 {
                    cue_tags.push("cue_fast".to_string());
                }
            }
            tags[index] = cue_tags;
        }
    }
    tags
}

fn split_scene_ranges(cues: &[SubtitleCue]) -> Vec<(usize, usize)> {
    if cues.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::<(usize, usize)>::new();
    let mut scene_start: Option<usize> = None;
    let mut scene_end_exclusive = 0usize;
    let mut prev_end = 0.0f64;
    for (idx, cue) in cues.iter().enumerate() {
        let (start, end) = match parse_srt_timing_line(&cue.timing) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if scene_start.is_none() {
            scene_start = Some(idx);
        } else {
            let gap = (start - prev_end).max(0.0);
            if gap >= 1.5 || (start - prev_end) <= -0.10 {
                if let Some(start_idx) = scene_start {
                    ranges.push((start_idx, scene_end_exclusive));
                }
                scene_start = Some(idx);
            }
        }
        scene_end_exclusive = idx + 1;
        prev_end = prev_end.max(end);
    }
    if let Some(start_idx) = scene_start {
        ranges.push((start_idx, scene_end_exclusive));
    }
    if ranges.is_empty() {
        vec![(0, cues.len())]
    } else {
        ranges
    }
}

fn estimate_scene_difficulty_for_tags(cues: &[SubtitleCue], profile: QualityProfile) -> f64 {
    if cues.is_empty() {
        return 0.5;
    }
    let mut short_ratio_count = 0usize;
    let mut fast_gap_count = 0usize;
    let mut exclaim_count = 0usize;
    let mut question_count = 0usize;
    let mut prev_end = 0.0f64;
    let mut parsed = 0usize;

    for cue in cues {
        if let Ok((start, end)) = parse_srt_timing_line(&cue.timing) {
            let duration = (end - start).max(0.001);
            if duration < 1.0 {
                short_ratio_count += 1;
            }
            if parsed > 0 {
                let gap = (start - prev_end).max(0.0);
                if gap < 0.12 {
                    fast_gap_count += 1;
                }
            }
            prev_end = prev_end.max(end);
            parsed += 1;
        }
        if cue.text.contains('！') || cue.text.contains('!') {
            exclaim_count += 1;
        }
        if cue.text.contains('？') || cue.text.contains('?') {
            question_count += 1;
        }
    }

    let total = cues.len().max(1) as f64;
    let short_ratio = short_ratio_count as f64 / total;
    let fast_gap_ratio = fast_gap_count as f64 / total;
    let exclaim_ratio = exclaim_count as f64 / total;
    let question_ratio = question_count as f64 / total;
    let profile_bias = match profile {
        QualityProfile::Fast => -0.04,
        QualityProfile::Balanced => 0.0,
        QualityProfile::Strict => 0.05,
    };

    (0.32
        + short_ratio * 0.28
        + fast_gap_ratio * 0.22
        + exclaim_ratio * 0.12
        + question_ratio * 0.10
        + profile_bias)
        .clamp(0.10, 0.95)
}

fn parse_srt_timing_line(line: &str) -> Result<(f64, f64), String> {
    let (start, end) = line
        .split_once("-->")
        .ok_or_else(|| format!("invalid timing line: {line}"))?;
    let start = parse_srt_timestamp_to_seconds(start)?;
    let end = parse_srt_timestamp_to_seconds(end)?;
    if end < start {
        return Err(format!("invalid timing (end < start): {line}"));
    }
    Ok((start, end))
}

fn parse_srt_timestamp_to_seconds(ts: &str) -> Result<f64, String> {
    let ts = ts.trim();
    let (hms, ms) = ts
        .split_once(',')
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?;
    let mut parts = hms.split(':');
    let h = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let m = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let s = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let ms = ms
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    Ok((h as f64) * 3600.0 + (m as f64) * 60.0 + (s as f64) + (ms as f64) / 1000.0)
}

fn normalize_short_japanese_cue(text: &str) -> String {
    text.chars()
        .filter(|ch| {
            !ch.is_whitespace()
                && !matches!(
                    ch,
                    '。' | '、'
                        | '！'
                        | '？'
                        | '!'
                        | '?'
                        | '…'
                        | '.'
                        | ','
                        | '，'
                        | '・'
                        | 'ー'
                        | '-'
                        | '～'
                        | '~'
                        | '「'
                        | '」'
                        | '『'
                        | '』'
                        | '（'
                        | '）'
                        | '('
                        | ')'
                        | '　'
                )
        })
        .collect::<String>()
}

fn map_short_japanese_interjection(source: &str) -> Option<&'static str> {
    match source {
        "え" | "えっ" | "えぇ" | "あれ" | "あれっ" => Some("Huh?"),
        "はい" | "うん" | "うんうん" => Some("Yeah."),
        "ほんと" | "ほんとう" | "本当" => Some("Really?"),
        "で" => Some("And?"),
        "ちょっと" | "待って" | "ちょっと待って" => Some("Wait."),
        "待って待って" => Some("Wait, wait."),
        "ありがとう" | "ありがとうございます" => Some("Thank you."),
        "行くか" => Some("Let's go."),
        "せーの" | "せーのー" => Some("Ready..."),
        _ => None,
    }
}

fn apply_source_aware_short_cue_repairs(
    source_cues: &[SubtitleCue],
    translated_cues: &mut [SubtitleCue],
) {
    for (source, translated) in source_cues.iter().zip(translated_cues.iter_mut()) {
        let normalized = normalize_short_japanese_cue(&source.text);
        if normalized.chars().count() > 10 {
            continue;
        }
        if let Some(mapped) = map_short_japanese_interjection(normalized.as_str()) {
            translated.text = mapped.to_string();
        }
    }
}

fn is_mt_memory_error(error: &str) -> bool {
    let lowered = error.to_ascii_lowercase();
    lowered.contains("out of memory")
        || lowered.contains("cuda failed with error out of memory")
        || lowered.contains("cuda out of memory")
        || lowered.contains("oom")
}

fn build_neural_emergency_plans(
    base: &NeuralMTConfig,
    quality_profile: QualityProfile,
) -> Vec<NeuralMTConfig> {
    let mut plans = Vec::<NeuralMTConfig>::new();
    plans.push(base.clone());

    let mut tuned = base.clone();
    tuned.batch_size = (tuned.batch_size * 3 / 4).max(8);
    tuned.max_batch_tokens = (tuned.max_batch_tokens * 3 / 4).max(2048);
    tuned.beam_size = (tuned.beam_size + 1).min(8);
    tuned.oom_retries = (tuned.oom_retries + 1).min(8);
    plans.push(tuned.clone());

    if base.allow_cpu_fallback_on_oom {
        let mut cpu = tuned.clone();
        cpu.gpu = false;
        cpu.allow_cpu_fallback_on_oom = true;
        plans.push(cpu.clone());

        if quality_profile != QualityProfile::Strict {
            if let Some(model) = next_smaller_mt_model(&cpu.model_name) {
                let mut smaller = cpu;
                smaller.model_name = model;
                smaller.batch_size = (smaller.batch_size * 3 / 4).max(6);
                smaller.max_batch_tokens = (smaller.max_batch_tokens * 3 / 4).max(1536);
                plans.push(smaller);
            }
        }
    } else if quality_profile != QualityProfile::Strict {
        if let Some(model) = next_smaller_mt_model(&tuned.model_name) {
            let mut smaller = tuned;
            smaller.model_name = model;
            smaller.batch_size = (smaller.batch_size * 3 / 4).max(6);
            smaller.max_batch_tokens = (smaller.max_batch_tokens * 3 / 4).max(1536);
            plans.push(smaller);
        }
    }

    dedup_plan_sequence(&plans)
}

fn dedup_plan_sequence(plans: &[NeuralMTConfig]) -> Vec<NeuralMTConfig> {
    let mut unique = Vec::<NeuralMTConfig>::new();
    for plan in plans {
        let duplicate = unique.iter().any(|seen| {
            seen.model_name == plan.model_name
                && seen.gpu == plan.gpu
                && seen.batch_size == plan.batch_size
                && seen.max_batch_tokens == plan.max_batch_tokens
                && seen.beam_size == plan.beam_size
        });
        if !duplicate {
            unique.push(plan.clone());
        }
    }
    unique
}

fn next_smaller_mt_model(model_name: &str) -> Option<String> {
    let lowered = model_name.to_ascii_lowercase();
    if lowered.contains("3.3b") {
        return Some(model_name.replace("3.3B", "1.3B").replace("3.3b", "1.3b"));
    }
    if lowered.contains("1.3b") {
        return Some(model_name.replace("1.3B", "600M").replace("1.3b", "600m"));
    }
    None
}

fn squash_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn tokenize_ascii_words(text: &str) -> Vec<String> {
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

fn token_has_double_apostrophe(tokens: &[String]) -> bool {
    tokens.iter().any(|token| token.matches('\'').count() >= 2)
}

fn cue_has_malformed_contraction(text: &str) -> bool {
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

fn cue_has_adjacent_repeat(tokens: &[String]) -> bool {
    if tokens.len() < 4 {
        return false;
    }
    let lowered: Vec<String> = tokens
        .iter()
        .map(|token| token.to_ascii_lowercase())
        .collect();

    for pair in lowered.windows(2) {
        if pair[0] == pair[1] {
            return true;
        }
    }
    for i in 0..(lowered.len().saturating_sub(3)) {
        if lowered[i..i + 2] == lowered[i + 2..i + 4] {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MTDevice {
    Cpu,
    Cuda,
}

fn resolve_mt_device(request_gpu: bool, require_gpu: bool) -> Result<MTDevice, String> {
    if !request_gpu {
        return Ok(MTDevice::Cpu);
    }

    let probe = neural_mt_cuda_device_count();
    match probe {
        Ok(count) if count > 0 => Ok(MTDevice::Cuda),
        Ok(_) => {
            if require_gpu {
                Err("GPU was required but ctranslate2 reports 0 CUDA devices.".to_string())
            } else {
                eprintln!(
                    "warning: --gpu was requested for translation, but ctranslate2 reports no CUDA devices; using CPU."
                );
                Ok(MTDevice::Cpu)
            }
        }
        Err(error) => {
            if require_gpu {
                Err(format!(
                    "GPU was required but ctranslate2 CUDA probe failed: {error}"
                ))
            } else {
                eprintln!(
                    "warning: --gpu was requested for translation, but CUDA probe failed ({error}); using CPU."
                );
                Ok(MTDevice::Cpu)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DecodeProfile {
    context_radius: usize,
    batch_size: usize,
    max_batch_tokens: usize,
    oom_retries: usize,
    beam_size: usize,
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
    prepend_prev_context: bool,
}

fn decode_profile(profile: QualityProfile) -> DecodeProfile {
    match profile {
        QualityProfile::Fast => DecodeProfile {
            context_radius: 1,
            batch_size: 32,
            max_batch_tokens: 8192,
            oom_retries: 1,
            beam_size: 2,
            repetition_penalty: 1.15,
            no_repeat_ngram_size: 2,
            prepend_prev_context: false,
        },
        QualityProfile::Balanced => DecodeProfile {
            context_radius: 2,
            batch_size: 24,
            max_batch_tokens: 6144,
            oom_retries: 2,
            beam_size: 4,
            repetition_penalty: 1.10,
            no_repeat_ngram_size: 3,
            prepend_prev_context: false,
        },
        QualityProfile::Strict => DecodeProfile {
            context_radius: 5,
            batch_size: 16,
            max_batch_tokens: 4096,
            oom_retries: 3,
            beam_size: 8,
            repetition_penalty: 1.02,
            no_repeat_ngram_size: 4,
            prepend_prev_context: true,
        },
    }
}

fn resolve_translate_script_path() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("SUB_ZERO_MT_SCRIPT") {
        let path = PathBuf::from(explicit.trim());
        if path.is_file() {
            return Some(path);
        }
    }

    let cwd_candidate = PathBuf::from("scripts").join("translate_batch.py");
    if cwd_candidate.is_file() {
        return Some(cwd_candidate);
    }

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from));
    if let Some(exe_dir) = exe_dir {
        let adjacent = exe_dir.join("scripts").join("translate_batch.py");
        if adjacent.is_file() {
            return Some(adjacent);
        }
        let parent_adjacent = exe_dir
            .join("..")
            .join("scripts")
            .join("translate_batch.py");
        if parent_adjacent.is_file() {
            return Some(parent_adjacent);
        }
    }

    None
}

// ── Phrase-table translator (legacy fallback) ────────────────────────────────

fn translate_ja_to_en(text: &str) -> String {
    let compact = normalize_for_match(text);
    if let Some(exact) = exact_phrase_translation(&compact) {
        return exact.to_string();
    }

    let mut output = text.to_string();
    for (source, target) in replacement_rules() {
        output = output.replace(source, target);
    }

    if output == text {
        return text.to_string();
    }

    cleanup_spacing(&output)
}

fn normalize_for_match(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_ascii_whitespace() && *c != '。' && *c != '！' && *c != '？')
        .collect::<String>()
}

fn exact_phrase_translation(text: &str) -> Option<&'static str> {
    match text {
        "こんにちは" => Some("hello"),
        "こんばんは" => Some("good evening"),
        "おはよう" | "おはようございます" => Some("good morning"),
        "ありがとう" | "ありがとうございます" => Some("thank you"),
        "すみません" => Some("excuse me"),
        "ごめんなさい" => Some("I'm sorry"),
        "はい" => Some("yes"),
        "いいえ" => Some("no"),
        "行こう" => Some("let's go"),
        "大丈夫" => Some("it's okay"),
        "危ない" => Some("watch out"),
        "何" => Some("what"),
        _ => None,
    }
}

fn replacement_rules() -> &'static [(&'static str, &'static str)] {
    &[
        ("です", "is"),
        ("でした", "was"),
        ("ます", ""),
        ("ません", "not"),
        ("ありがとう", "thank you"),
        ("ございます", ""),
        ("すみません", "excuse me"),
        ("ごめんなさい", "I'm sorry"),
        ("こんにちは", "hello"),
        ("こんばんは", "good evening"),
        ("おはよう", "good morning"),
        ("危ない", "watch out"),
        ("大丈夫", "it's okay"),
        ("行こう", "let's go"),
        ("行く", "go"),
        ("来て", "come"),
        ("待って", "wait"),
        ("お願い", "please"),
        ("何", "what"),
        ("誰", "who"),
        ("どこ", "where"),
        ("なぜ", "why"),
        ("私", "I"),
        ("あなた", "you"),
    ]
}

fn cleanup_spacing(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn contains_japanese(text: &str) -> bool {
    text.chars().any(|c| {
        ('\u{3040}'..='\u{309F}').contains(&c)
            || ('\u{30A0}'..='\u{30FF}').contains(&c)
            || ('\u{4E00}'..='\u{9FFF}').contains(&c)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        apply_source_aware_short_cue_repairs, assess_neural_translation_quality, contains_japanese,
        decode_profile, default_mt_model_for_profile, Translator, TranslatorConfig,
    };
    use crate::engine::srt::SubtitleCue;
    use crate::engine::transcribe::QualityProfile;

    #[test]
    fn detect_japanese_characters() {
        assert!(contains_japanese("こんにちは"));
        assert!(!contains_japanese("hello"));
    }

    #[test]
    fn translate_known_phrase_to_english() {
        let translator = Translator::new(TranslatorConfig {
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            offline: true,
            force_phrase_table: true,
            gpu: false,
            require_gpu: false,
            mt_model: None,
            mt_batch_size: None,
            mt_max_batch_tokens: None,
            mt_oom_retries: None,
            mt_allow_cpu_fallback: true,
            quality_profile: QualityProfile::Balanced,
        })
        .expect("translator should build");
        assert_eq!(translator.translate("こんにちは"), "hello");
    }

    #[test]
    fn preserve_english_text() {
        let translator = Translator::new(TranslatorConfig {
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            offline: true,
            force_phrase_table: true,
            gpu: false,
            require_gpu: false,
            mt_model: None,
            mt_batch_size: None,
            mt_max_batch_tokens: None,
            mt_oom_retries: None,
            mt_allow_cpu_fallback: true,
            quality_profile: QualityProfile::Balanced,
        })
        .expect("translator should build");
        assert_eq!(
            translator.translate("This is already English."),
            "This is already English."
        );
    }

    #[test]
    fn non_english_target_passthrough() {
        let translator = Translator::new(TranslatorConfig {
            source_lang: "ja".to_string(),
            target_lang: "fr".to_string(),
            offline: false,
            force_phrase_table: true,
            gpu: false,
            require_gpu: false,
            mt_model: None,
            mt_batch_size: None,
            mt_max_batch_tokens: None,
            mt_oom_retries: None,
            mt_allow_cpu_fallback: true,
            quality_profile: QualityProfile::Balanced,
        })
        .expect("translator should build");
        assert_eq!(translator.translate("こんにちは"), "こんにちは");
    }

    #[test]
    fn strict_profile_uses_stronger_decode_settings() {
        let strict = decode_profile(QualityProfile::Strict);
        let balanced = decode_profile(QualityProfile::Balanced);
        assert!(strict.beam_size > balanced.beam_size);
        assert!(strict.batch_size <= balanced.batch_size);
        assert!(strict.max_batch_tokens <= balanced.max_batch_tokens);
        assert!(strict.oom_retries >= balanced.oom_retries);
        assert!(strict.repetition_penalty < balanced.repetition_penalty);
        assert!(strict.no_repeat_ngram_size >= balanced.no_repeat_ngram_size);
        assert!(strict.prepend_prev_context);
    }

    #[test]
    fn strict_profile_prefers_stronger_mt_model() {
        assert_eq!(
            default_mt_model_for_profile(QualityProfile::Strict),
            "nllb-200-distilled-1.3B"
        );
        assert_eq!(
            default_mt_model_for_profile(QualityProfile::Balanced),
            "nllb-200-distilled-600M"
        );
    }

    #[test]
    fn semantic_quality_penalizes_lexical_collapse() {
        let cues: Vec<SubtitleCue> = (1..=160)
            .map(|index| SubtitleCue {
                index,
                timing: "00:00:00,000 --> 00:00:02,000".to_string(),
                text: "What's the matter with you? What's your problem?".to_string(),
            })
            .collect();
        let score = assess_neural_translation_quality(&cues, "en");
        assert!(
            score < 0.35,
            "expected collapsed output to score poorly, got {score}"
        );
    }

    #[test]
    fn semantic_quality_accepts_diverse_lines() {
        fn synthetic_word(mut seed: usize) -> String {
            let mut out = String::from("w");
            for _ in 0..6 {
                let ch = ((seed % 26) as u8 + b'a') as char;
                out.push(ch);
                seed = seed / 26 + 11;
            }
            out
        }

        let cues: Vec<SubtitleCue> = (1..=120)
            .map(|index| SubtitleCue {
                index,
                timing: "00:00:00,000 --> 00:00:02,000".to_string(),
                text: format!(
                    "{} {} {} {} {} {}.",
                    synthetic_word(index),
                    synthetic_word(index + 17),
                    synthetic_word(index + 41),
                    synthetic_word(index + 73),
                    synthetic_word(index + 101),
                    synthetic_word(index + 139),
                ),
            })
            .collect();
        let score = assess_neural_translation_quality(&cues, "en");
        assert!(score > 0.80, "expected healthy output quality, got {score}");
    }

    #[test]
    fn semantic_quality_penalizes_single_line_collapse() {
        let cues: Vec<SubtitleCue> = (1..=180)
            .map(|index| SubtitleCue {
                index,
                timing: "00:00:00,000 --> 00:00:02,000".to_string(),
                text: if index % 10 == 0 {
                    "Let's move.".to_string()
                } else {
                    "Oh, my God.".to_string()
                },
            })
            .collect();
        let score = assess_neural_translation_quality(&cues, "en");
        assert!(
            score < 0.45,
            "expected repeated interjection collapse to score poorly, got {score}"
        );
    }

    #[test]
    fn source_aware_short_cue_repair_maps_common_japanese_fillers() {
        let source = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "え？".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,000 --> 00:00:02,000".to_string(),
                text: "待って待って".to_string(),
            },
        ];
        let mut translated = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "Oh, my God.".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,000 --> 00:00:02,000".to_string(),
                text: "Hold on.".to_string(),
            },
        ];
        apply_source_aware_short_cue_repairs(&source, &mut translated);
        assert_eq!(translated[0].text, "Huh?");
        assert_eq!(translated[1].text, "Wait, wait.");
    }
}
