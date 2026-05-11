use super::{
    assess_name_inconsistency, assess_scene_quality, cue_has_adjacent_repeat,
    cue_has_low_function_word_coverage, cue_has_malformed_contraction, parse_srt_timing_line,
    token_has_double_apostrophe, tokenize_ascii_words,
};
use crate::engine::srt::SubtitleCue;
use crate::engine::transcribe::QualityProfile;
use std::collections::HashMap;

pub(super) fn build_metadata_warnings(semantic: &TranslationSemanticHealth) -> Vec<String> {
    let mut warnings = Vec::<String>::new();
    if semantic.scene_low_quality_ratio > 0.0 {
        warnings.push(format!(
            "scene_low_quality_ratio={:.2}%",
            semantic.scene_low_quality_ratio * 100.0
        ));
    }
    if semantic.name_inconsistency_ratio > 0.0 {
        warnings.push(format!(
            "name_inconsistency_ratio={:.2}%",
            semantic.name_inconsistency_ratio * 100.0
        ));
    }
    if semantic.malformed_contraction_ratio > 0.0 {
        warnings.push(format!(
            "malformed_contraction_ratio={:.2}%",
            semantic.malformed_contraction_ratio * 100.0
        ));
    }
    warnings
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SrtHealth {
    pub(super) cue_count: usize,
    pub(super) top_line_ratio: f64,
    pub(super) overlap_ratio: f64,
    pub(super) non_empty_ratio: f64,
}

impl SrtHealth {
    pub(super) fn is_pathological(&self, profile: QualityProfile) -> bool {
        let thresholds = HealthThresholds::for_profile(profile);
        if self.cue_count < thresholds.min_cues {
            return false;
        }
        self.top_line_ratio >= thresholds.max_top_line_ratio
            || self.overlap_ratio >= thresholds.max_overlap_ratio
            || self.non_empty_ratio < thresholds.min_non_empty_ratio
    }

    pub(super) fn summary(&self) -> String {
        format!(
            "cues={} top_line_ratio={:.2}% overlap_ratio={:.2}% non_empty_ratio={:.2}%",
            self.cue_count,
            self.top_line_ratio * 100.0,
            self.overlap_ratio * 100.0,
            self.non_empty_ratio * 100.0
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TranslationSemanticHealth {
    pub(super) cue_count: usize,
    pub(super) anomaly_ratio: f64,
    pub(super) malformed_contraction_ratio: f64,
    pub(super) low_function_word_ratio: f64,
    pub(super) adjacent_repeat_ratio: f64,
    pub(super) scene_low_quality_ratio: f64,
    pub(super) scene_count: usize,
    pub(super) name_inconsistency_ratio: f64,
}

impl TranslationSemanticHealth {
    pub(super) fn is_pathological(&self, profile: QualityProfile) -> bool {
        if self.cue_count == 0 {
            return false;
        }
        let thresholds = SemanticThresholds::for_profile(profile);
        self.anomaly_ratio >= thresholds.max_anomaly_ratio
            || self.malformed_contraction_ratio >= thresholds.max_malformed_contraction_ratio
            || self.low_function_word_ratio >= thresholds.max_low_function_word_ratio
            || self.adjacent_repeat_ratio >= thresholds.max_adjacent_repeat_ratio
            || self.scene_low_quality_ratio >= thresholds.max_scene_low_quality_ratio
            || self.name_inconsistency_ratio >= thresholds.max_name_inconsistency_ratio
    }

    pub(super) fn summary(&self) -> String {
        format!(
            "cues={} anomaly_ratio={:.2}% malformed_contraction_ratio={:.2}% low_function_word_ratio={:.2}% adjacent_repeat_ratio={:.2}% scene_low_quality_ratio={:.2}% scene_count={} name_inconsistency_ratio={:.2}%",
            self.cue_count,
            self.anomaly_ratio * 100.0,
            self.malformed_contraction_ratio * 100.0,
            self.low_function_word_ratio * 100.0,
            self.adjacent_repeat_ratio * 100.0,
            self.scene_low_quality_ratio * 100.0,
            self.scene_count,
            self.name_inconsistency_ratio * 100.0,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct HealthThresholds {
    min_cues: usize,
    max_top_line_ratio: f64,
    max_overlap_ratio: f64,
    min_non_empty_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
struct SemanticThresholds {
    max_anomaly_ratio: f64,
    max_malformed_contraction_ratio: f64,
    max_low_function_word_ratio: f64,
    max_adjacent_repeat_ratio: f64,
    max_scene_low_quality_ratio: f64,
    max_name_inconsistency_ratio: f64,
}

impl SemanticThresholds {
    fn for_profile(profile: QualityProfile) -> Self {
        match profile {
            QualityProfile::Fast => Self {
                max_anomaly_ratio: 0.35,
                max_malformed_contraction_ratio: 0.25,
                max_low_function_word_ratio: 0.35,
                max_adjacent_repeat_ratio: 0.30,
                max_scene_low_quality_ratio: 0.45,
                max_name_inconsistency_ratio: 0.20,
            },
            QualityProfile::Balanced => Self {
                max_anomaly_ratio: 0.25,
                max_malformed_contraction_ratio: 0.15,
                max_low_function_word_ratio: 0.22,
                max_adjacent_repeat_ratio: 0.18,
                max_scene_low_quality_ratio: 0.30,
                max_name_inconsistency_ratio: 0.12,
            },
            QualityProfile::Strict => Self {
                max_anomaly_ratio: 0.03,
                max_malformed_contraction_ratio: 0.01,
                max_low_function_word_ratio: 0.08,
                max_adjacent_repeat_ratio: 0.02,
                max_scene_low_quality_ratio: 0.04,
                max_name_inconsistency_ratio: 0.03,
            },
        }
    }
}

impl HealthThresholds {
    fn for_profile(profile: QualityProfile) -> Self {
        match profile {
            QualityProfile::Fast => Self {
                min_cues: 240,
                max_top_line_ratio: 0.35,
                max_overlap_ratio: 0.40,
                min_non_empty_ratio: 0.65,
            },
            QualityProfile::Balanced => Self {
                min_cues: 200,
                max_top_line_ratio: 0.30,
                max_overlap_ratio: 0.35,
                min_non_empty_ratio: 0.70,
            },
            QualityProfile::Strict => Self {
                min_cues: 80,
                max_top_line_ratio: 0.20,
                max_overlap_ratio: 0.25,
                min_non_empty_ratio: 0.90,
            },
        }
    }
}

pub(super) fn assess_srt_health(cues: &[SubtitleCue]) -> Result<SrtHealth, String> {
    if cues.is_empty() {
        return Ok(SrtHealth {
            cue_count: 0,
            top_line_ratio: 0.0,
            overlap_ratio: 0.0,
            non_empty_ratio: 0.0,
        });
    }

    let mut freq = HashMap::<String, usize>::new();
    let mut non_empty = 0usize;
    for cue in cues {
        let normalized = normalize_health_text(&cue.text);
        if normalized.is_empty() {
            continue;
        }
        non_empty += 1;
        *freq.entry(normalized).or_insert(0) += 1;
    }
    let top_count = freq.values().copied().max().unwrap_or(0);
    let top_line_ratio = (top_count as f64) / (cues.len() as f64);
    let non_empty_ratio = (non_empty as f64) / (cues.len() as f64);

    let mut overlaps = 0usize;
    let mut parsed = 0usize;
    let mut prev_end = 0.0f64;
    for cue in cues {
        let (start, end) = parse_srt_timing_line(&cue.timing)?;
        if parsed > 0 && start < prev_end {
            overlaps += 1;
        }
        parsed += 1;
        prev_end = prev_end.max(end);
    }
    let overlap_ratio = if parsed > 1 {
        (overlaps as f64) / ((parsed - 1) as f64)
    } else {
        0.0
    };

    Ok(SrtHealth {
        cue_count: cues.len(),
        top_line_ratio,
        overlap_ratio,
        non_empty_ratio,
    })
}

pub(super) fn normalize_health_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

pub(super) fn assess_translation_semantics(
    cues: &[SubtitleCue],
    target_lang: &str,
) -> TranslationSemanticHealth {
    // This semantic gate currently targets English fluency. For other targets,
    // we skip fluency scoring to avoid language-specific false positives.
    if !target_lang.eq_ignore_ascii_case("en") {
        return TranslationSemanticHealth {
            cue_count: cues.len(),
            anomaly_ratio: 0.0,
            malformed_contraction_ratio: 0.0,
            low_function_word_ratio: 0.0,
            adjacent_repeat_ratio: 0.0,
            scene_low_quality_ratio: 0.0,
            scene_count: 0,
            name_inconsistency_ratio: 0.0,
        };
    }

    let mut anomaly_count = 0usize;
    let mut malformed_contractions = 0usize;
    let mut low_function_word = 0usize;
    let mut adjacent_repeat = 0usize;

    for cue in cues {
        let tokens = tokenize_ascii_words(&cue.text);
        let mut anomalous = false;

        let has_malformed_contraction =
            cue_has_malformed_contraction(&cue.text) || token_has_double_apostrophe(&tokens);
        if has_malformed_contraction {
            malformed_contractions += 1;
            anomalous = true;
        }

        if cue_has_low_function_word_coverage(&tokens) {
            low_function_word += 1;
            anomalous = true;
        }

        if cue_has_adjacent_repeat(&tokens) {
            adjacent_repeat += 1;
            anomalous = true;
        }

        if anomalous {
            anomaly_count += 1;
        }
    }

    let total = cues.len().max(1) as f64;
    let scene_report = assess_scene_quality(cues);
    let name_inconsistency_ratio = assess_name_inconsistency(cues);
    TranslationSemanticHealth {
        cue_count: cues.len(),
        anomaly_ratio: (anomaly_count as f64) / total,
        malformed_contraction_ratio: (malformed_contractions as f64) / total,
        low_function_word_ratio: (low_function_word as f64) / total,
        adjacent_repeat_ratio: (adjacent_repeat as f64) / total,
        scene_low_quality_ratio: scene_report.low_quality_ratio,
        scene_count: scene_report.scene_count,
        name_inconsistency_ratio,
    }
}

pub(super) fn scene_semantic_penalty(health: &TranslationSemanticHealth) -> f64 {
    health.anomaly_ratio * 1.3
        + health.malformed_contraction_ratio * 1.5
        + health.low_function_word_ratio
        + health.adjacent_repeat_ratio * 1.2
}
