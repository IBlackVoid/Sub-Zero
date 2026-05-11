use super::health::{SrtHealth, TranslationSemanticHealth};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub(super) struct LearnedGateOutcome {
    pub(super) model_path: PathBuf,
    pub(super) enforce: bool,
    pub(super) threshold: f64,
    pub(super) score: f64,
    pub(super) pass: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct LearnedGateModel {
    version: String,
    kind: String,
    threshold: f64,
    bias: f64,
    features: Vec<String>,
    mean: Vec<f64>,
    std: Vec<f64>,
    weights: Vec<f64>,
}

static MODEL: OnceLock<Option<(PathBuf, LearnedGateModel)>> = OnceLock::new();

pub(super) fn evaluate(
    structural: &SrtHealth,
    semantic: &TranslationSemanticHealth,
    speaker_info: Option<&serde_json::Value>,
) -> Option<LearnedGateOutcome> {
    let (model_path, model) = model_ref()?;

    let enforce = env_truthy("SUB_ZERO_LEARNED_GATE_ENFORCE");
    let threshold = if model.threshold.is_finite() {
        model.threshold
    } else {
        0.5
    };

    let score = score_model(model, structural, semantic, speaker_info);
    let pass = score >= threshold;

    Some(LearnedGateOutcome {
        model_path: model_path.clone(),
        enforce,
        threshold,
        score,
        pass,
    })
}

fn model_ref() -> Option<&'static (PathBuf, LearnedGateModel)> {
    MODEL.get_or_init(load_model).as_ref()
}

fn load_model() -> Option<(PathBuf, LearnedGateModel)> {
    let Ok(path) = std::env::var("SUB_ZERO_LEARNED_GATE_MODEL") else {
        return None;
    };
    let path = PathBuf::from(path);
    if !path.is_file() {
        eprintln!(
            "warning: learned gate model path does not exist: {}",
            path.display()
        );
        return None;
    }

    let raw = match std::fs::read_to_string(&path) {
        Ok(v) => v,
        Err(error) => {
            eprintln!(
                "warning: failed to read learned gate model {}: {error}",
                path.display()
            );
            return None;
        }
    };
    let model: LearnedGateModel = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(error) => {
            eprintln!(
                "warning: invalid learned gate model JSON {}: {error}",
                path.display()
            );
            return None;
        }
    };

    if model.kind != "learned-quality-gate" || model.version != "1.0" {
        eprintln!(
            "warning: unsupported learned gate model format {} kind={} version={}",
            path.display(),
            model.kind,
            model.version
        );
        return None;
    }
    let n = model.features.len();
    if model.weights.len() != n || model.mean.len() != n || model.std.len() != n {
        eprintln!(
            "warning: learned gate model shape mismatch {}: features={} weights={} mean={} std={}",
            path.display(),
            model.features.len(),
            model.weights.len(),
            model.mean.len(),
            model.std.len(),
        );
        return None;
    }

    Some((path, model))
}

fn env_truthy(key: &str) -> bool {
    let Ok(v) = std::env::var(key) else {
        return false;
    };
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn score_model(
    model: &LearnedGateModel,
    structural: &SrtHealth,
    semantic: &TranslationSemanticHealth,
    speaker_info: Option<&serde_json::Value>,
) -> f64 {
    let mut z = model.bias;
    for (idx, name) in model.features.iter().enumerate() {
        let x = feature_value(name, structural, semantic, speaker_info);
        let mean = model.mean[idx];
        let std = model.std[idx];
        let w = model.weights[idx];
        let denom = if std.is_finite() && std.abs() > 1e-9 {
            std
        } else {
            1.0
        };
        let xs = (x - mean) / denom;
        z += w * xs;
    }
    sigmoid(z)
}

fn sigmoid(z: f64) -> f64 {
    if !z.is_finite() {
        return 0.0;
    }
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

fn feature_value(
    name: &str,
    structural: &SrtHealth,
    semantic: &TranslationSemanticHealth,
    speaker_info: Option<&serde_json::Value>,
) -> f64 {
    match name {
        "cue_count" => semantic.cue_count as f64,
        "top_line_ratio" => structural.top_line_ratio,
        "overlap_ratio" => structural.overlap_ratio,
        "non_empty_ratio" => structural.non_empty_ratio,
        "anomaly_ratio" => semantic.anomaly_ratio,
        "malformed_contraction_ratio" => semantic.malformed_contraction_ratio,
        "low_function_word_ratio" => semantic.low_function_word_ratio,
        "adjacent_repeat_ratio" => semantic.adjacent_repeat_ratio,
        "scene_low_quality_ratio" => semantic.scene_low_quality_ratio,
        "scene_count" => semantic.scene_count as f64,
        "name_inconsistency_ratio" => semantic.name_inconsistency_ratio,
        "register_speakers_observed" => {
            speaker_json_number(speaker_info, &["register", "speakers_observed"])
        }
        "register_speakers_formal" => {
            speaker_json_number(speaker_info, &["register", "speakers_formal"])
        }
        "register_cues_labeled" => speaker_json_number(speaker_info, &["register", "cues_labeled"]),
        "diar_speakers" => speaker_json_number(speaker_info, &["audio_diarization", "speakers"]),
        "diar_used_segments" => {
            speaker_json_number(speaker_info, &["audio_diarization", "used_segments"])
        }
        "diar_assigned_cues" => {
            speaker_json_number(speaker_info, &["audio_diarization", "assigned_cues"])
        }
        _ => 0.0,
    }
}

fn speaker_json_number(root: Option<&serde_json::Value>, path: &[&str]) -> f64 {
    let Some(mut cur) = root else {
        return 0.0;
    };
    for key in path {
        let Some(next) = cur.get(*key) else {
            return 0.0;
        };
        cur = next;
    }
    cur.as_f64()
        .unwrap_or_else(|| cur.as_i64().unwrap_or(0) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_is_bounded() {
        assert!(sigmoid(0.0) > 0.49 && sigmoid(0.0) < 0.51);
        assert!(sigmoid(20.0) > 0.999);
        assert!(sigmoid(-20.0) < 0.001);
    }
}
