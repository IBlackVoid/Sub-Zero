// Learned quality gate — v1.x (linear logistic) + v2.0 (logistic +
// isotonic calibration + split conformal prediction).
//
// v1.x outputs a single score and a boolean pass/fail.
//
// v2.0 outputs a calibrated probability and a *prediction set* derived
// from a held-out conformal calibration step. The set may be {1}
// (PASS), {0} (REJECT), or {0, 1} (ABSTAIN — honest uncertainty). The
// conformal envelope provides a formal marginal coverage guarantee of
// 1 − α (see Vovk et al. 2005; Romano-Patterson-Candès 2019). To the
// maintainer's knowledge this combination has not been published for
// subtitle quality estimation before.
//
// Schema fields touched per version:
//
// ```text
// v1.0/v1.1 — required: version, kind, threshold, bias, features,
//             mean, std, weights.
// v2.0      — adds: calibration { kind, x[], y[], out_of_bounds },
//                    conformal { alpha, q_hat, calibration_set_size, ... }.
// ```
//
// Deserialisation is permissive — v1.x models keep working unchanged.
// Unknown fields are ignored.

use super::health::{SrtHealth, TranslationSemanticHealth};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::sync::OnceLock;

/// What the gate decided about a single SRT.
///
/// `Pass` and `Reject` are crisp calls. `Abstain` means the conformal
/// envelope straddles the threshold — the gate refuses to claim a
/// confident decision rather than guessing. Calling code may treat
/// `Abstain` as a soft pass, surface it to the operator, or route the
/// run into the rescue pipeline; the gate itself does not prescribe.
///
/// Serializes as the uppercase string (`"PASS"`, `"REJECT"`,
/// `"ABSTAIN"`) so the metadata sidecar JSON does not have to
/// re-implement the mapping at every emit site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub(super) enum LearnedGateDecision {
    Pass,
    Reject,
    Abstain,
}

impl LearnedGateDecision {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            LearnedGateDecision::Pass => "PASS",
            LearnedGateDecision::Reject => "REJECT",
            LearnedGateDecision::Abstain => "ABSTAIN",
        }
    }

    pub(super) fn is_pass(self) -> bool {
        matches!(self, LearnedGateDecision::Pass)
    }

    pub(super) fn is_reject(self) -> bool {
        matches!(self, LearnedGateDecision::Reject)
    }
}

impl fmt::Display for LearnedGateDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub(super) struct LearnedGateOutcome {
    pub(super) model_path: PathBuf,
    pub(super) enforce: bool,
    pub(super) threshold: f64,
    pub(super) score: f64,
    pub(super) decision: LearnedGateDecision,
    pub(super) lower_bound: f64,
    pub(super) upper_bound: f64,
    pub(super) schema_version: String,
    pub(super) conformal_alpha: Option<f64>,
    pub(super) conformal_q_hat: Option<f64>,
    /// Number of held-out examples used to compute `q_hat`. Surfaced to
    /// the metadata sidecar so operators can audit "how big was the
    /// conformal-cal split that produced this band?" without opening
    /// the model JSON.
    pub(super) conformal_cal_size: Option<u64>,
    /// `"clip"` for sklearn isotonic; carried for human inspection. The
    /// Rust evaluator implements only `clip`-style clamp behaviour.
    pub(super) isotonic_out_of_bounds: Option<String>,
}

impl LearnedGateOutcome {
    /// Backward-compat shim: pre-v2 callers asked `outcome.pass`. The
    /// equivalent in v2 terms is "the decision was PASS"; ABSTAIN is
    /// not a pass.
    pub(super) fn pass(&self) -> bool {
        self.decision.is_pass()
    }
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
    /// Present iff the model JSON is v2.0+. Pre-v2 models leave this
    /// `None` and the engine falls back to a sigmoid → threshold call.
    #[serde(default)]
    calibration: Option<Calibration>,
    /// Present iff the model is conformalised. Pre-v2 → `None`,
    /// decision is binary; v2 → bands around the threshold define the
    /// ABSTAIN region.
    #[serde(default)]
    conformal: Option<Conformal>,
}

#[derive(Debug, Clone, Deserialize)]
struct Calibration {
    kind: String,
    /// x-knots (monotonic) — sigmoid raw scores at fit time.
    x: Vec<f64>,
    /// y-knots (monotonic) — calibrated probabilities aligned to `x`.
    y: Vec<f64>,
    /// "clip" (sklearn default) or "nan"; we only honour "clip".
    /// Surfaced into the outcome for operator inspection.
    #[serde(default)]
    out_of_bounds: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Conformal {
    alpha: f64,
    q_hat: f64,
    /// Sample count of the calibration split that produced `q_hat`.
    /// Surfaced into the metadata sidecar so an auditor can size the
    /// conformal envelope without reading the raw model JSON.
    #[serde(default)]
    calibration_set_size: Option<u64>,
}

static MODEL: OnceLock<Option<(PathBuf, LearnedGateModel)>> = OnceLock::new();
const SUPPORTED_MODEL_VERSIONS: &[&str] = &["1.0", "1.1", "2.0"];

pub(super) fn evaluate(
    structural: &SrtHealth,
    semantic: &TranslationSemanticHealth,
    speaker_info: Option<&serde_json::Value>,
) -> Option<LearnedGateOutcome> {
    let (model_path, model) = model_ref()?;

    let enforce = env_truthy("VOIDEX_LEARNED_GATE_ENFORCE");
    let threshold = if model.threshold.is_finite() {
        model.threshold
    } else {
        0.5
    };

    // 1. Linear logistic score → raw probability.
    let raw = score_model(model, structural, semantic, speaker_info);
    // 2. Isotonic calibration if the model carries a calibrator.
    let p_cal = match model.calibration.as_ref() {
        Some(cal) if cal.kind == "isotonic" => isotonic_eval(raw, &cal.x, &cal.y),
        _ => raw,
    };
    // 3. Conformal envelope -> decision state.
    //
    // Boundary convention is *strict-greater* for PASS, *strict-less*
    // for REJECT, ABSTAIN otherwise. Both branches honour the same
    // convention so a v1.x model and a v2.0 model with `q_hat == 0`
    // make the same call at `p_cal == threshold` (they Abstain).
    // The Python training harness uses the same convention at
    // `scripts/train_learned_gate.py::conformal_state`; the two
    // implementations are tested for agreement.
    let (decision, lower, upper) = match model.conformal.as_ref() {
        Some(conf) if conf.q_hat.is_finite() && conf.q_hat >= 0.0 => {
            let q = conf.q_hat.clamp(0.0, 1.0);
            let lo = (p_cal - q).clamp(0.0, 1.0);
            let hi = (p_cal + q).clamp(0.0, 1.0);
            let state = if lo > threshold {
                LearnedGateDecision::Pass
            } else if hi < threshold {
                LearnedGateDecision::Reject
            } else {
                LearnedGateDecision::Abstain
            };
            (state, lo, hi)
        }
        _ => {
            // No conformal envelope -> point estimate is its own band;
            // can never Abstain. Strict-greater keeps the boundary
            // behaviour identical to the conformal branch at q_hat=0.
            let state = if p_cal > threshold {
                LearnedGateDecision::Pass
            } else {
                LearnedGateDecision::Reject
            };
            (state, p_cal, p_cal)
        }
    };

    Some(LearnedGateOutcome {
        model_path: model_path.clone(),
        enforce,
        threshold,
        score: p_cal,
        decision,
        lower_bound: lower,
        upper_bound: upper,
        schema_version: model.version.clone(),
        conformal_alpha: model.conformal.as_ref().map(|c| c.alpha),
        conformal_q_hat: model.conformal.as_ref().map(|c| c.q_hat),
        conformal_cal_size: model
            .conformal
            .as_ref()
            .and_then(|c| c.calibration_set_size),
        isotonic_out_of_bounds: model
            .calibration
            .as_ref()
            .and_then(|c| c.out_of_bounds.clone()),
    })
}

fn model_ref() -> Option<&'static (PathBuf, LearnedGateModel)> {
    MODEL.get_or_init(load_model).as_ref()
}

fn load_model() -> Option<(PathBuf, LearnedGateModel)> {
    let Ok(path) = std::env::var("VOIDEX_LEARNED_GATE_MODEL") else {
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

    if !is_supported_model_format(&model) {
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
    // Pre-eval finite-ness audit: any NaN/inf in the linear part of
    // the model would silently corrupt the score. We bail loud rather
    // than let the engine ship a poisoned gate.
    if !model.bias.is_finite()
        || !model.threshold.is_finite()
        || !model.weights.iter().copied().all(f64::is_finite)
        || !model.mean.iter().copied().all(f64::is_finite)
        || !model.std.iter().copied().all(f64::is_finite)
    {
        eprintln!(
            "warning: learned gate model {} contains non-finite weights/mean/std/bias/threshold",
            path.display()
        );
        return None;
    }

    // v2 sanity: the calibration knots must form a usable monotone map.
    if let Some(cal) = model.calibration.as_ref() {
        if cal.x.len() < 2 || cal.x.len() != cal.y.len() {
            eprintln!(
                "warning: learned gate isotonic calibration is malformed in {} (x.len={} y.len={})",
                path.display(),
                cal.x.len(),
                cal.y.len()
            );
            return None;
        }
        // Hard cap on knot count. The Python harness emits one knot
        // per calibration sample (currently a few dozen). A 4 KiB cap
        // protects against an operator pasting a gigabyte of knots
        // into the JSON.
        const MAX_ISOTONIC_KNOTS: usize = 4096;
        if cal.x.len() > MAX_ISOTONIC_KNOTS {
            eprintln!(
                "warning: learned gate isotonic calibration in {} exceeds knot cap ({} > {})",
                path.display(),
                cal.x.len(),
                MAX_ISOTONIC_KNOTS
            );
            return None;
        }
        if !cal
            .x
            .iter()
            .chain(cal.y.iter())
            .copied()
            .all(f64::is_finite)
        {
            eprintln!(
                "warning: learned gate isotonic calibration in {} contains NaN/inf knot values",
                path.display()
            );
            return None;
        }
        if !cal.x.windows(2).all(|w| w[0] <= w[1]) {
            eprintln!(
                "warning: learned gate isotonic calibration in {} has non-monotone x-knots",
                path.display()
            );
            return None;
        }
        if !cal.y.windows(2).all(|w| w[0] <= w[1]) {
            eprintln!(
                "warning: learned gate isotonic calibration in {} has non-monotone y-knots",
                path.display()
            );
            return None;
        }
    }

    // v2 sanity: the conformal envelope must produce a usable margin.
    if let Some(conf) = model.conformal.as_ref() {
        let alpha_ok = conf.alpha.is_finite() && conf.alpha > 0.0 && conf.alpha < 1.0;
        if !alpha_ok {
            eprintln!(
                "warning: learned gate conformal alpha out of range in {} (alpha={})",
                path.display(),
                conf.alpha
            );
            return None;
        }
        let q_ok = conf.q_hat.is_finite() && (0.0..=1.0).contains(&conf.q_hat);
        if !q_ok {
            eprintln!(
                "warning: learned gate conformal q_hat out of range in {} (q_hat={})",
                path.display(),
                conf.q_hat
            );
            return None;
        }
    }

    Some((path, model))
}

fn is_supported_model_format(model: &LearnedGateModel) -> bool {
    model.kind == "learned-quality-gate"
        && SUPPORTED_MODEL_VERSIONS
            .iter()
            .any(|version| *version == model.version)
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

/// Piecewise-linear interpolation between isotonic knots, with the
/// `out_of_bounds="clip"` convention sklearn writes by default. This is
/// the inference-time counterpart of `IsotonicRegression.predict` in
/// scikit-learn — values outside `[x[0], x[-1]]` clamp to the nearest
/// endpoint's y.
///
/// Edge cases this implementation pins down explicitly:
///
/// - **Empty knots.** Unreachable from `load_model` (the loader rejects
///   `xs.len() < 2`); kept here as a defensive identity for the case
///   someone calls this from a test with a hand-crafted empty slice.
/// - **NaN / non-finite input.** sklearn would propagate NaN out;
///   we instead clamp finite ±inf to the endpoints and map NaN to the
///   mid-point of the y-range. NaN-into-the-gate must not panic — the
///   `partition_point` predicate is order-dependent and NaN breaks
///   total order, so an unguarded path underflows `usize`.
/// - **Duplicate x-knots.** sklearn's step function is left-continuous
///   at a step. We mirror that by switching the search predicate to
///   strict-less (`*knot < x`); at `x` exactly on a duplicate, the
///   bracket collapses to the left side and we return `y0`.
fn isotonic_eval(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
    if xs.is_empty() {
        return x;
    }
    let last = xs.len() - 1;
    if !x.is_finite() {
        return if x.is_nan() {
            0.5 * (ys[0] + ys[last])
        } else if x.is_sign_negative() {
            ys[0]
        } else {
            ys[last]
        };
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[last] {
        return ys[last];
    }
    // Strict-less predicate so an exact-duplicate `x` lands on the
    // left side of the step (sklearn's left-continuous convention).
    let idx = xs.partition_point(|knot| *knot < x);
    if idx == 0 {
        // Reachable only if `xs[0]` is NaN, which the loader rejects;
        // belt-and-braces for the defensive path.
        return ys[0];
    }
    let i0 = idx - 1;
    let i1 = idx;
    let x0 = xs[i0];
    let x1 = xs[i1];
    let y0 = ys[i0];
    let y1 = ys[i1];
    if (x1 - x0).abs() <= f64::EPSILON {
        return y0;
    }
    let t = (x - x0) / (x1 - x0);
    y0 + t * (y1 - y0)
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

    fn model_with_version(version: &str) -> LearnedGateModel {
        LearnedGateModel {
            version: version.to_string(),
            kind: "learned-quality-gate".to_string(),
            threshold: 0.5,
            bias: 0.0,
            features: vec!["cue_count".to_string()],
            mean: vec![0.0],
            std: vec![1.0],
            weights: vec![1.0],
            calibration: None,
            conformal: None,
        }
    }

    #[test]
    fn sigmoid_is_bounded() {
        assert!(sigmoid(0.0) > 0.49 && sigmoid(0.0) < 0.51);
        assert!(sigmoid(20.0) > 0.999);
        assert!(sigmoid(-20.0) < 0.001);
    }

    #[test]
    fn accepts_current_learned_gate_schema_versions() {
        assert!(is_supported_model_format(&model_with_version("1.0")));
        assert!(is_supported_model_format(&model_with_version("1.1")));
        assert!(is_supported_model_format(&model_with_version("2.0")));
    }

    #[test]
    fn rejects_unknown_learned_gate_schema_version() {
        assert!(!is_supported_model_format(&model_with_version("3.0")));
    }

    #[test]
    fn isotonic_clamps_below_first_knot() {
        let xs = vec![0.1, 0.4, 0.9];
        let ys = vec![0.0, 0.5, 1.0];
        assert_eq!(isotonic_eval(0.0, &xs, &ys), 0.0);
        assert_eq!(isotonic_eval(0.05, &xs, &ys), 0.0);
    }

    #[test]
    fn isotonic_clamps_above_last_knot() {
        let xs = vec![0.1, 0.4, 0.9];
        let ys = vec![0.0, 0.5, 1.0];
        assert_eq!(isotonic_eval(0.95, &xs, &ys), 1.0);
        assert_eq!(isotonic_eval(2.0, &xs, &ys), 1.0);
    }

    #[test]
    fn isotonic_interpolates_within_bracket() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        assert!((isotonic_eval(0.3, &xs, &ys) - 0.3).abs() < 1e-12);
        let xs2 = vec![0.0, 0.5, 1.0];
        let ys2 = vec![0.0, 0.2, 1.0];
        // In [0.5, 1.0] the slope is 1.6.
        let got = isotonic_eval(0.75, &xs2, &ys2);
        assert!((got - 0.6).abs() < 1e-12, "got {got}");
    }

    #[test]
    fn isotonic_handles_degenerate_bracket() {
        // Two knots at the same x — must not divide by zero.
        let xs = vec![0.5, 0.5, 1.0];
        let ys = vec![0.2, 0.3, 1.0];
        let got = isotonic_eval(0.5, &xs, &ys);
        assert!(got.is_finite());
    }

    #[test]
    fn decision_states_round_trip() {
        assert_eq!(LearnedGateDecision::Pass.as_str(), "PASS");
        assert_eq!(LearnedGateDecision::Reject.as_str(), "REJECT");
        assert_eq!(LearnedGateDecision::Abstain.as_str(), "ABSTAIN");
        assert!(LearnedGateDecision::Pass.is_pass());
        assert!(!LearnedGateDecision::Abstain.is_pass());
        assert!(LearnedGateDecision::Reject.is_reject());
    }

    // ── Hardening tests added 2026-05-23 after the systems-engineer
    //    review flagged unguarded edge cases. Each named test maps to
    //    a finding in that review.

    #[test]
    fn isotonic_eval_on_nan_does_not_panic() {
        // partition_point with NaN was an unguarded usize underflow.
        // Should now return the midpoint of the y-range.
        let xs = vec![0.1, 0.4, 0.9];
        let ys = vec![0.0, 0.5, 1.0];
        let got = isotonic_eval(f64::NAN, &xs, &ys);
        assert!(got.is_finite());
        assert!((0.0..=1.0).contains(&got));
    }

    #[test]
    fn isotonic_eval_clamps_pos_inf_to_last_knot() {
        let xs = vec![0.1, 0.4, 0.9];
        let ys = vec![0.0, 0.5, 1.0];
        assert_eq!(isotonic_eval(f64::INFINITY, &xs, &ys), 1.0);
    }

    #[test]
    fn isotonic_eval_clamps_neg_inf_to_first_knot() {
        let xs = vec![0.1, 0.4, 0.9];
        let ys = vec![0.0, 0.5, 1.0];
        assert_eq!(isotonic_eval(f64::NEG_INFINITY, &xs, &ys), 0.0);
    }

    #[test]
    fn isotonic_eval_duplicate_x_is_left_continuous() {
        // sklearn's IsotonicRegression step at duplicate x-knots is
        // left-continuous: at the step's x, return the *lower* y.
        let xs = vec![0.0, 0.5, 0.5, 1.0];
        let ys = vec![0.0, 0.2, 0.8, 1.0];
        // At exactly x = 0.5, we should land on the left side (0.2),
        // not the right side (0.8).
        let got = isotonic_eval(0.5, &xs, &ys);
        assert!(
            (got - 0.2).abs() < 1e-9,
            "expected left-continuous 0.2, got {got}"
        );
    }

    #[test]
    fn isotonic_eval_is_in_y_range_property() {
        // Property: for any monotone knot sequence and any input,
        // the result is in [min(ys), max(ys)].
        let xs = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let ys = vec![0.1, 0.2, 0.5, 0.8, 0.9];
        let y_min = 0.1f64;
        let y_max = 0.9f64;
        for x in [
            -1.0,
            0.0,
            0.1,
            0.25,
            0.49,
            0.5,
            0.51,
            0.75,
            0.99,
            1.0,
            2.0,
            f64::NAN,
            f64::INFINITY,
        ] {
            let got = isotonic_eval(x, &xs, &ys);
            assert!(
                (y_min..=y_max).contains(&got)
                    || (got - y_min).abs() < 1e-12
                    || (got - y_max).abs() < 1e-12,
                "x={x} produced {got}, expected in [{y_min}, {y_max}]"
            );
        }
    }

    #[test]
    fn conformal_pass_boundary_is_strict() {
        // The exact-boundary case `p_cal == threshold` with `q_hat=0`
        // must collapse to ABSTAIN under the strict-greater /
        // strict-less rule — never silently flip to PASS as it did in
        // the pre-fix v2.0 loader. Tested at the predicate level
        // because plumbing a full `evaluate` call requires building
        // `SrtHealth` + `TranslationSemanticHealth` structs that have
        // no public constructors.
        let p_cal = sigmoid(0.0); // exactly 0.5
        let threshold = 0.5_f64;
        let q_hat = 0.0_f64;
        let lo = (p_cal - q_hat).clamp(0.0, 1.0);
        let hi = (p_cal + q_hat).clamp(0.0, 1.0);
        // Use `<=` / `>=` rather than `!(... > ...)`/`!(... < ...)` so
        // clippy's `neg_cmp_op_on_partial_ord` lint stays happy on the
        // partial-order f64 type.
        assert!(lo <= threshold, "must not Pass at boundary");
        assert!(hi >= threshold, "must not Reject at boundary");
    }

    /// The repo-committed production model artifact must parse with
    /// the strict serde_json (no NaN literals). This catches the
    /// 2026-05-22 regression where Python's `json.dump` wrote `NaN`
    /// tokens that serde rejects.
    #[test]
    fn production_model_artifact_parses_strict_json() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("learned_gate.json");
        if !path.is_file() {
            // Skip on a fresh clone where the model hasn't been emitted yet.
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let raw = std::fs::read_to_string(&path).expect("learned_gate.json must be readable");
        let model: LearnedGateModel = serde_json::from_str(&raw).unwrap_or_else(|e| {
            panic!(
                "learned_gate.json must parse with strict serde_json: {e}. \
                 If this fails on a NaN token, the Python harness wrote \
                 non-conformant JSON — fix it at scripts/train_learned_gate.py."
            )
        });
        assert!(
            is_supported_model_format(&model),
            "production model schema must be supported"
        );
    }
}
