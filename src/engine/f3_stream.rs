//! F.3 streaming counterfactual MI estimator.
//!
//! Online counterpart to the batch kNN estimator in
//! `scripts/audit_learned_gate_features.py`. Uses a binned histogram
//! approach to estimate `I(X_S; Y)` — the mutual information between
//! masked features and the gate label — in `O(1)` per update with
//! `O(B * 2)` memory per feature.
//!
//! # Design
//!
//! The batch audit uses sklearn's kNN MI estimator (Ross 2014) which
//! requires a stored point cloud. That's infeasible for streaming.
//! Instead, we discretize each scalar feature into `B` equal-width bins
//! (quantiles frozen after a warmup phase) and estimate MI from the
//! joint count table `counts[bin][label]`.
//!
//! For a fully masked feature (one that doesn't exist at inference),
//! the masked MI is 0 by definition (the feature is constant-zero).
//! So `ΔI̅ = I_rich - I_masked = I_rich - 0 = I_rich` for those features.
//! The aggregate `ΔI̅` is the sum over masked features.
//!
//! This aggregate `ΔI̅` is exactly the `F3Sample::delta_i` that
//! `LfasScheduler` consumes.

use crate::engine::lfas::F3Sample;

/// Number of histogram bins per feature. 8 is a good default for
/// binary-label MI estimation: fine enough to capture non-linear
/// relationships, coarse enough that counts don't fragment.
const DEFAULT_BINS: usize = 8;

/// Minimum observations before the estimator produces a sample.
/// Below this, the histogram is too sparse for reliable MI.
const MIN_OBSERVATIONS: u32 = 30;

/// Per-feature streaming MI estimator using a binned histogram.
///
/// Maintains a `B × 2` count table (B bins × 2 labels: PASS/REJECT).
#[derive(Debug, Clone)]
struct FeatureMiEstimator {
    /// Joint counts: `counts[bin][label]`. `label` is 0 (REJECT) or 1 (PASS).
    counts: [[u32; 2]; DEFAULT_BINS],
    /// Total observations in the count table.
    total: u32,
    /// Bin edges (B+1 values). Frozen after warmup.
    edges: [f64; DEFAULT_BINS + 1],
    /// Whether edges have been frozen.
    edges_frozen: bool,
    /// Warmup buffer: (value, label) pairs. Replayed into counts after
    /// edges are frozen so no observations are lost.
    warmup: Vec<(f64, bool)>,
}

impl FeatureMiEstimator {
    fn new() -> Self {
        Self {
            counts: [[0; 2]; DEFAULT_BINS],
            total: 0,
            edges: [0.0; DEFAULT_BINS + 1],
            edges_frozen: false,
            warmup: Vec::with_capacity(MIN_OBSERVATIONS as usize),
        }
    }

    /// Record an observation: `(feature_value, label)`.
    /// `label`: `true` = PASS, `false` = REJECT.
    fn observe(&mut self, value: f64, label: bool) {
        if !self.edges_frozen {
            self.warmup.push((value, label));
            if self.warmup.len() >= MIN_OBSERVATIONS as usize {
                self.freeze_edges();
                self.edges_frozen = true;
                // Replay all warmup observations into the count table.
                let warmup = std::mem::take(&mut self.warmup);
                for (v, l) in warmup {
                    let bin = self.bin_index(v);
                    self.counts[bin][usize::from(l)] += 1;
                    self.total += 1;
                }
            }
            return;
        }

        let bin = self.bin_index(value);
        self.counts[bin][usize::from(label)] += 1;
        self.total += 1;
    }

    /// Compute the bin edges from the warmup buffer using equal-width binning.
    fn freeze_edges(&mut self) {
        if self.warmup.is_empty() {
            return;
        }
        let mut values: Vec<f64> = self.warmup.iter().map(|(v, _)| *v).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = values[0];
        let max_val = values[values.len() - 1];

        // Equal-width bins spanning the warmup range with margin.
        // Use ±10% margin so future values outside the warmup range
        // still land in sensible bins.
        let range = (max_val - min_val).max(1e-10);
        let margin = range * 0.1;
        let effective_min = min_val - margin;
        let effective_range = range + 2.0 * margin;
        let step = effective_range / DEFAULT_BINS as f64;

        for i in 0..=DEFAULT_BINS {
            self.edges[i] = effective_min + step * i as f64;
        }
        // Ensure the last edge is well above max_val.
        self.edges[DEFAULT_BINS] = max_val + margin + 1e-10;
    }

    /// Map a value to a bin index in `[0, B)`.
    fn bin_index(&self, value: f64) -> usize {
        for i in 0..DEFAULT_BINS {
            if value < self.edges[i + 1] {
                return i;
            }
        }
        DEFAULT_BINS - 1
    }

    /// Estimate the squared Hellinger distance H²(P(X|Y=1), P(X|Y=0))
    /// between the feature's conditional distributions given each label.
    ///
    /// This is computable directly from the binned histogram without kNN.
    /// H²(P, Q) = 1 − Σ_b √(P(b) · Q(b)), which is the Bhattacharyya
    /// coefficient subtracted from 1.
    ///
    /// Used by the Rényi-Hellinger sharpening (Theorem 4) to provide a
    /// tighter coverage bound than the standard C-BHC KL-based penalty.
    fn estimate_hellinger_squared(&self) -> Option<f64> {
        if self.total < MIN_OBSERVATIONS {
            return None;
        }

        // Marginal counts per label.
        let mut n_pass = 0u32;
        let mut n_reject = 0u32;
        for bin in &self.counts {
            n_pass += bin[1];
            n_reject += bin[0];
        }
        if n_pass == 0 || n_reject == 0 {
            // One class has zero samples: H² = 1 (maximally separated) if
            // any bin has data in the other class; but this is degenerate.
            return Some(1.0);
        }

        let n_p = f64::from(n_pass);
        let n_r = f64::from(n_reject);

        // Bhattacharyya coefficient: BC = Σ_b √(P(b|Y=1) · P(b|Y=0))
        let mut bc = 0.0;
        for bin in &self.counts {
            let p_pass = f64::from(bin[1]) / n_p;
            let p_reject = f64::from(bin[0]) / n_r;
            bc += (p_pass * p_reject).sqrt();
        }

        // H² = 1 − BC
        Some((1.0 - bc).clamp(0.0, 1.0))
    }

    /// Estimate `I(X; Y)` from the current counts.
    ///
    /// Returns `None` if insufficient data.
    fn estimate_mi(&self) -> Option<f64> {
        if self.total < MIN_OBSERVATIONS {
            return None;
        }

        let n = f64::from(self.total);
        let mut mi = 0.0;

        // Marginal label counts.
        let mut label_counts = [0u32; 2];
        for bin in &self.counts {
            label_counts[0] += bin[0];
            label_counts[1] += bin[1];
        }

        for bin in &self.counts {
            let bin_total: u32 = bin[0] + bin[1];
            if bin_total == 0 {
                continue;
            }

            for label in 0..2 {
                let joint = bin[label];
                if joint == 0 {
                    continue;
                }

                let p_xy = f64::from(joint) / n;
                let p_x = f64::from(bin_total) / n;
                let p_y = f64::from(label_counts[label]) / n;

                if p_x > 0.0 && p_y > 0.0 {
                    mi += p_xy * (p_xy / (p_x * p_y)).ln();
                }
            }
        }

        // MI should be non-negative; clamp numerical noise.
        Some(mi.max(0.0))
    }
}

/// Streaming F.3 counterfactual MI estimator.
///
/// Tracks MI for a set of features flagged as "masked" (unavailable at
/// inference). The aggregate `ΔI̅ = Σ_i I(X_i; Y)` for masked features
/// is the signal that feeds `LfasScheduler`.
///
/// # Usage
///
/// ```ignore
/// let mut estimator = F3StreamEstimator::new(&["delta_time_coverage_ratio",
///     "delta_weighted_token_f1"]);
/// // After each gate evaluation:
/// estimator.observe(&feature_values, gate_decision_is_pass);
/// if let Some(sample) = estimator.sample() {
///     doom_qlock.record_f3_audit(&prepared, sample);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct F3StreamEstimator {
    /// Per-feature estimators, indexed by masked feature position.
    features: Vec<FeatureMiEstimator>,
    /// Names of the masked features (for diagnostics).
    feature_names: Vec<String>,
    /// Total observations fed.
    observation_count: u32,
}

impl F3StreamEstimator {
    /// Create a new streaming estimator for the given masked feature names.
    ///
    /// `masked_features` lists the features that are unavailable at
    /// inference time (the same set flagged by the batch F.3 audit).
    pub fn new(masked_features: &[&str]) -> Self {
        Self {
            features: (0..masked_features.len())
                .map(|_| FeatureMiEstimator::new())
                .collect(),
            feature_names: masked_features.iter().map(|s| s.to_string()).collect(),
            observation_count: 0,
        }
    }

    /// Feed an observation: the masked feature values and the gate decision.
    ///
    /// `masked_values` must be the same length as `masked_features` passed
    /// to `new`. Each value is the feature's raw value (before masking).
    /// `is_pass` is the gate's binary decision (`true` = PASS, `false` = REJECT).
    pub fn observe(&mut self, masked_values: &[f64], is_pass: bool) {
        assert_eq!(
            masked_values.len(),
            self.features.len(),
            "masked_values length ({}) != feature count ({})",
            masked_values.len(),
            self.features.len()
        );

        for (estimator, &value) in self.features.iter_mut().zip(masked_values.iter()) {
            estimator.observe(value, is_pass);
        }
        self.observation_count += 1;
    }

    /// Produce an `F3Sample` if sufficient data has been collected.
    ///
    /// Returns `None` until `MIN_OBSERVATIONS` have been seen (the
    /// histogram needs enough counts for reliable MI estimation).
    ///
    /// The returned `delta_i` is the aggregate counterfactual MI:
    /// `ΔI̅ = Σ_i I(X_i; Y)` over all masked features. For fully
    /// masked features (inference-time value is 0), the masked MI is 0,
    /// so `ΔI̅ = I_rich` — the MI that evaporates under masking.
    pub fn sample(&self) -> Option<F3Sample> {
        if self.observation_count < MIN_OBSERVATIONS {
            return None;
        }

        let mut total_mi = 0.0;
        let mut total_h_sq = 0.0;
        let mut all_ready = true;
        let mut h_sq_available = true;

        for estimator in &self.features {
            match estimator.estimate_mi() {
                Some(mi) => total_mi += mi,
                None => {
                    all_ready = false;
                }
            }
            match estimator.estimate_hellinger_squared() {
                Some(h_sq) => total_h_sq += h_sq,
                None => {
                    h_sq_available = false;
                }
            }
        }

        if !all_ready {
            return None;
        }

        // Aggregate H²: for independent features, the total squared
        // Hellinger distance is bounded by the sum of per-feature H²
        // (union bound on the product measure). Clamp to [0, 1].
        let h_squared = if h_sq_available {
            Some(total_h_sq.clamp(0.0, 1.0))
        } else {
            None
        };

        Some(F3Sample {
            delta_i: total_mi,
            h_squared,
        })
    }

    /// Number of observations recorded so far.
    pub fn observation_count(&self) -> u32 {
        self.observation_count
    }

    /// Whether the estimator has enough data to produce samples.
    pub fn is_ready(&self) -> bool {
        self.sample().is_some()
    }

    /// Per-feature MI breakdown for diagnostics.
    pub fn per_feature_mi(&self) -> Vec<(&str, Option<f64>)> {
        self.feature_names
            .iter()
            .zip(self.features.iter())
            .map(|(name, est)| (name.as_str(), est.estimate_mi()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_needs_warmup() {
        let mut est = F3StreamEstimator::new(&["feat_a", "feat_b"]);
        // Before MIN_OBSERVATIONS, sample() returns None.
        for i in 0..MIN_OBSERVATIONS - 1 {
            est.observe(&[i as f64, 0.0], i % 2 == 0);
            assert!(est.sample().is_none());
        }
    }

    #[test]
    fn estimator_produces_sample_after_warmup() {
        let mut est = F3StreamEstimator::new(&["feat_a"]);
        // Feed data where feat_a perfectly predicts the label.
        // Label=PASS when feat_a > 5, REJECT when feat_a <= 5.
        for i in 0..100 {
            let value = i as f64;
            let is_pass = value > 50.0;
            est.observe(&[value], is_pass);
        }
        let sample = est.sample();
        assert!(
            sample.is_some(),
            "should produce sample after 100 observations"
        );
        let mi = sample.unwrap().delta_i;
        // With perfect prediction, MI should be > 0.
        assert!(
            mi > 0.0,
            "MI should be positive for predictive feature, got {mi}"
        );
    }

    #[test]
    fn zero_mi_for_independent_feature() {
        let mut est = F3StreamEstimator::new(&["noise"]);
        // Feature is constant — no information about label.
        for i in 0..200 {
            est.observe(&[42.0], i % 2 == 0);
        }
        let sample = est.sample();
        assert!(sample.is_some());
        let mi = sample.unwrap().delta_i;
        // Constant feature → all counts in one bin → MI ≈ 0.
        assert!(mi < 0.01, "MI should be ~0 for constant feature, got {mi}");
    }

    #[test]
    fn per_feature_breakdown() {
        let mut est = F3StreamEstimator::new(&["good", "noise"]);
        for i in 0..200 {
            let val = i as f64;
            let is_pass = val > 100.0;
            est.observe(&[val, 42.0], is_pass);
        }
        let breakdown = est.per_feature_mi();
        assert_eq!(breakdown.len(), 2);
        assert_eq!(breakdown[0].0, "good");
        assert_eq!(breakdown[1].0, "noise");
        // "good" should have higher MI than "noise".
        let mi_good = breakdown[0].1.unwrap_or(0.0);
        let mi_noise = breakdown[1].1.unwrap_or(0.0);
        assert!(
            mi_good > mi_noise,
            "predictive feature MI ({mi_good}) should exceed noise MI ({mi_noise})"
        );
    }
}
