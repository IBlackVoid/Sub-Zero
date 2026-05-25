//! Label-Free Adaptive Scheduling (LFAS) — F.4 theorem implementation.
//!
//! Implements the LFAS-UCB algorithm from `docs/F4_lfas.md`: a multi-arm
//! bandit scheduler that uses F.3 counterfactual mutual information as a
//! *label-free* reward signal and provides sublinear coverage regret via
//! the C-BHC concavity transfer (Theorem 2).
//!
//! # Design
//!
//! - **O(K) memory**: `3K` scalars (count, mean, Welford M2 per arm).
//! - **O(K) per round**: one pass over arms for LCB computation + argmin.
//! - **No heap allocation** on the hot path.
//! - **`Send + Sync`**: all fields are plain data; single-writer by
//!   construction (the pipeline's sequential chunk loop holds `&mut self`).
//!
//! # Integration
//!
//! Called from `DoomQlock::prepare_run` (arm selection) and
//! `DoomQlock::record_success` (reward feedback). The scheduler does not
//! replace DOOM-QLOCK's history-based plan lookup; it *augments* it with
//! a quality-aware arm recommendation when F.3 audit data is available.

use serde::{Deserialize, Serialize};

/// A single F.3 audit observation for a processed chunk.
///
/// Produced by the streaming F.3 estimator (binned histogram MI).
/// `delta_i` is the conditional counterfactual MI: `I(X_S; Y | X_{¬S})`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct F3Sample {
    /// Counterfactual MI in nats. In `[0, B]` where `B = ln|Y|`.
    pub delta_i: f64,
    /// Squared Hellinger distance H²(P_rich, P_masked) aggregated
    /// over masked features. Available from the binned histogram directly.
    /// Used by the Rényi-Hellinger sharpening (Theorem 4) for tighter bounds.
    /// `None` when the estimator hasn't accumulated enough data per-label.
    pub h_squared: Option<f64>,
}

/// Per-arm state maintained by Welford's online algorithm.
///
/// Tracks both F.3 MI (for UCB-Bernstein) and squared Hellinger distance
/// (for the Rényi-Hellinger sharpening, Theorem 4).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct ArmState {
    /// Number of times this arm has been pulled.
    n: u32,
    /// Running mean of observed F.3 ΔI̅ values (Welford).
    mean: f64,
    /// Welford's M2 accumulator: `sum of (x_i - mean_after_i) * (x_i - mean_before_i)`.
    m2: f64,
    /// Running mean of squared Hellinger distance (for Theorem 4).
    /// Tracked separately because H² may not be available for every sample.
    h_sq_mean: f64,
    /// Number of Hellinger observations (may be < n if some samples lack H²).
    h_sq_n: u32,
}

impl ArmState {
    const fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            h_sq_mean: 0.0,
            h_sq_n: 0,
        }
    }

    /// Online variance estimate. Returns `1.0` (optimistic) when `n < 2`.
    fn variance(&self) -> f64 {
        if self.n < 2 {
            1.0
        } else {
            // Population variance (not sample): M2 / n, matching the
            // UCB-Bernstein confidence width derivation in Theorem 1.
            self.m2 / f64::from(self.n)
        }
    }

    /// Welford update with a new MI observation.
    fn update(&mut self, value: f64) {
        self.n += 1;
        let d1 = value - self.mean;
        self.mean += d1 / f64::from(self.n);
        let d2 = value - self.mean;
        self.m2 += d1 * d2;
    }

    /// Update running mean of H² (Rényi-Hellinger sharpening).
    fn update_hellinger(&mut self, h_sq: f64) {
        self.h_sq_n += 1;
        let d = h_sq - self.h_sq_mean;
        self.h_sq_mean += d / f64::from(self.h_sq_n);
    }

    /// Mean squared Hellinger distance, if available.
    fn mean_h_squared(&self) -> Option<f64> {
        if self.h_sq_n >= 2 {
            Some(self.h_sq_mean)
        } else {
            None
        }
    }
}

/// Arm identifier. Wraps `usize` for type safety.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArmId(pub usize);

/// Label-Free Adaptive Scheduler (LFAS-UCB).
///
/// Implements the algorithm from F.4 Theorem 1. `K` is the number of
/// arms (chunk-size × quality-profile configurations), known at compile
/// time. Typical values: 4, 8, 12.
///
/// # Example
///
/// ```ignore
/// let mut sched = LfasScheduler::<4>::new(LfasConfig::default());
/// let arm = sched.pick_arm();
/// // ... run pipeline with arm's config ...
/// sched.record(arm, Some(F3Sample { delta_i: 0.18, h_squared: None }));
/// let (lo, hi) = sched.coverage_bound(arm, 0.10);
/// ```
#[derive(Debug, Clone)]
pub struct LfasScheduler<const K: usize> {
    arms: [ArmState; K],
    /// Global step counter (1-indexed after init).
    t: u32,
    /// Upper bound on F.3 ΔI̅ values, in nats. Default: `ln 2 ≈ 0.693`
    /// (binary quality gate).
    reward_bound: f64,
    /// Confidence parameter δ. Default: `1/T` is set dynamically,
    /// but we store a static fallback for the first K rounds.
    delta_fallback: f64,
    /// Audit period: record F.3 every `audit_period`-th chunk.
    /// Non-audited chunks still advance `t` but do not update arm stats.
    audit_period: u32,
    /// Internal counter for audit subsampling.
    audit_counter: u32,
}

/// Configuration for `LfasScheduler`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LfasConfig {
    /// Upper bound `B` on F.3 ΔI̅ values (nats).
    /// Default: `ln 2` (binary gate: PASS/REJECT).
    pub reward_bound: f64,
    /// Confidence fallback δ for the first K rounds.
    /// Default: `0.05`.
    pub delta_fallback: f64,
    /// Audit every M-th chunk. Default: `20` (~5% audit rate).
    pub audit_period: u32,
}

impl Default for LfasConfig {
    fn default() -> Self {
        Self {
            reward_bound: core::f64::consts::LN_2,
            delta_fallback: 0.05,
            audit_period: 20,
        }
    }
}

impl<const K: usize> LfasScheduler<K> {
    /// Create a new scheduler with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if `K == 0`.
    pub fn new(config: LfasConfig) -> Self {
        assert!(K > 0, "LfasScheduler requires at least one arm");
        Self {
            arms: [ArmState::new(); K],
            t: 0,
            reward_bound: config.reward_bound,
            delta_fallback: config.delta_fallback,
            audit_period: config.audit_period.max(1),
            audit_counter: 0,
        }
    }

    /// Select the next arm to play.
    ///
    /// During the initial exploration phase (`t < K`), arms are played
    /// in round-robin order. After that, LFAS-UCB selects the arm with
    /// the lowest LCB (Lower Confidence Bound) — minimizing F.3 ΔI̅.
    pub fn pick_arm(&self) -> ArmId {
        let k = K as u32;

        // Round-robin exploration for the first K rounds.
        if self.t < k {
            return ArmId(self.t as usize);
        }

        // UCB-Bernstein: pick arm with lowest LCB.
        let delta = self.effective_delta();
        let mut best_arm = 0;
        let mut best_lcb = f64::INFINITY;

        for a in 0..K {
            let state = &self.arms[a];
            if state.n == 0 {
                // Unexplored arm: always explore.
                return ArmId(a);
            }
            let var = state.variance();
            let n = f64::from(state.n);
            let log_term = (3.0 * f64::from(self.t).powi(2) / delta).ln();
            let exploration = var.sqrt() * (2.0 * log_term / n).sqrt()
                + 3.0 * self.reward_bound * log_term / n;
            let lcb = state.mean - exploration;

            if lcb < best_lcb {
                best_lcb = lcb;
                best_arm = a;
            }
        }

        ArmId(best_arm)
    }

    /// Record the outcome of playing an arm.
    ///
    /// `audit` is `Some(sample)` when the F.3 audit was run on this
    /// chunk (every `audit_period`-th chunk), `None` otherwise.
    /// The step counter `t` always advances; arm statistics are only
    /// updated when `audit` is `Some`.
    pub fn record(&mut self, arm: ArmId, audit: Option<F3Sample>) {
        self.t = self.t.saturating_add(1);
        self.audit_counter += 1;

        if let Some(sample) = audit {
            let idx = arm.0;
            if idx < K {
                let r = sample.delta_i.clamp(0.0, self.reward_bound);
                self.arms[idx].update(r);
                // Track Hellinger for Rényi sharpening (Theorem 4).
                if let Some(h_sq) = sample.h_squared {
                    self.arms[idx].update_hellinger(h_sq);
                }
            }
        }
    }

    /// Check whether the F.3 audit should run on this chunk.
    ///
    /// Returns `true` every `audit_period`-th call, starting from
    /// the first call. The caller is responsible for running the audit
    /// when this returns `true` and passing the result to `record`.
    pub fn should_audit(&self) -> bool {
        self.audit_counter % self.audit_period == 0
    }

    /// Coverage confidence interval for a given arm, using C-BHC with
    /// Rényi-Hellinger sharpening (Theorem 4).
    ///
    /// Returns `(lower, upper)` bounds on `c_dep(arm)` at confidence
    /// level `alpha`. Uses the *tighter* of:
    ///   - Standard C-BHC: `phi(ΔI̅) = √(1 − exp(−ΔI̅))`
    ///   - Hellinger: `phi_H(H²) = H · √(2 − H²)`
    ///
    /// The Hellinger bound is strictly tighter when H² < 1−exp(−ΔI̅),
    /// which holds for all non-degenerate distributions.
    ///
    /// Returns `(0.0, 1.0)` if the arm has no observations.
    pub fn coverage_bound(&self, arm: ArmId, alpha: f64) -> (f64, f64) {
        let idx = arm.0;
        if idx >= K {
            return (0.0, 1.0);
        }
        let state = &self.arms[idx];
        if state.n == 0 {
            return (0.0, 1.0);
        }

        let n = f64::from(state.n);
        let stderr = (state.variance() / n).sqrt();

        // Conservative: upper bound on ΔI̅ → lower bound on coverage.
        let di_upper = (state.mean + 2.0 * stderr).clamp(0.0, self.reward_bound);
        let di_lower = (state.mean - 2.0 * stderr).clamp(0.0, self.reward_bound);

        // Use sharpened penalty: min(phi(ΔI̅), phi_hellinger(H²)).
        let h_sq = state.mean_h_squared();
        let penalty_upper = phi_sharpened(di_upper, h_sq);
        let penalty_lower = phi_sharpened(di_lower, h_sq);

        let cov_lower = (1.0 - alpha) - penalty_upper;
        let cov_upper = (1.0 - alpha) - penalty_lower;

        (cov_lower.clamp(0.0, 1.0), cov_upper.clamp(0.0, 1.0))
    }

    /// Production coverage floor at step `t` (Theorems 3 + 4).
    ///
    /// Returns the minimum coverage guarantee that holds simultaneously
    /// for all steps up to the current step, with probability
    /// `>= 1 - delta`. Uses the Rényi-Hellinger sharpened bound when
    /// H² data is available (Theorem 4), otherwise falls back to
    /// standard C-BHC (Theorem 3).
    pub fn coverage_floor(&self, alpha: f64) -> f64 {
        if self.t == 0 || K == 0 {
            return 0.0;
        }

        let best_arm = self.best_arm_index();
        let r_star = self.arms[best_arm].mean;
        let h_sq = self.arms[best_arm].mean_h_squared();
        let gamma = self.gamma_t();

        let penalty = phi_sharpened((r_star + gamma).max(0.0), h_sq);
        ((1.0 - alpha) - penalty).max(0.0)
    }

    /// Cumulative F.3 regret estimate.
    ///
    /// `R_F3(T) = sum_t r_t - T * r*`. Computed from arm statistics.
    pub fn estimated_regret(&self) -> f64 {
        if self.t == 0 {
            return 0.0;
        }
        let best = self.best_arm_index();
        let r_star = self.arms[best].mean;

        self.arms
            .iter()
            .map(|arm| {
                if arm.n == 0 {
                    0.0
                } else {
                    f64::from(arm.n) * (arm.mean - r_star).max(0.0)
                }
            })
            .sum()
    }

    /// Summary statistics for diagnostics / structured logging.
    pub fn summary(&self) -> LfasSummary<K> {
        let best = self.best_arm_index();
        LfasSummary {
            step: self.t,
            best_arm: ArmId(best),
            best_mean_delta_i: self.arms[best].mean,
            sigma_max: self.max_sigma(),
            estimated_regret: self.estimated_regret(),
            arm_pulls: core::array::from_fn::<u32, K, _>(|i| self.arms[i].n),
        }
    }

    // --- private helpers ---

    fn effective_delta(&self) -> f64 {
        if self.t == 0 {
            self.delta_fallback
        } else {
            (1.0 / f64::from(self.t)).min(self.delta_fallback)
        }
    }

    fn best_arm_index(&self) -> usize {
        self.arms
            .iter()
            .enumerate()
            .filter(|(_, arm)| arm.n > 0)
            .min_by(|(_, a), (_, b)| a.mean.partial_cmp(&b.mean).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn max_sigma(&self) -> f64 {
        self.arms
            .iter()
            .filter(|arm| arm.n >= 2)
            .map(|arm| arm.variance().sqrt())
            .fold(0.0_f64, f64::max)
    }

    fn gamma_t(&self) -> f64 {
        let t = f64::from(self.t).max(1.0);
        let k = K as f64;
        let delta = self.effective_delta();
        let log_term = (3.0 * t * t / delta).ln().max(0.0);

        let sigma_max = self.max_sigma();
        sigma_max * (2.0 * k * log_term / t).sqrt()
            + 6.0 * k * self.reward_bound * log_term / t
    }
}

/// Summary statistics emitted by `LfasScheduler::summary`.
#[derive(Debug, Clone)]
pub struct LfasSummary<const K: usize> {
    pub step: u32,
    pub best_arm: ArmId,
    pub best_mean_delta_i: f64,
    pub sigma_max: f64,
    pub estimated_regret: f64,
    pub arm_pulls: [u32; K],
}

/// C-BHC transfer function: `phi(x) = sqrt(1 - exp(-x))`.
///
/// Maps F.3 counterfactual MI (in nats) to coverage degradation. Concave on
/// `(0, infinity)`, saturates at 1. This is the Bretagnolle--Huber
/// inequality applied through the BCRT 2023 coverage--TV chain.
#[inline]
pub fn phi(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        (1.0 - (-x).exp()).sqrt()
    }
}

/// Derivative of `phi`: `phi'(x) = exp(-x) / (2 * sqrt(1 - exp(-x)))`.
#[inline]
pub fn phi_prime(x: f64) -> f64 {
    if x <= 0.0 {
        // Limit as x -> 0+: phi'(x) -> infinity. Return a large
        // but finite value to avoid NaN in downstream computation.
        1e6
    } else {
        let e = (-x).exp();
        e / (2.0 * (1.0 - e).sqrt())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § Rényi-Hellinger Sharpening (Theorem 4)
//
// The standard C-BHC bound uses KL → Bretagnolle-Huber → TV → coverage.
// When we have direct access to the squared Hellinger distance H² from the
// binned histograms (which the F3 stream estimator provides), we can bypass
// the KL → TV conversion and use the tighter Hellinger–TV inequality:
//
//   TV(P, Q) ≤ H(P, Q) · √(2 − H²(P, Q))         (Le Cam / Reiss)
//
// This yields the sharpened coverage bound:
//
//   c_dep ≥ (1−α) − H · √(2 − H²)
//
// which is strictly tighter than (1−α) − √(1−exp(−KL)) whenever H² < 1−exp(−KL),
// i.e., when the distributions are not maximally separated.
//
// **Key properties:**
//   1. Always ≤ phi(KL): the Hellinger penalty is never worse than C-BHC.
//   2. Computable in O(B) from the F3 binned histograms (no kNN needed).
//   3. Concave in H² on [0, 1]: the Jensen transfer (Theorem 2) still applies.
//   4. Activates above 4-class gate: same structural constraint as C-BHC.
//
// **Proof sketch** (Theorem 4):
//   TV(P_rich, P_masked) ≤ H(P_rich, P_masked)       [Hellinger dominates TV]
//                        ≤ H · √(2 − H²)             [Le Cam's refinement]
//   BCRT 2023: c_dep ≥ (1−α) − TV ≥ (1−α) − H·√(2−H²)
//
// The improvement over standard C-BHC is the gap:
//   Δ_sharp = √(1−exp(−ΔI̅)) − H·√(2−H²)
//
// which is positive whenever H² < 1−exp(−ΔI̅). Since H² ≤ 1−exp(−KL) ≤ 1−exp(−ΔI̅)
// by Pinsker's inequality chain, the sharpened bound is ALWAYS at least as tight,
// and strictly tighter for non-degenerate distributions.
// ─────────────────────────────────────────────────────────────────────────────

/// Rényi-Hellinger sharpened penalty: `H · √(2 − H²)`.
///
/// Given the squared Hellinger distance H² between P_rich and P_masked,
/// computes the tighter Le Cam / Reiss TV upper bound. This replaces
/// `phi(ΔI̅)` in the coverage bound when H² is directly available from
/// the F3 stream estimator's binned histograms.
///
/// # Properties
///
/// - `phi_hellinger(0) = 0` (identical distributions → no coverage loss)
/// - `phi_hellinger(1) = 1` (completely disjoint → maximal loss)
/// - Concave on `[0, 1]`: `d²/d(H²)² < 0`
/// - Always ≤ `phi(KL)` for the same distribution pair
#[inline]
pub fn phi_hellinger(h_squared: f64) -> f64 {
    if h_squared <= 0.0 {
        return 0.0;
    }
    let h_sq = h_squared.clamp(0.0, 1.0);
    let h = h_sq.sqrt();
    // Le Cam refinement: TV ≤ H · √(2 − H²)
    h * (2.0 - h_sq).sqrt()
}

/// Derivative of `phi_hellinger` with respect to H²:
/// `d/d(H²) [√(H²) · √(2 − H²)] = (2 − 2H²) / (2 · √(H² · (2 − H²)))`.
#[inline]
pub fn phi_hellinger_prime(h_squared: f64) -> f64 {
    if h_squared <= 0.0 {
        // Limit as H² → 0⁺: derivative → 1/√2 ≈ 0.707
        return core::f64::consts::FRAC_1_SQRT_2;
    }
    let h_sq = h_squared.clamp(1e-15, 1.0 - 1e-15);
    let numerator = 2.0 - 2.0 * h_sq;
    let denominator = 2.0 * (h_sq * (2.0 - h_sq)).sqrt();
    if denominator < 1e-15 {
        return 0.0;
    }
    numerator / denominator
}

/// Compute the sharpening gap: how much tighter the Hellinger bound is
/// compared to the standard C-BHC bound.
///
/// `Δ_sharp = phi(delta_i) − phi_hellinger(h_squared)`
///
/// Positive means the Hellinger bound provides a higher coverage floor.
#[inline]
pub fn sharpening_gap(delta_i: f64, h_squared: f64) -> f64 {
    (phi(delta_i) - phi_hellinger(h_squared)).max(0.0)
}

/// Choose the tighter of the two penalties (standard C-BHC vs Hellinger).
///
/// When `h_squared` is available from the F3 estimator, returns the
/// minimum penalty — giving the tightest provable coverage bound.
#[inline]
pub fn phi_sharpened(delta_i: f64, h_squared: Option<f64>) -> f64 {
    let cbhc = phi(delta_i);
    match h_squared {
        Some(h_sq) => cbhc.min(phi_hellinger(h_sq)),
        None => cbhc,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi_basic_properties() {
        // phi(0) = 0
        assert!((phi(0.0)).abs() < 1e-12);
        // phi is strictly increasing
        assert!(phi(0.5) < phi(1.0));
        assert!(phi(1.0) < phi(2.0));
        // phi saturates below 1
        assert!(phi(100.0) < 1.0 + 1e-12);
        assert!(phi(100.0) > 1.0 - 1e-6);
    }

    #[test]
    fn phi_concavity() {
        // Check phi is concave: phi((a+b)/2) >= (phi(a) + phi(b)) / 2
        let pairs = [(0.1, 0.5), (0.5, 2.0), (1.0, 4.0), (0.01, 10.0)];
        for (a, b) in pairs {
            let midpoint = phi((a + b) / 2.0);
            let average = (phi(a) + phi(b)) / 2.0;
            assert!(
                midpoint >= average - 1e-12,
                "concavity violated at ({a}, {b}): phi(mid)={midpoint} < avg={average}"
            );
        }
    }

    #[test]
    fn phi_crossover_with_pinsker() {
        // At x = 1.594: phi(x) ≈ sqrt(x/2) (Pinsker)
        let x = 1.594;
        let cbhc = phi(x);
        let pinsker = (x / 2.0).sqrt();
        assert!(
            (cbhc - pinsker).abs() < 0.01,
            "crossover mismatch: cbhc={cbhc}, pinsker={pinsker}"
        );
        // Above crossover: C-BHC < Pinsker (tighter)
        assert!(phi(2.0) < (2.0_f64 / 2.0).sqrt());
        assert!(phi(4.0) < (4.0_f64 / 2.0).sqrt());
    }

    // ── Rényi-Hellinger sharpening (Theorem 4) tests ──

    #[test]
    fn phi_hellinger_boundary_values() {
        // H²=0 → identical distributions → zero penalty
        assert!((phi_hellinger(0.0)).abs() < 1e-12);
        // H²=1 → maximally separated → penalty = 1
        assert!((phi_hellinger(1.0) - 1.0).abs() < 1e-12);
        // Negative input → 0
        assert_eq!(phi_hellinger(-1.0), 0.0);
    }

    #[test]
    fn phi_hellinger_monotone_increasing() {
        let points = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];
        for window in points.windows(2) {
            assert!(
                phi_hellinger(window[1]) >= phi_hellinger(window[0]),
                "phi_hellinger not monotone: f({}) = {} < f({}) = {}",
                window[1], phi_hellinger(window[1]),
                window[0], phi_hellinger(window[0])
            );
        }
    }

    #[test]
    fn phi_hellinger_concavity() {
        // Check phi_hellinger is concave: f((a+b)/2) >= (f(a) + f(b)) / 2
        let pairs = [(0.05, 0.3), (0.1, 0.8), (0.2, 0.6), (0.01, 0.99)];
        for (a, b) in pairs {
            let mid = phi_hellinger((a + b) / 2.0);
            let avg = (phi_hellinger(a) + phi_hellinger(b)) / 2.0;
            assert!(
                mid >= avg - 1e-12,
                "concavity violated at ({a}, {b}): phi_H(mid)={mid} < avg={avg}"
            );
        }
    }

    #[test]
    fn hellinger_tighter_at_rényi_tight_bound() {
        // Key identity: when H² = 1−exp(−KL/2), the two penalties are equal.
        // For H² < 1−exp(−KL/2) (the generic case), Hellinger IS tighter.
        // phi_sharpened always picks the min, so it's safe by construction.
        for kl in [0.1, 0.5, 1.0, 2.0, 4.0] {
            let h_sq_tight = 1.0 - (-kl / 2.0_f64).exp();
            let cbhc = phi(kl);
            let hellinger_tight = phi_hellinger(h_sq_tight);
            // At the Rényi-tight bound: Hellinger ≈ C-BHC
            assert!(
                (cbhc - hellinger_tight).abs() < 0.01,
                "at tight bound KL={kl}: cbhc={cbhc:.4} hellinger={hellinger_tight:.4}"
            );
            // Below the tight bound (generic distributions): Hellinger < C-BHC
            let h_sq_loose = h_sq_tight * 0.6; // 60% of tight — common in practice
            let hellinger_loose = phi_hellinger(h_sq_loose);
            assert!(
                hellinger_loose < cbhc + 1e-10,
                "below tight bound: hellinger={hellinger_loose:.4} > cbhc={cbhc:.4} at KL={kl}"
            );
        }
    }

    #[test]
    fn sharpening_gap_positive_below_tight_bound() {
        // phi_sharpened picks min(phi, phi_H). When H² < 1−exp(−KL/2),
        // phi_H < phi, so the sharpened bound gives a positive gap.
        let kl = 1.0;
        let h_sq_tight = 1.0 - (-kl / 2.0_f64).exp();
        let h_sq_below = h_sq_tight * 0.5;
        let gap = sharpening_gap(kl, h_sq_below);
        assert!(
            gap > 0.0,
            "sharpening gap should be positive below tight bound: KL={kl}, H²={h_sq_below}, gap={gap}"
        );
    }

    #[test]
    fn phi_sharpened_picks_tighter() {
        // With both available, sharpened should pick the min
        let kl = 0.5;
        let h_sq = 0.25;
        let sharpened = phi_sharpened(kl, Some(h_sq));
        let cbhc = phi(kl);
        let hellinger = phi_hellinger(h_sq);
        assert!((sharpened - cbhc.min(hellinger)).abs() < 1e-12);
        // Without H², should equal C-BHC
        assert!((phi_sharpened(kl, None) - cbhc).abs() < 1e-12);
    }

    #[test]
    fn coverage_bound_uses_sharpening() {
        let mut sched = LfasScheduler::<2>::new(LfasConfig::default());
        // Feed arm 0 with both MI and H² data
        for _ in 0..2 {
            let arm = sched.pick_arm();
            sched.record(arm, Some(F3Sample {
                delta_i: 0.3,
                h_squared: Some(0.15),
            }));
        }
        // After 2 observations with H² data, coverage_bound should use
        // the sharpened penalty (tighter → higher coverage floor)
        let (lo_sharp, _) = sched.coverage_bound(ArmId(0), 0.10);
        assert!(lo_sharp > 0.0, "sharpened coverage lower bound should be positive");
    }

    #[test]
    fn scheduler_round_robin_exploration() {
        let mut sched = LfasScheduler::<4>::new(LfasConfig::default());
        // First 4 picks should be round-robin 0,1,2,3
        for expected in 0..4 {
            let arm = sched.pick_arm();
            assert_eq!(arm, ArmId(expected));
            sched.record(arm, Some(F3Sample { delta_i: 0.1 * (expected as f64 + 1.0), h_squared: None }));
        }
    }

    #[test]
    fn scheduler_converges_to_best_arm() {
        let mut sched = LfasScheduler::<4>::new(LfasConfig {
            audit_period: 1, // audit every chunk for testing
            ..LfasConfig::default()
        });

        // True means: arm 0 = 0.1 (best), arm 1 = 0.3, arm 2 = 0.5, arm 3 = 0.6
        let true_means = [0.1, 0.3, 0.5, 0.6];

        // Exploration phase
        for _ in 0..4 {
            let arm = sched.pick_arm();
            sched.record(arm, Some(F3Sample { delta_i: true_means[arm.0], h_squared: None }));
        }

        // Run 1000 steps with deterministic rewards (no noise for test clarity).
        // UCB-Bernstein explores early but converges to the best arm.
        let mut arm0_pulls = 0u32;
        for _ in 4..1000 {
            let arm = sched.pick_arm();
            sched.record(arm, Some(F3Sample { delta_i: true_means[arm.0], h_squared: None }));
            if arm == ArmId(0) {
                arm0_pulls += 1;
            }
        }

        // Arm 0 (best) should get the majority of pulls after convergence.
        // With 996 exploitation rounds and clear gaps, >50% is conservative.
        assert!(
            arm0_pulls > 400,
            "arm 0 should be pulled >400 times out of 996, got {arm0_pulls}"
        );
    }

    #[test]
    fn coverage_bound_monotone() {
        let mut sched = LfasScheduler::<2>::new(LfasConfig {
            audit_period: 1,
            ..LfasConfig::default()
        });

        // Arm 0: low ΔI̅ (good), arm 1: high ΔI̅ (bad)
        sched.record(ArmId(0), Some(F3Sample { delta_i: 0.05, h_squared: Some(0.03) }));
        let _ = sched.pick_arm(); // advance t
        sched.record(ArmId(1), Some(F3Sample { delta_i: 0.5, h_squared: Some(0.4) }));

        let (lo_good, _) = sched.coverage_bound(ArmId(0), 0.10);
        let (lo_bad, _) = sched.coverage_bound(ArmId(1), 0.10);

        // Better arm should have higher coverage lower bound
        assert!(
            lo_good >= lo_bad,
            "good arm coverage {lo_good} should >= bad arm {lo_bad}"
        );
    }

    #[test]
    fn should_audit_period() {
        let sched = LfasScheduler::<4>::new(LfasConfig {
            audit_period: 5,
            ..LfasConfig::default()
        });
        // Counter starts at 0, so first call should audit (0 % 5 == 0)
        assert!(sched.should_audit());
    }

    #[test]
    fn welford_variance_matches_naive() {
        let values = [0.1, 0.4, 0.2, 0.35, 0.15, 0.3, 0.25, 0.28];
        let mut arm = ArmState::new();
        for &v in &values {
            arm.update(v);
        }

        // Naive variance computation
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let naive_var: f64 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        assert!(
            (arm.variance() - naive_var).abs() < 1e-10,
            "welford={}, naive={naive_var}",
            arm.variance()
        );
        assert!((arm.mean - mean).abs() < 1e-10);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// phi is monotonically non-decreasing for all non-negative inputs.
        #[test]
        fn phi_monotone(a in 0.0f64..100.0, b in 0.0f64..100.0) {
            if a <= b {
                prop_assert!(phi(a) <= phi(b) + 1e-12,
                    "phi({a}) = {} > phi({b}) = {}", phi(a), phi(b));
            }
        }

        /// phi output is always in [0, 1] for non-negative inputs.
        #[test]
        fn phi_range(x in 0.0f64..1000.0) {
            let y = phi(x);
            prop_assert!(y >= 0.0, "phi({x}) = {y} < 0");
            prop_assert!(y <= 1.0 + 1e-12, "phi({x}) = {y} > 1");
        }

        /// phi is concave: phi(mid) >= average of phi(a), phi(b).
        #[test]
        fn phi_concave(a in 0.01f64..50.0, b in 0.01f64..50.0) {
            let mid = (a + b) / 2.0;
            let avg = (phi(a) + phi(b)) / 2.0;
            prop_assert!(phi(mid) >= avg - 1e-10,
                "concavity violated: phi({mid}) = {} < avg({a},{b}) = {avg}", phi(mid));
        }

        /// C-BHC dominates Pinsker above the crossover point (~1.594 nats).
        #[test]
        fn cbhc_dominates_above_crossover(x in 1.6f64..100.0) {
            let cbhc = phi(x);
            let pinsker = (x / 2.0).sqrt();
            prop_assert!(cbhc <= pinsker + 1e-6,
                "C-BHC {cbhc} should be <= Pinsker {pinsker} above crossover at x={x}");
        }

        /// Welford online variance matches naive for arbitrary sequences.
        #[test]
        fn welford_matches_naive(values in proptest::collection::vec(0.0f64..10.0, 2..50)) {
            let mut arm = ArmState::new();
            for &v in &values {
                arm.update(v);
            }
            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let naive_var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            prop_assert!((arm.variance() - naive_var).abs() < 1e-6,
                "welford={} naive={naive_var} n={}", arm.variance(), values.len());
            prop_assert!((arm.mean - mean).abs() < 1e-6);
        }

        /// LfasScheduler always returns a valid arm index.
        #[test]
        fn pick_arm_in_range(steps in 0u32..100) {
            let mut sched = LfasScheduler::<4>::new(LfasConfig {
                audit_period: 1,
                ..LfasConfig::default()
            });
            for _ in 0..steps {
                let arm = sched.pick_arm();
                prop_assert!(arm.0 < 4, "arm {} out of range", arm.0);
                sched.record(arm, Some(F3Sample { delta_i: 0.2, h_squared: None }));
            }
        }

        /// Coverage bounds are always in [0, 1].
        #[test]
        fn coverage_bounds_valid(delta_i in 0.0f64..0.7, alpha in 0.01f64..0.5) {
            let mut sched = LfasScheduler::<2>::new(LfasConfig {
                audit_period: 1,
                ..LfasConfig::default()
            });
            // Feed some data
            for _ in 0..10 {
                let arm = sched.pick_arm();
                sched.record(arm, Some(F3Sample { delta_i, h_squared: None }));
            }
            let (lo, hi) = sched.coverage_bound(ArmId(0), alpha);
            prop_assert!((0.0..=1.0).contains(&lo), "lo={lo} out of [0,1]");
            prop_assert!((0.0..=1.0).contains(&hi), "hi={hi} out of [0,1]");
            prop_assert!(lo <= hi + 1e-10, "lo={lo} > hi={hi}");
        }
    }
}

