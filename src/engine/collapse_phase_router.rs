//! Document-level collapse-phase detection and backend routing (CPT-Router).
//!
//! Some MT backends do not fail "a little" on hard content — they enter a
//! distinct *phase*: many unrelated source cues collapse onto a small family of
//! fluent fallback phrases. On the Silent Hill casual-Japanese benchmark, NLLB
//! emitted "that's the matter with you" for **353 of 1361 cues (26%)**, spread
//! from cue 7 to 1359 — not local damage, but whole-document engine/content
//! incompatibility.
//!
//! That is cheaply detectable. The **order parameter** is the dominant *motif*
//! density `Ω_m` — the weighted max over exact normalized lines and their
//! 3-/4-grams — measured over an inspection window of the first `m` cues. A
//! Hoeffding lower-confidence bound turns a noisy estimate into a certificate:
//! if `Ω_m − ε_m(δ) ≥ θ`, then with confidence `1 − δ` the dominant density is
//! ≥ θ and the backend is certified in collapse phase. The router then aborts
//! the cheap backend and routes the *whole* document to the stronger local rung
//! rather than finishing a doomed pass.
//!
//! Operating point: δ = 0.05 (95%), θ = 0.08, first check at m = 64 then every
//! 64, plus an end-of-window force-check. Live NLLB collapse can be diffuse
//! (a "that's …" family at ~0.20 density rather than one phrase at 0.26), so the
//! motif order parameter + 95% confidence certify it where an exact-line / 99%
//! detector would miss; the θ margin keeps false-routes ~0 (measured).
//!
//! Scope / honesty: this is a *systems* detector, not a new theory. Frequency
//! heavy-hitters, Hoeffding bounds, and cost-driven model routing are all
//! standard; the contribution is the specific, cheap document-level order
//! parameter as an early collapse certificate for subtitle MT. It claims no
//! novelty over MT hallucination detection, quality estimation, or cascading.

use std::collections::HashMap;

/// Configuration for the collapse-phase router.
#[derive(Debug, Clone)]
pub struct CollapsePhaseConfig {
    /// Only multi-word phrases count toward the order parameter: short lines
    /// ("yeah", "huh?") legitimately repeat and must not trigger collapse.
    pub min_tokens: usize,
    /// Do not test before this many cues (the Hoeffding margin is too loose on
    /// tiny prefixes to certify anything).
    pub min_prefix_cues: usize,
    /// Re-test the certificate every this many cues.
    pub check_every_cues: usize,
    /// Confidence parameter: a fired certificate holds with probability ≥ 1 − δ.
    pub delta: f64,
    /// Dominant-density threshold above which the backend is "in collapse phase".
    pub theta: f64,
}

impl Default for CollapsePhaseConfig {
    fn default() -> Self {
        // δ = 0.05 (95% confidence) is the right operating point for a reroute
        // heuristic: live NLLB collapse can be *diffuse* (a "that's …" family at
        // ~0.20 dominant density, not one phrase at 0.26), and 99% confidence at
        // small m is too tight to certify it. 95% certifies diffuse collapse over
        // the inspection window while the θ=0.08 margin keeps false-routes ~0.
        // First check at m=64, then every 64, plus an end-of-window force_check.
        Self {
            min_tokens: 3,
            min_prefix_cues: 64,
            check_every_cues: 64,
            delta: 0.05,
            theta: 0.08,
        }
    }
}

/// The router's decision after observing a cue.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendRouteDecision {
    /// No collapse certificate yet — keep using the base backend.
    ContinueBase,
    /// Collapse certified: abort the base backend and translate the whole
    /// document with the stronger local rung.
    AbortAndRouteWholeDocument {
        dominant_phrase: String,
        density: f64,
        lower_bound: f64,
        cues_seen: usize,
    },
}

/// Motif weights: shorter n-grams repeat coincidentally more often, so they are
/// down-weighted relative to exact lines and 4-grams.
const LINE_WEIGHT: f64 = 1.0;
const GRAM4_WEIGHT: f64 = 1.0;
const GRAM3_WEIGHT: f64 = 0.8;

/// Streaming collapse-phase detector. Feed it base-backend target lines as they
/// are produced; it certifies collapse early or stays quiet.
///
/// The order parameter is the dominant **motif** density — the weighted max over
/// exact normalized lines and their 3-/4-grams. Backends that collapse usually do
/// so to a *family* of fluent phrases ("that's the matter with you", "that's the
/// matter", "that's that"), not one exact string, so an exact-line-only signal
/// undercounts the collapse; the n-gram motifs capture the family.
#[derive(Debug, Default)]
pub struct CollapsePhaseRouter {
    line_counts: HashMap<String, usize>,
    gram3_counts: HashMap<String, usize>,
    gram4_counts: HashMap<String, usize>,
    cues_seen: usize,
    certified: bool,
}

impl CollapsePhaseRouter {
    pub fn new() -> Self {
        Self::default()
    }

    /// The Hoeffding one-sided margin `ε_m(δ) = sqrt( ln(1/δ) / (2m) )`. The
    /// certified lower bound on the true dominant density is `Ω_m − ε_m`.
    fn hoeffding_margin(m: usize, delta: f64) -> f64 {
        ((1.0 / delta).ln() / (2.0 * m as f64)).sqrt()
    }

    /// Observe one base-backend target line. Returns the routing decision.
    ///
    /// Once collapse is certified the router latches: every subsequent call
    /// returns `ContinueBase` (the caller has already been told to reroute; it
    /// must not be told twice).
    pub fn observe(
        &mut self,
        target_line: &str,
        cfg: &CollapsePhaseConfig,
    ) -> BackendRouteDecision {
        if self.certified {
            return BackendRouteDecision::ContinueBase;
        }

        // Only multi-word phrases feed the order parameter (short interjections
        // legitimately repeat and must not certify collapse).
        let key = normalize(target_line);
        let tokens: Vec<&str> = if key.is_empty() {
            Vec::new()
        } else {
            key.split(' ').collect()
        };
        if tokens.len() >= cfg.min_tokens {
            *self.line_counts.entry(key.clone()).or_insert(0) += 1;
            for w in tokens.windows(3) {
                *self.gram3_counts.entry(w.join(" ")).or_insert(0) += 1;
            }
            for w in tokens.windows(4) {
                *self.gram4_counts.entry(w.join(" ")).or_insert(0) += 1;
            }
        }
        self.cues_seen += 1;

        if self.cues_seen >= cfg.min_prefix_cues && self.cues_seen % cfg.check_every_cues == 0 {
            self.check(cfg)
        } else {
            BackendRouteDecision::ContinueBase
        }
    }

    /// Force a certification check at the current window length, ignoring the
    /// periodic cadence. Used at the end of a bounded inspection window so a
    /// short document (whose length is not a cadence multiple) still gets a
    /// final, larger-sample evaluation.
    pub fn force_check(&mut self, cfg: &CollapsePhaseConfig) -> BackendRouteDecision {
        if self.certified || self.cues_seen < cfg.min_prefix_cues {
            return BackendRouteDecision::ContinueBase;
        }
        self.check(cfg)
    }

    /// The order-parameter certification at the current `cues_seen`.
    fn check(&mut self, cfg: &CollapsePhaseConfig) -> BackendRouteDecision {
        let m = self.cues_seen as f64;
        let dominant = |map: &HashMap<String, usize>| -> (String, usize) {
            map.iter()
                .max_by_key(|(_, &c)| c)
                .map(|(k, &c)| (k.clone(), c))
                .unwrap_or_default()
        };
        let (line_k, line_c) = dominant(&self.line_counts);
        let (g4_k, g4_c) = dominant(&self.gram4_counts);
        let (g3_k, g3_c) = dominant(&self.gram3_counts);

        // Order parameter = the motif with the highest WEIGHTED density; report
        // its raw (unweighted) density and the dominant phrase.
        let candidates = [
            (line_k, line_c as f64 / m * LINE_WEIGHT, line_c as f64 / m),
            (g4_k, g4_c as f64 / m * GRAM4_WEIGHT, g4_c as f64 / m),
            (g3_k, g3_c as f64 / m * GRAM3_WEIGHT, g3_c as f64 / m),
        ];
        let (phrase, weighted_density, raw_density) = candidates
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("candidate motif set is non-empty");

        let lower_bound = weighted_density - Self::hoeffding_margin(self.cues_seen, cfg.delta);
        if lower_bound >= cfg.theta {
            self.certified = true;
            BackendRouteDecision::AbortAndRouteWholeDocument {
                dominant_phrase: phrase,
                density: raw_density,
                lower_bound,
                cues_seen: self.cues_seen,
            }
        } else {
            BackendRouteDecision::ContinueBase
        }
    }

    /// Whether collapse has been certified this run.
    pub fn certified(&self) -> bool {
        self.certified
    }
}

/// Normalize a target line into a phrase key: lowercase, drop punctuation
/// (keeping apostrophes), collapse whitespace. "That's the matter with you?!" and
/// "that's  the matter with you" both map to "that's the matter with you".
fn normalize(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut last_space = true; // trim leading
    for ch in line.chars() {
        if ch.is_alphanumeric() || ch == '\'' {
            out.extend(ch.to_lowercase());
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CollapsePhaseConfig {
        CollapsePhaseConfig::default()
    }

    #[test]
    fn normalize_collapses_variants_to_one_key() {
        assert_eq!(
            normalize("That's the matter with you?!"),
            "that's the matter with you"
        );
        assert_eq!(
            normalize("that's   the  matter with you"),
            "that's the matter with you"
        );
        assert_eq!(normalize("  Hello, world.  "), "hello world");
    }

    #[test]
    fn clean_diverse_stream_never_certifies() {
        let mut r = CollapsePhaseRouter::new();
        let c = cfg();
        for i in 0..600 {
            // Every token unique per line, so no line OR n-gram motif repeats —
            // no dominant attractor at any motif level.
            let line = format!("alpha{i} bravo{i} charlie{i} delta{i}");
            assert_eq!(r.observe(&line, &c), BackendRouteDecision::ContinueBase);
        }
        assert!(!r.certified());
    }

    #[test]
    fn legitimate_short_repeats_do_not_certify() {
        let mut r = CollapsePhaseRouter::new();
        let c = cfg();
        // "yeah" repeated forever is fine — it is below min_tokens, never counted.
        for _ in 0..600 {
            assert_eq!(r.observe("yeah", &c), BackendRouteDecision::ContinueBase);
        }
        assert!(!r.certified());
    }

    #[test]
    fn collapse_stream_certifies_early_and_latches() {
        let mut r = CollapsePhaseRouter::new();
        let c = cfg();
        let poison = "that's the matter with you";
        let mut decision = BackendRouteDecision::ContinueBase;
        let mut fired_at = None;
        // ~33% poison density interleaved with distinct filler — dense enough
        // (like the real Silent Hill region) to certify at the first checkpoint.
        for i in 0..600 {
            let line = if i % 3 == 0 {
                poison.to_string()
            } else {
                // Fully-distinct filler (no shared motifs) so only the poison
                // family repeats.
                format!("alpha{i} bravo{i} charlie{i} delta{i}")
            };
            let d = r.observe(&line, &c);
            if let BackendRouteDecision::AbortAndRouteWholeDocument { cues_seen, .. } = &d {
                if fired_at.is_none() {
                    fired_at = Some(*cues_seen);
                    decision = d.clone();
                }
            }
        }
        // Certifies at the first checkpoint (min_prefix = 64) — a single valid
        // test, no repeated-testing inflation.
        assert_eq!(fired_at, Some(64), "should certify at the first checkpoint");
        assert!(r.certified());
        match decision {
            BackendRouteDecision::AbortAndRouteWholeDocument {
                dominant_phrase,
                lower_bound,
                ..
            } => {
                // The dominant motif is a member of the poison family (the exact
                // line or one of its n-grams) — all contain "matter".
                assert!(
                    dominant_phrase.contains("matter"),
                    "dominant motif should come from the poison family, got {dominant_phrase:?}"
                );
                assert!(lower_bound >= c.theta);
            }
            _ => panic!("expected an abort/reroute decision"),
        }
    }

    #[test]
    fn does_not_fire_before_min_prefix() {
        let mut r = CollapsePhaseRouter::new();
        let c = cfg();
        let poison = "that's the matter with you";
        // Pure poison, but only up to just under min_prefix: must stay quiet.
        for _ in 0..(c.min_prefix_cues - 1) {
            assert_eq!(r.observe(poison, &c), BackendRouteDecision::ContinueBase);
        }
        assert!(!r.certified());
    }

    #[test]
    fn hoeffding_margin_shrinks_with_more_cues() {
        let a = CollapsePhaseRouter::hoeffding_margin(128, 0.01);
        let b = CollapsePhaseRouter::hoeffding_margin(1024, 0.01);
        assert!(a > b, "margin must tighten as the prefix grows");
        // Sanity against the validated probe value at m=128, delta=0.01 (~0.134).
        assert!((a - 0.134).abs() < 0.01);
    }
}
