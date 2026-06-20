// Per-segment MT escalation ladder.
//
// A "segment" here is a scene (see `scenes.rs`). When a scene's translation
// falls below its quality floor, this ladder retries ONLY that scene against a
// progressively stronger MT backend, bounded by a per-profile policy and a
// session-wide budget cap.
//
// Policy rationale + evidence: see agent-memory finding_ja_en_casual_mt_quality
// (JA→EN casual MT limit). The benchmark established that escalating 600M→1.3B
// repairs systematic mistranslation on *some* content (the "What→That's" case)
// but does NOT rescue genuinely casual short-utterance speech — 1.3B degenerates
// differently. So escalation must be bounded: profile-gated (#1), per-segment
// (#2), budget-capped (#3 — if >30% of segments need it, the content class is
// NLLB-incompatible and more compute will not help), one-attempt-per-rung (#4 —
// a segment may climb 600M→1.3B→LLM but never re-attempts the same rung),
// telemetered (#5), and expressed as an ordered backend list (#6) so the
// LLM-based backend is one more entry rather than new branch surgery.
//
// The orchestration is deliberately decoupled from NLLB via the `SegmentBackend`
// seam: production wires the real neural translator, tests inject a scripted
// fake. This makes the ladder logic deterministic and CI-safe.

use crate::engine::srt::SubtitleCue;
use crate::engine::transcribe::QualityProfile;

/// Fraction of segments that may be escalated before the budget cap trips.
/// Above this, the content class is treated as NLLB-incompatible (guardrail #3).
pub(super) const ESCALATION_BUDGET_RATIO: f64 = 0.30;

/// Which translation engine drives a rung. Lets the ladder carry an LLM rung as
/// one more entry (guardrail #6) while the backend routes by kind instead of by
/// sniffing the model name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BackendKind {
    /// NLLB neural MT (the 600M / 1.3B rungs).
    Nllb,
    /// Local LLM MT via the llama-server sidecar (the Qwen3 rung). Constructed
    /// by the pipeline only when the binary + model resolve; otherwise the
    /// ladder stays NLLB-only.
    Llm,
}

/// One rung of the escalation ladder (guardrail #6). Adding a future backend
/// (e.g. an LLM-based MT) is a single entry here — no new control flow.
#[derive(Debug, Clone)]
pub(super) struct MtBackendStep {
    /// Stable identifier used in telemetry (e.g. "nllb-600M", "nllb-1.3B").
    pub(super) label: &'static str,
    /// Concrete MT model name handed to the backend.
    pub(super) model_name: String,
    /// Beam width for this rung (a higher beam is a knob, not a new branch).
    pub(super) beam_size: usize,
    /// Which engine this rung drives. The real backend router (`NeuralSegmentBackend`)
    /// branches on this to pick the NLLB translator or the llama-server sidecar.
    pub(super) kind: BackendKind,
}

/// The full ladder plus the per-profile reach into it (guardrail #1).
#[derive(Debug, Clone)]
pub(super) struct EscalationPolicy {
    /// Rung 0 is the base model already used for the document; rungs 1.. are
    /// the escalation targets, in strictly increasing strength.
    steps: Vec<MtBackendStep>,
    /// How many rungs *beyond rung 0* a profile may walk.
    max_escalations: usize,
    /// Whether a still-failing final rung is fatal (Strict) or downgraded to
    /// best-effort (Fast/Balanced). Drives guardrail #1's hard-fail tail.
    hard_fail_on_exhaustion: bool,
}

impl EscalationPolicy {
    /// Build the escalation ladder gated by profile (guardrail #1).
    ///
    /// - Fast:     never escalates (max_escalations = 0).
    /// - Balanced: 600M → 1.3B (→ LLM if supplied), best-effort if still failing.
    /// - Strict:   600M → 1.3B (→ LLM if supplied), hard-fail tail.
    ///
    /// `llm_rung` is the optional Qwen3 tier. When `Some`, Balanced/Strict gain
    /// one more rung to walk to (`max_escalations` rises to 2); when `None` the
    /// ladder is byte-for-byte the historical NLLB-only ladder, so a build
    /// without the llama-server sidecar behaves exactly as before.
    pub(super) fn for_profile(
        profile: QualityProfile,
        base_beam: usize,
        llm_rung: Option<MtBackendStep>,
    ) -> Self {
        let base = MtBackendStep {
            label: "nllb-600M",
            model_name: "nllb-200-distilled-600M".to_string(),
            beam_size: base_beam,
            kind: BackendKind::Nllb,
        };
        let escalated = MtBackendStep {
            label: "nllb-1.3B",
            model_name: "nllb-200-distilled-1.3B".to_string(),
            beam_size: (base_beam + 2).min(8),
            kind: BackendKind::Nllb,
        };

        if matches!(profile, QualityProfile::Fast) {
            return EscalationPolicy {
                steps: vec![base],
                max_escalations: 0,
                hard_fail_on_exhaustion: false,
            };
        }

        // Balanced and Strict share the same ladder; they differ only in the
        // hard-fail tail (guardrail #1).
        let mut steps = vec![base, escalated];
        let mut max_escalations = 1;
        if let Some(llm) = llm_rung {
            steps.push(llm);
            max_escalations = 2;
        }
        EscalationPolicy {
            steps,
            max_escalations,
            hard_fail_on_exhaustion: matches!(profile, QualityProfile::Strict),
        }
    }

    pub(super) fn can_escalate(&self) -> bool {
        self.max_escalations > 0 && self.steps.len() > 1
    }

    /// Guardrail #1 tail: when true (Strict), a scene that escalated and still
    /// failed makes the whole job fatal; when false (Fast/Balanced) it is
    /// downgraded to best-effort.
    pub(super) fn hard_fail_on_exhaustion(&self) -> bool {
        self.hard_fail_on_exhaustion
    }

    /// The escalation rungs a segment may walk, in increasing strength: rung 1
    /// up to `max_escalations` rungs beyond the base. A segment tries each in
    /// turn and stops at the first that recovers (guardrail #4 is now "one
    /// attempt *per rung*", not "one attempt total").
    fn escalation_rungs(&self) -> &[MtBackendStep] {
        let upper = (1 + self.max_escalations).min(self.steps.len());
        if upper <= 1 {
            &[]
        } else {
            &self.steps[1..upper]
        }
    }
}

/// The backend seam (guardrail / testability): the ladder asks a backend to
/// translate one segment with a given rung. Production wires NLLB; tests inject
/// a scripted fake. Returning the translated cues lets the orchestrator apply
/// the *real* scene-quality + semantic-penalty safeguards uniformly.
pub(super) trait SegmentBackend {
    fn translate_segment(
        &self,
        step: &MtBackendStep,
        source: &[SubtitleCue],
    ) -> Result<Vec<SubtitleCue>, String>;
}

/// Derives `(scene_score, semantic_penalty)` for a translated segment. Production
/// uses the real scene-quality logic; tests script scores per segment so the
/// ladder decisions are deterministic without running a model.
pub(super) trait SceneScorer {
    fn score(&self, translated: &[SubtitleCue]) -> (f64, f64);
}

/// A segment that scored below its floor at the base rung and is a candidate
/// for escalation.
#[derive(Debug, Clone, Copy)]
pub(super) struct FailingSegment {
    /// Scene index (0-based) — the `segment` field in telemetry.
    pub(super) index: usize,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) base_score: f64,
    pub(super) floor: f64,
}

/// Per-segment outcome after the ladder runs.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum SegmentOutcome {
    /// Escalated and the stronger rung passed the floor and beat the base score.
    Recovered { score_after: f64 },
    /// Escalated once, still failed; downgraded to best-effort. No further retries.
    BestEffortExhausted { score_after: f64 },
    /// Not escalated because the per-profile reach is zero (e.g. Fast).
    SkippedByPolicy,
    /// Not escalated because the session budget cap (guardrail #3) tripped first.
    SkippedByBudget,
}

/// A telemetry record emitted on every escalation *attempt* (guardrail #5).
/// Serialized to the exact `mt_escalation` schema by `as_event`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct EscalationEvent {
    pub(super) segment: usize,
    pub(super) from: &'static str,
    pub(super) to: &'static str,
    pub(super) recovered: bool,
    pub(super) gate_score_before: f64,
    pub(super) gate_score_after: f64,
}

impl EscalationEvent {
    /// The exact JSONL shape consumed by the TUI / trace stream.
    pub(super) fn as_event(&self) -> serde_json::Value {
        serde_json::json!({
            "event": "mt_escalation",
            "segment": self.segment,
            "from": self.from,
            "to": self.to,
            "reason": "quality_gate",
            "outcome": if self.recovered { "recovered" } else { "still_failed" },
            "gate_score_before": self.gate_score_before,
            "gate_score_after": self.gate_score_after,
        })
    }
}

/// Result of running the ladder over all failing segments.
#[derive(Debug, Clone)]
pub(super) struct EscalationReport {
    /// Per-segment outcomes, in input order.
    pub(super) outcomes: Vec<(usize, SegmentOutcome)>,
    /// Telemetry, one per actual escalation attempt (guardrail #5).
    pub(super) events: Vec<EscalationEvent>,
    /// Recovered segments paired with the text to splice back into the document.
    pub(super) recovered_texts: Vec<RecoveredSegment>,
    /// True if the budget cap tripped mid-document (guardrail #3 signal).
    pub(super) budget_exceeded: bool,
}

#[derive(Debug, Clone)]
pub(super) struct RecoveredSegment {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) texts: Vec<String>,
}

/// Run the per-segment escalation ladder.
///
/// `total_segments` is the document's full scene count, used for the budget
/// ratio (guardrail #3). `failing` must already be the below-floor segments
/// (sorted by the caller; processing order is preserved). `current` is the
/// already-translated document, sliced per segment to score the base rung.
pub(super) fn run_escalation_ladder<B, S>(
    policy: &EscalationPolicy,
    backend: &B,
    scorer: &S,
    source_cues: &[SubtitleCue],
    current: &[SubtitleCue],
    failing: &[FailingSegment],
    total_segments: usize,
) -> EscalationReport
where
    B: SegmentBackend,
    S: SceneScorer,
{
    let mut report = EscalationReport {
        outcomes: Vec::with_capacity(failing.len()),
        events: Vec::new(),
        recovered_texts: Vec::new(),
        budget_exceeded: false,
    };

    // Guardrail #1: a profile with no reach never escalates.
    let rungs = policy.escalation_rungs();
    if rungs.is_empty() || !policy.can_escalate() {
        for segment in failing {
            report
                .outcomes
                .push((segment.index, SegmentOutcome::SkippedByPolicy));
        }
        return report;
    }
    let base_label = policy.steps[0].label;

    // Guardrail #3: cap the number of segments allowed to escalate this session.
    // At least one escalation is always permitted when there is work to do — a
    // single bad scene in an otherwise-good document is precisely the case the
    // ladder exists to repair; `floor(n * 0.30)` alone would round to 0 for any
    // document under ~4 scenes and silently disable the feature on short inputs.
    let budget = if total_segments == 0 {
        0
    } else {
        (((total_segments as f64) * ESCALATION_BUDGET_RATIO).floor() as usize).max(1)
    };

    let mut escalated = 0usize;
    for segment in failing {
        if escalated >= budget {
            report.budget_exceeded = true;
            report
                .outcomes
                .push((segment.index, SegmentOutcome::SkippedByBudget));
            continue;
        }
        escalated += 1;

        let expected = segment.end - segment.start;
        let (_base_again, base_penalty) = scorer.score(&current[segment.start..segment.end]);

        // Walk the rungs in increasing strength; stop at the first that recovers
        // (guardrail #4: one attempt *per rung*, never a re-attempt of the same
        // rung). `from` tracks the rung being climbed from, so telemetry reads
        // 600M→1.3B then 1.3B→LLM.
        let mut from = base_label;
        let mut last_score = segment.base_score;
        let mut seg_recovered = false;

        for rung in rungs {
            let retried =
                match backend.translate_segment(rung, &source_cues[segment.start..segment.end]) {
                    Ok(cues) => cues,
                    Err(_) => {
                        // Backend failure on this rung: record it and climb on; the
                        // same rung is never re-attempted.
                        report.events.push(EscalationEvent {
                            segment: segment.index,
                            from,
                            to: rung.label,
                            recovered: false,
                            gate_score_before: segment.base_score,
                            gate_score_after: last_score,
                        });
                        from = rung.label;
                        continue;
                    }
                };

            // Cue-count-mismatch safeguard: a mismatched retry is rejected
            // (treated as still-failing), not spliced in.
            let (score_after, recovered) = if retried.len() == expected {
                let (retry_score, retry_penalty) = scorer.score(&retried);
                // Acceptance: strictly beats BOTH the original base score and the
                // floor, and does not regress the semantic penalty.
                let ok = retry_score > segment.base_score + 0.01
                    && retry_score >= segment.floor
                    && retry_penalty <= base_penalty + 0.01;
                (retry_score, ok)
            } else {
                (segment.base_score, false)
            };

            report.events.push(EscalationEvent {
                segment: segment.index,
                from,
                to: rung.label,
                recovered,
                gate_score_before: segment.base_score,
                gate_score_after: score_after,
            });
            last_score = score_after;

            if recovered {
                report.recovered_texts.push(RecoveredSegment {
                    start: segment.start,
                    end: segment.end,
                    texts: retried.into_iter().map(|cue| cue.text).collect(),
                });
                seg_recovered = true;
                break;
            }
            from = rung.label;
        }

        if seg_recovered {
            report.outcomes.push((
                segment.index,
                SegmentOutcome::Recovered {
                    score_after: last_score,
                },
            ));
        } else {
            report.outcomes.push((
                segment.index,
                SegmentOutcome::BestEffortExhausted {
                    score_after: last_score,
                },
            ));
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    fn cue(index: usize, text: &str) -> SubtitleCue {
        SubtitleCue {
            index,
            timing: "00:00:00,000 --> 00:00:02,000".to_string(),
            text: text.to_string(),
        }
    }

    /// Fake backend: records every call and returns scripted text keyed by the
    /// rung label. Lets tests assert how many times each segment was invoked
    /// and which rung was used (guardrails #1 and #4).
    struct FakeBackend {
        /// (model_label, source_first_text) -> replacement text for each cue.
        scripted: HashMap<&'static str, &'static str>,
        calls: RefCell<Vec<(&'static str, usize)>>,
    }

    impl FakeBackend {
        fn new(scripted: HashMap<&'static str, &'static str>) -> Self {
            Self {
                scripted,
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl SegmentBackend for FakeBackend {
        fn translate_segment(
            &self,
            step: &MtBackendStep,
            source: &[SubtitleCue],
        ) -> Result<Vec<SubtitleCue>, String> {
            self.calls.borrow_mut().push((step.label, source.len()));
            let text = self.scripted.get(step.label).copied().unwrap_or("?");
            Ok(source
                .iter()
                .map(|c| cue(c.index, text))
                .collect::<Vec<_>>())
        }
    }

    /// Scripted scorer: maps a cue's text to a fixed (score, penalty). Makes
    /// ladder decisions deterministic without a model.
    struct FakeScorer {
        scores: HashMap<&'static str, (f64, f64)>,
    }

    impl SceneScorer for FakeScorer {
        fn score(&self, translated: &[SubtitleCue]) -> (f64, f64) {
            let key = translated.first().map(|c| c.text.as_str()).unwrap_or("");
            self.scores.get(key).copied().unwrap_or((0.0, 1.0))
        }
    }

    fn seg(index: usize, start: usize, end: usize, base_score: f64, floor: f64) -> FailingSegment {
        FailingSegment {
            index,
            start,
            end,
            base_score,
            floor,
        }
    }

    #[test]
    fn escalates_and_recovers_what_to_thats_case() {
        // Models the real "What→That's" recovery: 600M below floor, 1.3B above.
        let backend = FakeBackend::new(HashMap::from([("nllb-1.3B", "That's right.")]));
        let scorer = FakeScorer {
            scores: HashMap::from([
                ("What's right.", (0.40, 0.6)), // base text, below floor
                ("That's right.", (0.90, 0.1)), // 1.3B retry, above floor
            ]),
        };
        let source = vec![cue(1, "そうだね")];
        let current = vec![cue(1, "What's right.")];
        let failing = [seg(0, 0, 1, 0.40, 0.70)];

        let policy = EscalationPolicy::for_profile(QualityProfile::Balanced, 4, None);
        let report =
            run_escalation_ladder(&policy, &backend, &scorer, &source, &current, &failing, 1);

        assert_eq!(report.events.len(), 1);
        let ev = &report.events[0];
        assert!(ev.recovered);
        assert_eq!(ev.from, "nllb-600M");
        assert_eq!(ev.to, "nllb-1.3B");
        assert!(ev.gate_score_after > ev.gate_score_before);
        assert_eq!(report.recovered_texts.len(), 1);
        assert_eq!(report.recovered_texts[0].texts, vec!["That's right."]);
        assert!(matches!(
            report.outcomes[0].1,
            SegmentOutcome::Recovered { .. }
        ));
    }

    #[test]
    fn escalates_and_stays_failed_marks_best_effort_no_second_retry() {
        // Casual-speech sample: fails at BOTH 600M and 1.3B.
        let backend = FakeBackend::new(HashMap::from([("nllb-1.3B", "Huh? Huh? Huh?")]));
        let scorer = FakeScorer {
            scores: HashMap::from([
                ("Huh? What?", (0.20, 0.9)),
                ("Huh? Huh? Huh?", (0.22, 0.95)), // still below floor
            ]),
        };
        let source = vec![cue(1, "えっ")];
        let current = vec![cue(1, "Huh? What?")];
        let failing = [seg(0, 0, 1, 0.20, 0.70)];

        let policy = EscalationPolicy::for_profile(QualityProfile::Strict, 8, None);
        let report =
            run_escalation_ladder(&policy, &backend, &scorer, &source, &current, &failing, 1);

        assert_eq!(report.events.len(), 1);
        assert!(!report.events[0].recovered);
        assert!(matches!(
            report.outcomes[0].1,
            SegmentOutcome::BestEffortExhausted { .. }
        ));
        assert!(report.recovered_texts.is_empty());

        // Guardrail #4: the segment was translated EXACTLY ONCE here (rung 1).
        // The base rung (600M) was produced by the document pass, not this ladder.
        let calls = backend.calls.borrow();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "nllb-1.3B");
    }

    #[test]
    fn budget_cap_stops_escalation_when_over_threshold() {
        // 10 segments total; >30% fail. Budget = floor(10 * 0.30) = 3.
        let backend = FakeBackend::new(HashMap::from([("nllb-1.3B", "Fixed.")]));
        let scorer = FakeScorer {
            scores: HashMap::from([("Broken.", (0.30, 0.6)), ("Fixed.", (0.95, 0.05))]),
        };
        let source: Vec<SubtitleCue> = (0..6).map(|i| cue(i + 1, "src")).collect();
        let current: Vec<SubtitleCue> = (0..6).map(|i| cue(i + 1, "Broken.")).collect();
        let failing: Vec<FailingSegment> = (0..6).map(|i| seg(i, i, i + 1, 0.30, 0.70)).collect();

        let policy = EscalationPolicy::for_profile(QualityProfile::Balanced, 4, None);
        let report =
            run_escalation_ladder(&policy, &backend, &scorer, &source, &current, &failing, 10);

        assert!(report.budget_exceeded);
        let escalated = report
            .outcomes
            .iter()
            .filter(|(_, o)| !matches!(o, SegmentOutcome::SkippedByBudget))
            .count();
        let skipped = report
            .outcomes
            .iter()
            .filter(|(_, o)| matches!(o, SegmentOutcome::SkippedByBudget))
            .count();
        assert_eq!(escalated, 3, "only the budget (3) may escalate");
        assert_eq!(skipped, 3, "the remaining fall through to best-effort");
        // Backend invoked once per escalated segment only.
        assert_eq!(backend.calls.borrow().len(), 3);
    }

    #[test]
    fn fast_profile_never_escalates() {
        let backend = FakeBackend::new(HashMap::from([("nllb-1.3B", "Fixed.")]));
        let scorer = FakeScorer {
            scores: HashMap::from([("Broken.", (0.30, 0.6))]),
        };
        let source = vec![cue(1, "src")];
        let current = vec![cue(1, "Broken.")];
        let failing = [seg(0, 0, 1, 0.30, 0.70)];

        let policy = EscalationPolicy::for_profile(QualityProfile::Fast, 2, None);
        assert!(!policy.can_escalate());
        let report =
            run_escalation_ladder(&policy, &backend, &scorer, &source, &current, &failing, 1);

        assert!(report.events.is_empty());
        assert!(backend.calls.borrow().is_empty(), "1.3B never invoked");
        assert!(matches!(
            report.outcomes[0].1,
            SegmentOutcome::SkippedByPolicy
        ));
    }

    #[test]
    fn balanced_escalates_at_most_once_per_segment() {
        let policy = EscalationPolicy::for_profile(QualityProfile::Balanced, 4, None);
        assert_eq!(policy.max_escalations, 1);
        assert_eq!(policy.steps.len(), 2);
        // Without an LLM rung there is exactly one rung to walk (1.3B).
        assert_eq!(policy.escalation_rungs().len(), 1);
        assert!(policy.steps.get(2).is_none());
    }

    #[test]
    fn strict_walks_full_list_with_hard_fail_tail() {
        let policy = EscalationPolicy::for_profile(QualityProfile::Strict, 8, None);
        assert!(policy.can_escalate());
        assert!(policy.hard_fail_on_exhaustion);
        assert_eq!(policy.steps[1].label, "nllb-1.3B");
    }

    #[test]
    fn telemetry_event_matches_exact_schema() {
        let event = EscalationEvent {
            segment: 7,
            from: "nllb-600M",
            to: "nllb-1.3B",
            recovered: true,
            gate_score_before: 0.40,
            gate_score_after: 0.91,
        };
        let json = event.as_event();
        assert_eq!(json["event"], "mt_escalation");
        assert_eq!(json["segment"], 7);
        assert_eq!(json["from"], "nllb-600M");
        assert_eq!(json["to"], "nllb-1.3B");
        assert_eq!(json["reason"], "quality_gate");
        assert_eq!(json["outcome"], "recovered");
        assert_eq!(json["gate_score_before"], 0.40);
        assert_eq!(json["gate_score_after"], 0.91);

        let failed = EscalationEvent {
            recovered: false,
            ..event
        };
        assert_eq!(failed.as_event()["outcome"], "still_failed");
    }

    #[test]
    fn cue_count_mismatch_is_rejected_not_spliced() {
        // 1.3B returns the wrong number of cues — must be treated as still-failed.
        struct MismatchBackend;
        impl SegmentBackend for MismatchBackend {
            fn translate_segment(
                &self,
                _step: &MtBackendStep,
                _source: &[SubtitleCue],
            ) -> Result<Vec<SubtitleCue>, String> {
                Ok(vec![cue(1, "only one")]) // expected 2
            }
        }
        let scorer = FakeScorer {
            scores: HashMap::from([("only one", (0.99, 0.0))]),
        };
        let source = vec![cue(1, "a"), cue(2, "b")];
        let current = vec![cue(1, "bad"), cue(2, "bad")];
        let failing = [seg(0, 0, 2, 0.20, 0.70)];

        let policy = EscalationPolicy::for_profile(QualityProfile::Balanced, 4, None);
        let report = run_escalation_ladder(
            &policy,
            &MismatchBackend,
            &scorer,
            &source,
            &current,
            &failing,
            1,
        );
        assert!(!report.events[0].recovered);
        assert!(report.recovered_texts.is_empty());
        assert!(matches!(
            report.outcomes[0].1,
            SegmentOutcome::BestEffortExhausted { .. }
        ));
    }

    /// A fake LLM rung (the Qwen3 tier wired live in WS-A increment 2).
    fn llm_step() -> MtBackendStep {
        MtBackendStep {
            label: "qwen3-4b",
            model_name: "qwen3-4b-q4km".to_string(),
            beam_size: 1,
            kind: BackendKind::Llm,
        }
    }

    #[test]
    fn llm_rung_extends_ladder_to_three_steps() {
        let policy = EscalationPolicy::for_profile(QualityProfile::Balanced, 4, Some(llm_step()));
        assert_eq!(policy.steps.len(), 3);
        assert_eq!(policy.max_escalations, 2);
        // Two rungs to walk now: 1.3B then the LLM.
        assert_eq!(policy.escalation_rungs().len(), 2);
        assert_eq!(policy.steps[2].kind, BackendKind::Llm);
        assert_eq!(policy.steps[2].label, "qwen3-4b");
    }

    #[test]
    fn multi_rung_climbs_to_llm_when_nllb_still_fails() {
        // 1.3B improves but stays below floor; the LLM rung clears it.
        let backend = FakeBackend::new(HashMap::from([
            ("nllb-1.3B", "still rough"),
            ("qwen3-4b", "Finally right."),
        ]));
        let scorer = FakeScorer {
            scores: HashMap::from([
                ("Broken.", (0.30, 0.6)),         // base
                ("still rough", (0.50, 0.5)),     // 1.3B, still below floor 0.70
                ("Finally right.", (0.92, 0.05)), // LLM, clears floor
            ]),
        };
        let source = vec![cue(1, "src")];
        let current = vec![cue(1, "Broken.")];
        let failing = [seg(0, 0, 1, 0.30, 0.70)];

        let policy = EscalationPolicy::for_profile(QualityProfile::Strict, 8, Some(llm_step()));
        let report =
            run_escalation_ladder(&policy, &backend, &scorer, &source, &current, &failing, 1);

        // Both rungs attempted, in order.
        let calls = backend.calls.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, "nllb-1.3B");
        assert_eq!(calls[1].0, "qwen3-4b");

        // Two telemetry events: 600M→1.3B (failed), 1.3B→qwen3-4b (recovered).
        assert_eq!(report.events.len(), 2);
        assert_eq!(
            (report.events[0].from, report.events[0].to),
            ("nllb-600M", "nllb-1.3B")
        );
        assert!(!report.events[0].recovered);
        assert_eq!(
            (report.events[1].from, report.events[1].to),
            ("nllb-1.3B", "qwen3-4b")
        );
        assert!(report.events[1].recovered);

        assert_eq!(report.recovered_texts.len(), 1);
        assert_eq!(report.recovered_texts[0].texts, vec!["Finally right."]);
        assert!(matches!(
            report.outcomes[0].1,
            SegmentOutcome::Recovered { .. }
        ));
    }

    #[test]
    fn multi_rung_stops_at_first_recovery_llm_never_called() {
        // 1.3B already clears the floor; the LLM rung must not be invoked.
        let backend = FakeBackend::new(HashMap::from([
            ("nllb-1.3B", "Fixed early."),
            ("qwen3-4b", "should not be used"),
        ]));
        let scorer = FakeScorer {
            scores: HashMap::from([("Broken.", (0.30, 0.6)), ("Fixed early.", (0.90, 0.05))]),
        };
        let source = vec![cue(1, "src")];
        let current = vec![cue(1, "Broken.")];
        let failing = [seg(0, 0, 1, 0.30, 0.70)];

        let policy = EscalationPolicy::for_profile(QualityProfile::Strict, 8, Some(llm_step()));
        let report =
            run_escalation_ladder(&policy, &backend, &scorer, &source, &current, &failing, 1);

        let calls = backend.calls.borrow();
        assert_eq!(
            calls.len(),
            1,
            "LLM rung must not be reached after recovery"
        );
        assert_eq!(calls[0].0, "nllb-1.3B");
        assert_eq!(report.events.len(), 1);
        assert!(report.events[0].recovered);
        assert_eq!(report.recovered_texts[0].texts, vec!["Fixed early."]);
    }
}
