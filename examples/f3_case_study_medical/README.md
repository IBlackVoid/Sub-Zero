# F.3 case study — synthetic medical mortality

This directory ships a self-contained, reproducible second-domain
validation of the **F.3 leakage diagnostic** (theory at
`docs/F3_leakage_diagnostic.md`, runnable code at
`scripts/audit_learned_gate_features.py`).

> **Why it exists.** The F.3 diagnostic was developed to catch a
> specific failure mode in VoiDex's subtitle quality gate
> (reference-derived features that don't exist at inference). The
> theory-lab review of 2026-05-23 argued that the diagnostic
> generalises beyond subtitles and recommended a second-domain case
> study to make the methods claim defensible at peer review. This is
> that case study.

## TL;DR

The audit, **unchanged from the subtitle pipeline**, catches the
canonical "time-traveling feature" leakage bug in a synthetic
medical-risk corpus with 100% precision and recall on the leakers.
The two models trained on the same data — one with the leakers, one
without — differ by 24 percentage points of test accuracy. The Fano-
style information-theoretic lower bound (F.3 §3) predicts a non-zero
gap in the right direction, comfortably below the empirical 24-point
gap (as expected for a Fano inequality, which is loose).

For the up-to-date numerical report see `case_study_report.md`,
written by `run_case_study.py`.

## Files

| File | What it is |
|---|---|
| `generate_corpus.py` | Synthesises the 500-patient corpus from a fixed seed. |
| `corpus.jsonl` | The corpus. Committed so the case study is one-command reproducible. |
| `run_case_study.py` | End-to-end driver: audit → train two models → compute Fano bound → write the Markdown report. |
| `feature_audit.json` | F.3 audit output with the explicit medical mask. |
| `feature_audit.no_drop.json` | Sham audit with no flagged features, so the leaky model can be trained on every feature. |
| `model_with_leakage.json` | Model A — trained with all 13 features. F1 ≈ 1.00 (the leakage illusion). |
| `model_clean.json` | Model B — trained without the 5 audit-flagged features. F1 ≈ 0.86 honest. |
| `case_study_report.md` | The auto-generated report with the full audit table + Fano-vs-empirical comparison. |

## The task

A synthetic 30-day mortality prediction at hospital admission.

**Honest features** (8 — available the moment the patient walks in):
- `age`, `sex`, `systolic_bp`, `heart_rate`, `oxygen_saturation`,
- `glasgow_coma_scale`, `admission_complaint_severity`, `comorbidity_count`.

**Post-admission features** (5 — *not* available at the admission
decision; the canonical "time-traveling" leakers):
- `peak_lactate_24h`, `icu_days`, `max_inflammatory_marker`,
- `intubated_within_24h`, `max_pressor_dose`.

The synthesiser (`generate_corpus.py`) draws a latent severity per
patient, the outcome from a sigmoid of that severity, and the
features as weak (honest) or strong (post-admission) functions of
severity + outcome. The post-admission features deliberately
**correlate with the realised label**, not just severity, because
the body's response over the next 24 hours is part of the outcome.

## What the diagnostic does

1. Compute `I_univ(X_i; Y)` per feature in the rich regime (data
   as-is).
2. Zero out the 5 named features → compute the same MI in the
   masked regime.
3. For each feature: `leak_score_univariate = (MI_rich − MI_masked) / MI_rich`.

A clean diagnostic would flag every named leaker at `leak_score ≈ 1`
and every honest feature at `leak_score ≈ 0`. The audit does exactly
that — see the table in `case_study_report.md`.

## Reproducing from scratch

```bash
# from the repository root
python examples/f3_case_study_medical/run_case_study.py --regenerate --retrain
```

The driver regenerates the corpus, runs the audit, trains both
models, computes the Fano bound, and writes `case_study_report.md`.
Every random source is seeded; re-running produces byte-identical
outputs across runs on the same Python + sklearn versions.

## What this argues, at peer review

- The F.3 diagnostic is **not subtitle-specific**. The script and the
  v2 retrain harness were used in this case study without code
  changes — only CLI flags (`--mask-pattern none --mask-features ...`
  and `--feature-schema corpus`).
- The diagnostic catches a canonical leakage bug (Kaufman et al.
  TKDD 2012 "time-traveling features") with 100% precision/recall on
  this synthetic ground truth.
- The empirical accuracy gap (24 pp) sits above the loose Fano
  lower bound, validating the qualitative direction of the bound.

This is what the theory-lab review asked for: a second domain,
unmodified diagnostic, ground-truth leakers, calibrated outcome. The
methods paper has its second case study.
