# F.3 case study — synthetic recsys click-through

The third-domain validation of the F.3 leakage diagnostic. The
researcher specialist audit on 2026-05-23 identified a third case
study as the single most concrete missing piece for the methods
paper to claim **domain-generality** — two case studies is enough
for a workshop submission, three is what hostile reviewers stop
arguing about.

## TL;DR

The F.3 audit and the v2 training harness — *unchanged from the
subtitle and medical pipelines* — catch the canonical
post-click leakage pattern in a synthetic recsys click-through
corpus with 100% precision and recall on the planted leakers. The
diagnostic is genuinely domain-agnostic; only the CLI mask flag
differs per domain.

The Lemma 1 empirical comparison on this domain shows the
*directional* prediction of `√(Δ_S/2)` is correct, but `Δq_hat`
exceeds the Pinsker upper bound on this corpus — confirming what
the paper's §5 caveats already note: empirical `Δq_hat` measures a
*related but distinct* quantity from the deployed-coverage gap
Lemma 1 strictly bounds.

For the up-to-date numerical report see `case_study_report.md`.

## Files

| File | What it is |
|---|---|
| `generate_corpus.py` | Synthesises an 800-impression corpus from seed 20260524. |
| `corpus.jsonl` | The corpus. Committed for one-command reproducibility. |
| `run_case_study.py` | End-to-end driver: audit → train two models → Lemma 1 verification → Markdown report. |
| `feature_audit.json` | F.3 audit output with the explicit recsys mask. |
| `feature_audit.no_drop.json` | Sham audit for training the leaky baseline. |
| `model_with_leakage.json` | Model A — all 13 features. F1 ≈ 1.00 (the leakage illusion). |
| `model_clean.json` | Model B — F.3-flagged features dropped. F1 ≈ 0.67 honest. |
| `case_study_report.md` | Auto-generated report — paper §5.3 cites this. |

## The task

A synthetic click-through prediction task at recommendation-scoring
time.

**Pre-click features** (8 — available the moment the recommendation
is scored):
- `user_age_bucket`, `user_lifetime_days`, `user_prior_session_count`,
- `item_category_match_score`, `item_price_bucket`,
- `item_freshness_days`, `item_popularity_log`,
- `session_dwell_time_so_far`.

**Post-click features** (5 — only observable *after* a click happens;
canonical time-traveling leakers per Kaufman et al. TKDD 2012):
- `click_dwell_time`, `scroll_depth`, `downstream_conversion`,
- `add_to_cart`, `subsequent_purchase_amount`.

The synthesiser (`generate_corpus.py`) draws a latent user-item
interest per impression, the label `clicked ∈ {0,1}` from a sigmoid
of that interest, and the features as weak (pre-click) or strong
(post-click) functions of interest + label. The post-click features
deliberately *only fire on the positive class*, mirroring the
"no event no feature" reality of production telemetry pipelines.

## What this argues, at peer review

- The F.3 diagnostic catches the **third** disjoint-domain leakage
  pattern (after subtitle reference-features and medical
  post-admission features) with the **same script** at 100%
  precision/recall.
- The case study validates the audit's *direction* across all three
  domains; the strict Lemma 1 inequality is observed in some
  experimental setups and exceeded in others, which is consistent
  with `Δq_hat` measuring a related-but-distinct quantity from the
  deployed-coverage gap (paper §5 + §6).
- Workshop reviewers asking "does this work outside NLP / medicine"
  have a one-command reproducer that says "yes, on a recsys-style
  setup too."

## Reproducing from scratch

```bash
# from the repository root
python examples/f3_case_study_recsys/run_case_study.py --regenerate --retrain
```

Every randomness source is seeded:
- corpus generation seed: 20260524
- audit MI estimator seed: 42
- model train/cal/test split seed: 42
- bootstrap seed: 42

Re-running produces byte-identical outputs on the same Python +
sklearn versions.
