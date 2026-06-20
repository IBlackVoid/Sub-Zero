# VoiDex human-evaluation harness (F.2)

A paper-ready, three-stage harness for blind side-by-side rating of
VoiDex translation outputs.

```
prepare_pairs.py  -->  rater_cli.py  -->  aggregate.py
   (offline)          (per rater)        (offline)
```

## Stage 1 — sample blinded pairs

For each case in `benchmarks/runs/`, every pair of variants
(e.g. `fast` vs `fast_speaker_diarize`) becomes one comparison.
The script samples `--cues-per-pair` aligned cues from each variant,
assigns a random `pair_id`, and randomises the A/B label so the rater
cannot tell which variant they are scoring.

```powershell
python scripts/human_eval/prepare_pairs.py `
    --runs-dir benchmarks/runs `
    --out-dir benchmarks/human_eval `
    --cues-per-pair 8 `
    --max-pairs 60 `
    --seed 1337
```

Outputs:

- `benchmarks/human_eval/pairs.jsonl` — blinded pairs the rater sees
- `benchmarks/human_eval/.truth/mapping.jsonl` — A/B → variant truth
  (kept separately so the rater never opens it)

## Stage 2 — rate

Each rater runs `rater_cli.py` with their own initials and an output
file. Ratings are written incrementally so a rater can quit and
resume.

```powershell
python scripts/human_eval/rater_cli.py `
    --pairs benchmarks/human_eval/pairs.jsonl `
    --out   benchmarks/human_eval/ratings_alice.jsonl `
    --rater alice
```

For each pair the rater enters:

1. **Preference**: `a` / `b` / `t` (tie / can't tell).
2. *(Optional)* **Likert 1–5** for each of A and B on adequacy.

`q` quits and saves progress. `s` skips the current pair.

## Stage 3 — aggregate

```powershell
python scripts/human_eval/aggregate.py `
    --truth benchmarks/human_eval/.truth/mapping.jsonl `
    --ratings benchmarks/human_eval/ratings_*.jsonl
```

The aggregator reports:

- Per-variant **win rate** and 95 % Wilson confidence interval.
- **Paired sign test** on preferences (`scipy`-free; uses the exact
  binomial CDF with ties split symmetrically).
- **Krippendorff's α** across raters on the Likert data
  (interval scale; computed via the coincidence-matrix formula).
- Per-variant Likert mean ± 95 % CI, plus paired delta with a
  bootstrapped confidence interval.

Outputs `benchmarks/human_eval/summary.json` plus a markdown table
suitable for the paper.

## Statistical methodology

- **Sign test** on preferences. We treat each pair as a Bernoulli
  trial with `p = P(rater prefers A)`. Ties are split evenly between
  A and B. Two-sided p-value uses the exact binomial CDF.
- **Krippendorff's α** for Likert ratings. We use the interval-data
  variant: `α = 1 − D_o / D_e`, where `D_o` is the mean squared
  difference between rater pairs on each unit and `D_e` is the same
  quantity computed across all rater–unit cells.
- **Wilson 95 % CI** on win-rate proportions, since binomial counts
  near 0 or 1 are ill-served by the normal approximation.
- **Bootstrapped CI** on Likert deltas: 10 000 resamples of the paired
  differences with replacement, percentile method at `[2.5%, 97.5%]`.

All routines are pure-Python (`statistics` + `math` only), so the
harness has no third-party dependencies beyond the project's existing
`numpy` (used only by `train_gate.py`, optional here).
