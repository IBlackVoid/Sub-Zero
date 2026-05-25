#!/usr/bin/env python3
"""Re-train the Sub-Zero learned quality gate.

This harness emits either:

- **v1.1** — the legacy schema: linear logistic regression with a single
  threshold (`sigmoid(z) >= threshold` → pass). Kept for backward
  compatibility with the Rust engine's existing path.

- **v2.0** — the new schema (`--v2`):

      logistic regression  →  isotonic calibration  →  split conformal

  The model emits *three* states at inference time — PASS, REJECT,
  ABSTAIN — with a formal coverage guarantee (Vovk et al. 2005;
  Romano-Patterson-Candès 2019). The isotonic stage closes the gap
  between raw sigmoid scores and true class probabilities; the split
  conformal stage produces a `q_hat` such that the engine's prediction
  set covers the true label with probability ≥ 1−α on data from the
  same distribution as the calibration set.

  The intent of ABSTAIN is *honest uncertainty*: the gate does not
  pretend to know whether a run is good or bad when the calibrated
  probability lies within `q_hat` of the threshold. Calling code can
  treat ABSTAIN as a soft pass, surface it to the operator, or kick
  the run into the rescue pipeline — the schema does not prescribe.

Why this is novel
-----------------
Every shipped subtitle-quality estimator the maintainer has surveyed
(Subtitle Edit's heuristic checker; COMETKiwi-22 sentence-level QE;
the closed enterprise QE tools) emits a *single calibrated-ish score*
with no formal coverage statement. Wrapping that score in a split
conformal envelope is standard in ML literature but, to the maintainer's
knowledge, has not been published for subtitle quality estimation.

Output schema v2.0 (top-level)
------------------------------

```
{
  "version": "2.0",
  "kind": "learned-quality-gate",
  "model_kind": "logistic+isotonic+conformal",
  "threshold": 0.5,
  "bias": <float>,
  "features": [<str>, ...],
  "mean": [<float>, ...],
  "std":  [<float>, ...],
  "weights": [<float>, ...],
  "calibration": {
    "kind": "isotonic",
    "x": [<float>, ...],   // sigmoid-score knots, monotonic
    "y": [<float>, ...]    // calibrated-probability knots, monotonic
  },
  "conformal": {
    "alpha": <float>,
    "q_hat": <float>,
    "calibration_set_size": <int>,
    "interpretation": "PASS if p_cal - q_hat > threshold; REJECT if p_cal + q_hat < threshold; ABSTAIN otherwise."
  },
  "holdout_metrics": {...},
  "training_seed": <int>,
  "data_sha256": <hex>,
  "git_commit": <hex>,
  "training_set_size": <int>,
  "feature_audit": { "path": <str>, "sha256": <hex>, "flagged_features": [...] }
}
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    print("error: numpy required (`pip install numpy`)", file=sys.stderr)
    raise SystemExit(2) from exc

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
except ImportError as exc:  # pragma: no cover
    print(
        "error: scikit-learn required (`pip install scikit-learn`)",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


# The canonical engine-side feature list, kept in sync with
# `src/engine/pipeline/learned_gate.rs:feature_value()`.
ENGINE_FEATURES: tuple[str, ...] = (
    "cue_count",
    "top_line_ratio",
    "overlap_ratio",
    "non_empty_ratio",
    "anomaly_ratio",
    "malformed_contraction_ratio",
    "low_function_word_ratio",
    "adjacent_repeat_ratio",
    "scene_low_quality_ratio",
    "scene_count",
    "name_inconsistency_ratio",
    "register_speakers_observed",
    "register_speakers_formal",
    "register_cues_labeled",
    "diar_speakers",
    "diar_used_segments",
    "diar_assigned_cues",
)
DELTA_FEATURES: tuple[str, ...] = (
    "delta_cue_density_ratio",
    "delta_japanese_char_ratio",
    "delta_line_char_similarity",
    "delta_non_empty_rate",
    "delta_reference_char_similarity",
    "delta_time_coverage_ratio",
    "delta_token_overlap_f1",
    "delta_weighted_timing_iou",
    "delta_weighted_token_f1",
)
VARIANT_FEATURES: tuple[str, ...] = (
    "variant_fast",
    "variant_fast_speaker_diarize",
)


@dataclass
class Example:
    label: int
    features: dict[str, float]


def load_corpus(path: Path) -> list[Example]:
    rows: list[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"corpus {path}:{line_no}: invalid JSON: {exc}")
            label = obj.get("label")
            if label not in (0, 1):
                raise SystemExit(
                    f"corpus {path}:{line_no}: label must be 0/1, got {label!r}"
                )
            feats = obj.get("features")
            if not isinstance(feats, dict):
                raise SystemExit(
                    f"corpus {path}:{line_no}: 'features' must be an object"
                )
            cleaned: dict[str, float] = {}
            for k, v in feats.items():
                try:
                    cleaned[k] = float(v)
                except (TypeError, ValueError):
                    continue
            rows.append(Example(label=int(label), features=cleaned))
    if not rows:
        raise SystemExit(f"corpus {path}: no usable rows")
    return rows


def hash_corpus(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def pick_features(
    corpus: list[Example],
    exclude_delta: bool,
    drop_features: set[str],
    schema: str,
) -> list[str]:
    """Return the ordered feature list to use for fitting.

    Filters:
      - `schema == "subtitle"` (default): intersection with the
        subtitle engine's hard-coded feature schema. Keeps the
        production pipeline tight — no surprise features at inference.
      - `schema == "corpus"`: accept any feature present in the
        corpus. Use for second-domain experiments and F.3 case
        studies, where the engine doesn't have a fixed schema.
      - In every mode: intersection with what the corpus carries
        (no zero-only columns), and minus `drop_features` (typically
        from the MI audit).
      - `exclude_delta` removes the subtitle-specific delta family;
        a no-op in `corpus` mode where the names rarely apply.
    """
    seen: set[str] = set()
    seen_order: list[str] = []
    for row in corpus:
        for name in row.features:
            if name not in seen:
                seen.add(name)
                seen_order.append(name)
    if schema == "subtitle":
        engine_set = set(ENGINE_FEATURES) | set(VARIANT_FEATURES)
        if not exclude_delta:
            engine_set |= set(DELTA_FEATURES)
        ordered = (*ENGINE_FEATURES, *DELTA_FEATURES, *VARIANT_FEATURES)
        selected = [
            f
            for f in ordered
            if f in engine_set and f in seen and f not in drop_features
        ]
    elif schema == "corpus":
        # Preserve corpus-encounter order so the fit reproduces
        # deterministically across runs on the same corpus.
        selected = [f for f in seen_order if f not in drop_features]
    else:
        raise SystemExit(f"unknown --feature-schema: {schema!r}")
    if not selected:
        raise SystemExit("no usable features remained after filtering")
    return selected


def to_matrix(
    rows: Iterable[Example], features: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array(
        [[row.features.get(f, 0.0) for f in features] for row in rows],
        dtype=np.float64,
    )
    ys = np.array([row.label for row in rows], dtype=np.int64)
    return xs, ys


def bootstrap_f1(
    y_true: np.ndarray, y_pred: np.ndarray, *, samples: int, rng: random.Random
) -> tuple[float | None, float | None]:
    """Bootstrap a 95% CI for held-out F1. Returns ``(None, None)``
    when no bootstrap could be drawn so the model JSON stays
    strict-JSON-conformant (no NaN tokens).
    """
    if samples <= 0 or len(y_true) == 0:
        return (None, None)
    n = len(y_true)
    estimates: list[float] = []
    for _ in range(samples):
        idx = [rng.randrange(0, n) for _ in range(n)]
        yt = y_true[idx]
        yp = y_pred[idx]
        if yt.sum() == 0 and yp.sum() == 0:
            estimates.append(1.0)
            continue
        estimates.append(float(f1_score(yt, yp, zero_division=0)))
    estimates.sort()
    lo = estimates[int(0.025 * samples)]
    hi = estimates[int(0.975 * samples) - 1]
    return (lo, hi)


def isotonic_knots(iso: IsotonicRegression) -> tuple[list[float], list[float]]:
    """Extract a JSON-serialisable representation of the isotonic step
    function. The Rust loader uses these as a piecewise-linear lookup.

    sklearn stores the breakpoints as `X_thresholds_` (monotonic x) and
    `y_thresholds_` (monotonic y). For x outside [X_thresholds_[0],
    X_thresholds_[-1]] the function clamps to the nearest endpoint —
    that's the same convention the Rust evaluator uses, so we don't add
    sentinel points.
    """
    x = [float(v) for v in iso.X_thresholds_.tolist()]
    y = [float(v) for v in iso.y_thresholds_.tolist()]
    return x, y


def compute_conformal_q_hat(
    p_cal: np.ndarray, y_cal: np.ndarray, alpha: float
) -> float:
    """Split-conformal nonconformity threshold for binary classification.

    Nonconformity is `1 - p(y_true | x)`. The conformal quantile is
    `ceil((n+1) * (1-α)) / n` of the sorted nonconformity scores,
    matching the standard split-conformal definition (Vovk 2005).

    Returns `q_hat ∈ [0, 1]`. Calling code interprets it as a margin
    around the threshold: a prediction is "confident" (i.e. the
    prediction set is a singleton) iff `|p_cal - threshold| > q_hat`.
    """
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if len(p_cal) == 0:
        return 1.0
    # nonconformity per row
    nonconf = np.where(y_cal == 1, 1.0 - p_cal, p_cal)
    n = len(nonconf)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    rank = max(1, min(n, rank))
    sorted_scores = np.sort(nonconf)
    return float(sorted_scores[rank - 1])


def conformal_state(p_cal: float, threshold: float, q_hat: float) -> str:
    """Map a calibrated probability into PASS / REJECT / ABSTAIN
    using the conformal margin."""
    lower = p_cal - q_hat
    upper = p_cal + q_hat
    if lower > threshold:
        return "PASS"
    if upper < threshold:
        return "REJECT"
    return "ABSTAIN"


def evaluate_conformal_on_test(
    p_cal: np.ndarray, y_test: np.ndarray, threshold: float, q_hat: float
) -> dict:
    """Coverage + ABSTAIN rate + per-decision metrics on a test set.

    Returns JSON-conformant types only — when a quantity is undefined
    (e.g., no decisive predictions → no per-decision accuracy), we emit
    ``None`` rather than ``float('nan')``. The Rust loader uses strict
    ``serde_json`` which rejects the literal ``NaN`` token; Python's
    ``json.dump`` will happily write it. Keeping NaN out of the model
    artifact is non-negotiable.
    """
    states = [conformal_state(p, threshold, q_hat) for p in p_cal]
    n = len(states)
    abstain_n = sum(1 for s in states if s == "ABSTAIN")
    pass_n = sum(1 for s in states if s == "PASS")
    reject_n = sum(1 for s in states if s == "REJECT")

    # Coverage: did the prediction set contain the true label?
    # PASS -> {1}, REJECT -> {0}, ABSTAIN -> {0,1}.
    covered = 0
    for state, y in zip(states, y_test.tolist()):
        if state == "ABSTAIN":
            covered += 1
        elif state == "PASS" and y == 1:
            covered += 1
        elif state == "REJECT" and y == 0:
            covered += 1
    coverage: float | None = (covered / n) if n else None

    # Conditional decision quality: among non-abstain decisions, how
    # often were they correct? Undefined (None) when nothing was decisive.
    decisive_n = pass_n + reject_n
    decisive_correct = 0
    for state, y in zip(states, y_test.tolist()):
        if state == "PASS" and y == 1:
            decisive_correct += 1
        elif state == "REJECT" and y == 0:
            decisive_correct += 1
    decisive_acc: float | None = (
        decisive_correct / decisive_n if decisive_n else None
    )

    abstain_rate: float | None = (abstain_n / n) if n else None

    return {
        "n": int(n),
        "pass": int(pass_n),
        "reject": int(reject_n),
        "abstain": int(abstain_n),
        "abstain_rate": abstain_rate,
        "coverage": coverage,
        "decisive_accuracy": decisive_acc,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="decision threshold to embed in the model (default 0.5)",
    )
    parser.add_argument(
        "--exclude-delta-features",
        action="store_true",
        help="drop the full delta_* family (audit's coarse leakage ablation)",
    )
    parser.add_argument(
        "--feature-audit",
        type=Path,
        default=None,
        help="JSON produced by audit_learned_gate_features.py; flagged features are dropped",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="bootstrap resample count for F1 CI on the test split",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="emit the v2.0 schema (isotonic + split conformal)",
    )
    parser.add_argument(
        "--conformal-alpha",
        type=float,
        default=0.10,
        help="target miscoverage rate α; v2 prediction set covers true label with prob ≥ 1−α",
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.30,
        help="combined cal+test fraction (default 0.30; v1 uses all of it for test)",
    )
    parser.add_argument(
        "--feature-schema",
        type=str,
        default="subtitle",
        choices=("subtitle", "corpus"),
        help=(
            "Which feature set to allow. 'subtitle' (default) only "
            "uses the subtitle engine's hard-coded schema. 'corpus' "
            "accepts any feature present in the corpus — required for "
            "F.3 second-domain experiments where the engine has no "
            "fixed feature list."
        ),
    )
    parser.add_argument(
        "--alpha-sweep",
        type=str,
        default=None,
        help=(
            "comma-separated list of conformal alpha values; emits a "
            "selective-classification curve (El-Yaniv & Wiener 2010) into "
            "the model JSON under holdout_metrics.alpha_sweep. Has no "
            "effect on the production conformal_alpha used to gate "
            "shipping decisions."
        ),
    )
    args = parser.parse_args(argv)

    if not args.corpus.is_file():
        print(f"error: corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    rows = load_corpus(args.corpus)
    print(f"corpus: {len(rows)} examples  pos={sum(r.label for r in rows)}")

    # Feature-audit-driven leakage filter.
    drop_features: set[str] = set()
    audit_meta: dict | None = None
    if args.feature_audit is not None:
        if not args.feature_audit.is_file():
            print(
                f"error: --feature-audit path not found: {args.feature_audit}",
                file=sys.stderr,
            )
            return 2
        with args.feature_audit.open("r", encoding="utf-8") as f:
            audit_obj = json.load(f)
        drop_features = set(audit_obj.get("flagged_features") or [])
        audit_meta = {
            "path": str(args.feature_audit),
            "sha256": hash_file(args.feature_audit),
            "flagged_features": sorted(drop_features),
            "leak_threshold": audit_obj.get("leak_threshold"),
        }
        print(f"feature audit: dropping {len(drop_features)} flagged features")

    features = pick_features(
        rows,
        exclude_delta=args.exclude_delta_features,
        drop_features=drop_features,
        schema=args.feature_schema,
    )
    print(
        f"features: {len(features)} "
        f"(schema={args.feature_schema}, delta_excluded={args.exclude_delta_features}, "
        f"audit_dropped={len(drop_features)})"
    )

    x, y = to_matrix(rows, features)
    if len(set(y.tolist())) < 2:
        print("error: corpus has only one class; cannot fit", file=sys.stderr)
        return 2

    if not args.v2:
        return _emit_v1(
            args=args,
            x=x,
            y=y,
            features=features,
            rng=rng,
            audit_meta=audit_meta,
        )

    return _emit_v2(
        args=args,
        x=x,
        y=y,
        features=features,
        rng=rng,
        audit_meta=audit_meta,
    )


# ── v1.1 emitter — kept for backward compat ───────────────────────────


def _emit_v1(
    *,
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    features: list[str],
    rng: random.Random,
    audit_meta: dict | None,
) -> int:
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=args.holdout, random_state=args.seed, stratify=y
    )
    mean = x_tr.mean(axis=0)
    std = x_tr.std(axis=0)
    std_safe = np.where(std < 1e-9, 1.0, std)
    x_tr_z = (x_tr - mean) / std_safe
    x_te_z = (x_te - mean) / std_safe

    model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=args.seed
    )
    model.fit(x_tr_z, y_tr)

    proba_te = model.predict_proba(x_te_z)[:, 1]
    yhat_te = (proba_te >= args.threshold).astype(np.int64)
    loss = float(
        -np.mean(
            y_te * np.log(np.clip(proba_te, 1e-12, 1.0))
            + (1 - y_te) * np.log(np.clip(1.0 - proba_te, 1e-12, 1.0))
        )
    )
    acc = float((yhat_te == y_te).mean())
    prec = float(precision_score(y_te, yhat_te, zero_division=0))
    rec = float(recall_score(y_te, yhat_te, zero_division=0))
    f1 = float(f1_score(y_te, yhat_te, zero_division=0))
    boot_lo, boot_hi = bootstrap_f1(y_te, yhat_te, samples=args.bootstrap, rng=rng)

    out = {
        "version": "1.1",
        "kind": "learned-quality-gate",
        "threshold": float(args.threshold),
        "bias": float(model.intercept_[0]),
        "features": features,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": model.coef_[0].tolist(),
        "holdout_metrics": {
            "loss": loss,
            "acc": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f1_bootstrap_ci95": [boot_lo, boot_hi],
        },
        "holdout_size": int(len(y_te)),
        "training_seed": int(args.seed),
        "data_sha256": hash_corpus(args.corpus),
        "git_commit": git_commit(),
        "training_set_size": int(len(y_tr)),
        "exclude_delta_features": bool(args.exclude_delta_features),
    }
    if audit_meta is not None:
        out["feature_audit"] = audit_meta

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # `allow_nan=False` is non-negotiable. Without it, Python's `json`
    # module will happily write the bare `NaN` / `Infinity` tokens that
    # the strict `serde_json` parser on the Rust side flat-out rejects.
    # That bug shipped the v2.0 production artifact once already; the
    # tests at `learned_gate.rs::production_model_artifact_parses`
    # guard against a regression.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")

    print()
    print(f"v1.1 holdout n={len(y_te)}")
    print(f"  acc       = {acc:.4f}")
    print(f"  precision = {prec:.4f}")
    print(f"  recall    = {rec:.4f}")
    print(f"  f1        = {f1:.4f}   (95% bootstrap CI: [{boot_lo:.4f}, {boot_hi:.4f}])")
    print(f"  log-loss  = {loss:.4f}")
    print(f"wrote model -> {args.output}")
    return 0


# ── v2.0 emitter — isotonic + split conformal ─────────────────────────


def _emit_v2(
    *,
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    features: list[str],
    rng: random.Random,
    audit_meta: dict | None,
) -> int:
    # Four-way split: train -> fit LR; iso_cal -> fit isotonic;
    # conf_cal -> compute conformal q_hat on the *frozen* isotonic
    # mapping; test -> report empirical coverage. Sharing the cal set
    # between isotonic and conformal (the previous v2 design) only
    # delivered asymptotic coverage because the score function then
    # depended on the cal labels (Bian & Barber 2023). Splitting them
    # restores the strict marginal-coverage guarantee under
    # exchangeability (Vovk et al. 2005; Lei et al. 2018, Thm 2.1).
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x, y, test_size=args.holdout, random_state=args.seed, stratify=y
    )
    # Split holdout into three equal-ish pieces: iso_cal | conf_cal | test.
    # `train_test_split` only does 2-way; chain two calls.
    x_holdout_a, x_test, y_holdout_a, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=1.0 / 3.0,
        random_state=args.seed,
        stratify=y_holdout,
    )
    x_iso, x_conf, y_iso, y_conf = train_test_split(
        x_holdout_a,
        y_holdout_a,
        test_size=0.5,
        random_state=args.seed,
        stratify=y_holdout_a,
    )

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std_safe = np.where(std < 1e-9, 1.0, std)
    x_train_z = (x_train - mean) / std_safe
    x_iso_z = (x_iso - mean) / std_safe
    x_conf_z = (x_conf - mean) / std_safe
    x_test_z = (x_test - mean) / std_safe

    base = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=args.seed
    )
    base.fit(x_train_z, y_train)

    # Stage 2: isotonic fit on the iso_cal split *only*. Frozen before
    # stage 3 sees it; this is what restores the strict marginal
    # coverage guarantee.
    raw_iso = base.predict_proba(x_iso_z)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_iso, y_iso)
    cal_x_knots, cal_y_knots = isotonic_knots(iso)

    # Stage 3: conformal q_hat on the conf_cal split, using the frozen
    # `iso` and `base`. No labels from conf_cal leak into the score
    # function -> exchangeability holds for the test point.
    raw_conf = base.predict_proba(x_conf_z)[:, 1]
    p_conf_iso = iso.predict(raw_conf)
    q_hat = compute_conformal_q_hat(p_conf_iso, y_conf, args.conformal_alpha)

    # Stage 4: held-out test evaluation.
    raw_test = base.predict_proba(x_test_z)[:, 1]
    p_test_iso = iso.predict(raw_test)

    # Test-set metrics:
    #   1. Crisp F1 at the threshold (ignoring conformal margin)
    yhat_test = (p_test_iso >= args.threshold).astype(np.int64)
    crisp_f1 = float(f1_score(y_test, yhat_test, zero_division=0))
    crisp_prec = float(precision_score(y_test, yhat_test, zero_division=0))
    crisp_rec = float(recall_score(y_test, yhat_test, zero_division=0))
    crisp_acc = float((yhat_test == y_test).mean())
    boot_lo, boot_hi = bootstrap_f1(
        y_test, yhat_test, samples=args.bootstrap, rng=rng
    )

    #   2. Conformal coverage + ABSTAIN rate
    conformal_metrics = evaluate_conformal_on_test(
        p_test_iso, y_test, args.threshold, q_hat
    )

    # Sanity: empirical coverage on the conformal-cal set itself should
    # be close to (1 - alpha). This is the standard self-check; under
    # the 3-way split it now uses the held-out conf_cal split, not the
    # same data the isotonic was fit on.
    cal_metrics = evaluate_conformal_on_test(
        p_conf_iso, y_conf, args.threshold, q_hat
    )

    # Selective-classification curve (El-Yaniv & Wiener 2010): for each
    # alpha in the sweep, recompute q_hat on the conformal-cal split,
    # then report (coverage, abstain_rate, decisive_accuracy) on test.
    # The curve is the actual scientific object — it shows the
    # data-vs-confidence trade-off the model can support.
    alpha_sweep_curve: list[dict] = []
    if args.alpha_sweep:
        for raw_a in args.alpha_sweep.split(","):
            raw_a = raw_a.strip()
            if not raw_a:
                continue
            try:
                a = float(raw_a)
            except ValueError:
                print(f"warning: skipping non-numeric alpha {raw_a!r}", file=sys.stderr)
                continue
            if not (0.0 < a < 1.0):
                print(f"warning: skipping out-of-range alpha {a}", file=sys.stderr)
                continue
            q_a = compute_conformal_q_hat(p_conf_iso, y_conf, a)
            m_a = evaluate_conformal_on_test(p_test_iso, y_test, args.threshold, q_a)
            alpha_sweep_curve.append({
                "alpha": float(a),
                "q_hat": float(q_a),
                "test_coverage": m_a["coverage"],
                "test_abstain_rate": m_a["abstain_rate"],
                "test_decisive_accuracy": m_a["decisive_accuracy"],
                "test_pass": m_a["pass"],
                "test_reject": m_a["reject"],
                "test_abstain": m_a["abstain"],
            })

    out = {
        "version": "2.0",
        "kind": "learned-quality-gate",
        "model_kind": "logistic+isotonic+conformal",
        "threshold": float(args.threshold),
        "bias": float(base.intercept_[0]),
        "features": features,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": base.coef_[0].tolist(),
        "calibration": {
            "kind": "isotonic",
            "x": cal_x_knots,
            "y": cal_y_knots,
            "out_of_bounds": "clip",
        },
        "conformal": {
            "alpha": float(args.conformal_alpha),
            "q_hat": float(q_hat),
            "calibration_set_size": int(len(y_conf)),
            "isotonic_cal_set_size": int(len(y_iso)),
            "split_kind": "disjoint_iso_and_conf",
            "interpretation": (
                "PASS if p_cal - q_hat > threshold; "
                "REJECT if p_cal + q_hat < threshold; "
                "ABSTAIN otherwise."
            ),
        },
        "holdout_metrics": {
            "crisp_acc": crisp_acc,
            "crisp_precision": crisp_prec,
            "crisp_recall": crisp_rec,
            "crisp_f1": crisp_f1,
            "crisp_f1_bootstrap_ci95": [boot_lo, boot_hi],
            "conformal_test": conformal_metrics,
            "conformal_calibration_selfcheck": cal_metrics,
            "alpha_sweep": alpha_sweep_curve,
        },
        "holdout_size": int(len(y_test)),
        "training_seed": int(args.seed),
        "data_sha256": hash_corpus(args.corpus),
        "git_commit": git_commit(),
        "training_set_size": int(len(y_train)),
        "exclude_delta_features": bool(args.exclude_delta_features),
    }
    if audit_meta is not None:
        out["feature_audit"] = audit_meta

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # `allow_nan=False` is non-negotiable. Without it, Python's `json`
    # module will happily write the bare `NaN` / `Infinity` tokens that
    # the strict `serde_json` parser on the Rust side flat-out rejects.
    # That bug shipped the v2.0 production artifact once already; the
    # tests at `learned_gate.rs::production_model_artifact_parses`
    # guard against a regression.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")

    boot_lo_str = f"{boot_lo:.4f}" if boot_lo is not None else "  N/A "
    boot_hi_str = f"{boot_hi:.4f}" if boot_hi is not None else "  N/A "
    cov_str = (
        f"{conformal_metrics['coverage']:.4f}"
        if conformal_metrics["coverage"] is not None
        else "  N/A "
    )
    abs_rate_str = (
        f"{conformal_metrics['abstain_rate'] * 100:.1f}%"
        if conformal_metrics["abstain_rate"] is not None
        else "  N/A"
    )
    sel_str = (
        f"{cal_metrics['coverage']:.4f}"
        if cal_metrics["coverage"] is not None
        else "  N/A "
    )
    print()
    print(
        f"v2.0 splits: train={len(y_train)} iso={len(y_iso)} "
        f"conf={len(y_conf)} test={len(y_test)}"
    )
    print(f"  crisp test F1  = {crisp_f1:.4f}   (95% CI [{boot_lo_str}, {boot_hi_str}])")
    print(f"  conformal alpha = {args.conformal_alpha}")
    print(f"  q_hat          = {q_hat:.4f}")
    # ASCII-only stdout; some Windows consoles default to cp1252 and
    # crash on a literal `>=` (>=).
    print(
        f"  test coverage  = {cov_str}   (target >= {1 - args.conformal_alpha:.2f})"
    )
    print(
        f"  test abstain%  = {abs_rate_str}   "
        f"(pass={conformal_metrics['pass']} reject={conformal_metrics['reject']} "
        f"abstain={conformal_metrics['abstain']} of n={conformal_metrics['n']})"
    )
    if conformal_metrics["decisive_accuracy"] is not None:
        print(
            f"  decisive acc   = {conformal_metrics['decisive_accuracy']:.4f}  "
            f"(precision on non-ABSTAIN calls)"
        )
    print(f"  cal selfcheck  = coverage {sel_str}")
    print(f"wrote model -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
