#!/usr/bin/env python3
"""Information-theoretic leakage audit (F.3 diagnostic).

This is the runnable counterpart of `docs/F3_leakage_diagnostic.md`. It
estimates mutual information between each feature and the label, twice
— once on the data as-is, once with a domain-specific *mask* applied
— and flags features whose information collapses under the mask.

The mask encodes "what features will not be available at production
inference time." In the Sub-Zero subtitle case this is "anything
derived from a reference SRT." In a medical-risk case it would be
"anything recorded after admission." The diagnostic is domain-agnostic;
only the mask is domain-specific.

Mask selection
--------------
- `--mask-pattern subtitle` (default): names starting with `delta_` or
  `reference_`, or containing `similarity`. This is the original
  subtitle-QE mask the diagnostic was developed for.
- `--mask-features f1,f2,...`: explicit comma-separated feature list.
  Overrides any pattern. Use this for second-domain case studies.
- `--mask-pattern none`: no automatic masking; rely entirely on
  `--mask-features`. Useful when the masked set has no naming pattern.

Output
------
A JSON document at `--output` listing every feature with its scores,
sorted descending by univariate `mi_rich`. Features with
`leak_score_univariate >= --leak-threshold` are flagged in the
`flagged_features` array — these are the features the v2 retrain
(`train_learned_gate.py --feature-audit <path>`) will drop.

References
----------
- Mutual information estimation uses scikit-learn's
  `mutual_info_classif` — Ross 2014 kNN estimator, unbiased for
  continuous-discrete pairs and stable down to a few hundred samples.
- The Fano-derived Bayes-risk interpretation of the leak score is in
  `docs/F3_leakage_diagnostic.md` §3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    print("error: numpy required (`pip install numpy`)", file=sys.stderr)
    raise SystemExit(2) from exc

try:
    from sklearn.feature_selection import mutual_info_classif
except ImportError as exc:  # pragma: no cover
    print("error: scikit-learn required (`pip install scikit-learn`)", file=sys.stderr)
    raise SystemExit(2) from exc


# Built-in subtitle mask. Kept as the default so calling the script
# with no mask flags still does what every checked-in subtitle CI run
# expects. Match by *substring* so a future rename (e.g. `delta2_*` or
# `_reference_token_overlap`) is still caught.
SUBTITLE_REFERENCE_PREFIXES: tuple[str, ...] = ("delta_", "reference_")
SUBTITLE_REFERENCE_TOKENS: tuple[str, ...] = ("similarity", "_ref_", "reference",)


def is_subtitle_reference_derived(name: str) -> bool:
    lower = name.lower()
    if any(lower.startswith(p) for p in SUBTITLE_REFERENCE_PREFIXES):
        return True
    return any(tok in lower for tok in SUBTITLE_REFERENCE_TOKENS)


def build_mask_set(
    features: list[str],
    *,
    pattern: str,
    explicit: set[str],
) -> set[str]:
    """Return the set of feature names to zero in the masked regime.

    Combines a built-in pattern (`subtitle` or `none`) with an explicit
    user-supplied feature list. Names in `explicit` that are not in
    `features` are silently ignored — useful for shared mask files
    that target a superset of the current corpus.
    """
    masked: set[str] = set()
    if pattern == "subtitle":
        for name in features:
            if is_subtitle_reference_derived(name):
                masked.add(name)
    elif pattern == "none":
        pass
    else:
        raise SystemExit(f"unknown --mask-pattern: {pattern!r}")
    for name in explicit:
        if name in features:
            masked.add(name)
    return masked


@dataclass
class FeatureAudit:
    name: str
    # `mi_rich` / `mi_masked` are *joint-kNN* MI estimates from sklearn's
    # `mutual_info_classif`. They are biased on small samples and
    # depend on the geometry of the full feature space, not just the
    # column. Numerically convenient for ranking but the theory-lab
    # review (2026-05-22) flagged that the joint estimate can produce
    # the absurd result `mi_drop < 0` for a single feature whose value
    # is unchanged under the mask. The two univariate MI fields below
    # are unambiguous per-column estimates: `I(X_i; Y)` on `X_i` alone,
    # under both regimes. They are what `leak_score` should be derived
    # from when a reviewer asks "what does this number mean."
    mi_rich: float
    mi_masked: float
    mi_drop: float
    leak_score: float
    mi_rich_univariate: float
    mi_masked_univariate: float
    mi_drop_univariate: float
    leak_score_univariate: float
    masked_by_audit: bool


def load_corpus(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = []
    feature_union: list[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            obj = json.loads(raw)
            label = obj.get("label")
            feats = obj.get("features", {})
            if label not in (0, 1):
                raise SystemExit(f"{path}:{line_no}: label must be 0/1, got {label!r}")
            if not isinstance(feats, dict):
                raise SystemExit(f"{path}:{line_no}: 'features' must be an object")
            for k in feats:
                if k not in seen:
                    seen.add(k)
                    feature_union.append(k)
            rows.append((int(label), feats))
    if not rows:
        raise SystemExit(f"{path}: corpus is empty")
    y = np.array([r[0] for r in rows], dtype=np.int64)
    x = np.array(
        [[float(r[1].get(name, 0.0)) for name in feature_union] for r in rows],
        dtype=np.float64,
    )
    return x, y, feature_union


def mi_per_feature(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Joint-kNN MI estimate (sklearn default). Geometry-dependent —
    the estimate for feature `i` reflects neighbourhoods in the full
    feature space, not just the column. Useful for ranking; can produce
    paradoxical `mi_drop < 0` outcomes on masked columns. Pair with
    :func:`mi_univariate` for the clean per-column estimate.
    """
    if len(set(y.tolist())) < 2:
        return np.zeros(x.shape[1], dtype=np.float64)
    return mutual_info_classif(x, y, random_state=seed)


def mi_univariate(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Per-feature univariate MI: `I(X_i ; Y)` on column `i` alone.
    Unambiguous, never depends on other columns' geometry, and
    guaranteed `>= 0` (estimator noise aside). This is what the leakage
    diagnostic's `leak_score_univariate` is computed from.
    """
    if len(set(y.tolist())) < 2:
        return np.zeros(x.shape[1], dtype=np.float64)
    out = np.empty(x.shape[1], dtype=np.float64)
    for i in range(x.shape[1]):
        col = x[:, i : i + 1]
        out[i] = float(mutual_info_classif(col, y, random_state=seed)[0])
    return out


def hash_corpus(path: Path) -> str:
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for the MI kNN estimator"
    )
    parser.add_argument(
        "--leak-threshold",
        type=float,
        default=0.5,
        help="univariate leak_score above which a feature is flagged (default 0.5)",
    )
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="subtitle",
        choices=("subtitle", "none"),
        help=(
            "Built-in mask pattern. 'subtitle' (default) zeroes any "
            "feature whose name matches the reference-SRT-derived "
            "pattern. 'none' disables pattern masking — use with "
            "`--mask-features` for non-subtitle domains."
        ),
    )
    parser.add_argument(
        "--mask-features",
        type=str,
        default="",
        help=(
            "Comma-separated explicit feature names to mask. Combined "
            "(union) with the `--mask-pattern` selection."
        ),
    )
    args = parser.parse_args(argv)

    x, y, features = load_corpus(args.corpus)
    print(f"corpus: {len(y)} rows, {len(features)} features", file=sys.stderr)
    print(f"label balance: pos={int(y.sum())} neg={int((1 - y).sum())}", file=sys.stderr)

    # Resolve the mask set from the CLI options.
    explicit_mask: set[str] = {
        n.strip() for n in args.mask_features.split(",") if n.strip()
    }
    mask_set = build_mask_set(features, pattern=args.mask_pattern, explicit=explicit_mask)
    print(
        f"mask: pattern={args.mask_pattern!r} explicit={len(explicit_mask)} -> "
        f"masking {len(mask_set)} feature(s): {sorted(mask_set) if mask_set else '<none>'}",
        file=sys.stderr,
    )

    # Reference-rich regime: as-is.
    mi_rich = mi_per_feature(x, y, args.seed)
    mi_rich_uni = mi_univariate(x, y, args.seed)

    # Reference-masked regime: zero out masked features.
    x_masked = x.copy()
    masked_idx = [i for i, name in enumerate(features) if name in mask_set]
    for i in masked_idx:
        x_masked[:, i] = 0.0
    mi_masked = mi_per_feature(x_masked, y, args.seed)
    mi_masked_uni = mi_univariate(x_masked, y, args.seed)

    audits: list[FeatureAudit] = []
    for i, name in enumerate(features):
        rich = float(mi_rich[i])
        masked = float(mi_masked[i])
        drop = rich - masked
        leak = (drop / rich) if rich > 1e-9 else 0.0
        rich_u = float(mi_rich_uni[i])
        masked_u = float(mi_masked_uni[i])
        drop_u = rich_u - masked_u
        leak_u = (drop_u / rich_u) if rich_u > 1e-9 else 0.0
        audits.append(
            FeatureAudit(
                name=name,
                mi_rich=rich,
                mi_masked=masked,
                mi_drop=drop,
                leak_score=leak,
                mi_rich_univariate=rich_u,
                mi_masked_univariate=masked_u,
                mi_drop_univariate=drop_u,
                leak_score_univariate=leak_u,
                masked_by_audit=(name in mask_set),
            )
        )

    audits.sort(key=lambda a: a.mi_rich_univariate, reverse=True)

    # Flag on the univariate score — that's the unambiguous one. The
    # joint score is kept in the artifact for reviewer inspection.
    flagged = [a.name for a in audits if a.leak_score_univariate >= args.leak_threshold]

    out = {
        "schema_version": "1.1",
        "kind": "learned-gate-feature-audit",
        "corpus_path": str(args.corpus),
        "corpus_sha256": hash_corpus(args.corpus),
        "git_commit": git_commit(),
        "seed": args.seed,
        "leak_threshold": args.leak_threshold,
        "mask_pattern": args.mask_pattern,
        "mask_features_explicit": sorted(explicit_mask),
        "mask_resolved": sorted(mask_set),
        "n_rows": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "flagged_features": flagged,
        "features": [asdict(a) for a in audits],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    # Human-readable summary to stderr so > redirects keep the JSON
    # clean. Reports both univariate (the diagnostic) and joint (the
    # legacy ranking) so reviewers can cross-check.
    print(file=sys.stderr)
    print(
        f"{'feature':40} {'MI_uni':>8} {'MI_uni_m':>9} {'leak_u%':>8}  {'MI_joint':>9}  flag",
        file=sys.stderr,
    )
    print("-" * 90, file=sys.stderr)
    for a in audits:
        flag = "LEAK" if a.leak_score_univariate >= args.leak_threshold else ""
        print(
            f"{a.name:40} {a.mi_rich_univariate:8.4f} {a.mi_masked_univariate:9.4f} "
            f"{a.leak_score_univariate * 100:7.1f}% {a.mi_rich:9.4f}  {flag}",
            file=sys.stderr,
        )
    print(file=sys.stderr)
    print(f"flagged ({len(flagged)}): {', '.join(flagged) if flagged else '<none>'}", file=sys.stderr)
    print(f"wrote audit -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
