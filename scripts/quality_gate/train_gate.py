#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def split_by_case(
    rows: list[dict[str, Any]], holdout_frac: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group rows by report (case), then split case-wise to avoid leakage.

    All variants of the same source media must go to the same partition;
    otherwise the model learns case-specific patterns and held-out
    accuracy becomes a useless estimate of generalisation.
    """
    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        case = r.get("report") or "<unknown>"
        by_case.setdefault(case, []).append(r)
    cases = sorted(by_case.keys())
    rng = random.Random(seed)
    rng.shuffle(cases)
    n_holdout = max(1, int(round(len(cases) * holdout_frac)))
    holdout_cases = set(cases[:n_holdout])
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for case, group in by_case.items():
        (test_rows if case in holdout_cases else train_rows).extend(group)
    return train_rows, test_rows


def rows_to_arrays(
    rows: list[dict[str, Any]], feature_keys: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    x_list: list[list[float]] = []
    y_list: list[int] = []
    for r in rows:
        feats = r.get("features") or {}
        if not isinstance(feats, dict):
            continue
        y = r.get("label")
        if y not in (0, 1):
            continue
        x_list.append([float(feats.get(k) or 0.0) for k in feature_keys])
        y_list.append(int(y))
    return (
        np.asarray(x_list, dtype=np.float64),
        np.asarray(y_list, dtype=np.float64),
    )


def metrics_from(p: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    eps = 1e-9
    loss = -float(np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
    preds = (p >= threshold).astype(np.int32)
    y_int = y.astype(np.int32)
    acc = float(np.mean(preds == y_int))
    tp = int(((preds == 1) & (y_int == 1)).sum())
    fp = int(((preds == 1) & (y_int == 0)).sum())
    fn = int(((preds == 0) & (y_int == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"loss": loss, "acc": acc, "precision": prec, "recall": rec, "f1": f1}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset JSONL from build_dataset.py")
    ap.add_argument("--out", required=True, help="output model JSON path")
    ap.add_argument("--lr", type=float, default=0.25, help="learning rate (default: 0.25)")
    ap.add_argument("--steps", type=int, default=800, help="gradient steps (default: 800)")
    ap.add_argument("--l2", type=float, default=1e-3, help="L2 regularization (default: 1e-3)")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="decision threshold stored in the model (default: 0.5)",
    )
    ap.add_argument(
        "--holdout-frac",
        type=float,
        default=0.0,
        help=(
            "fraction of cases (not rows) reserved for held-out evaluation. "
            "0 disables splitting (full-corpus training). default: 0"
        ),
    )
    ap.add_argument(
        "--holdout-seed",
        type=int,
        default=1337,
        help="seed for the case shuffle (default: 1337)",
    )
    args = ap.parse_args(argv)

    rows = load_rows(Path(args.data))
    if not rows:
        print("error: dataset is empty", file=sys.stderr)
        return 2

    feature_keys = sorted({k for r in rows for k in (r.get("features") or {}).keys()})
    if not feature_keys:
        print("error: no features found in dataset", file=sys.stderr)
        return 2

    if args.holdout_frac > 0.0:
        train_rows, test_rows = split_by_case(rows, args.holdout_frac, args.holdout_seed)
    else:
        train_rows, test_rows = rows, []

    x_train, y_train = rows_to_arrays(train_rows, feature_keys)
    x_test, y_test = rows_to_arrays(test_rows, feature_keys)
    if x_train.size == 0 or y_train.size == 0:
        print("error: training set has no usable rows", file=sys.stderr)
        return 2

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos < 2 or n_neg < 2:
        print(
            f"error: need at least 2 positives and 2 negatives to train; got "
            f"pos={n_pos} neg={n_neg}. Extend the corpus or relax labeling.",
            file=sys.stderr,
        )
        return 2

    # Standardize features using train-set stats only (avoid test-set leakage).
    mean = x_train.mean(axis=0)
    raw_std = x_train.std(axis=0)
    if not np.any(raw_std >= 1e-9):
        print(
            "error: all features are constant across rows (zero variance). "
            "The dataset is degenerate — likely duplicated metadata sidecars.",
            file=sys.stderr,
        )
        return 2
    std = np.where(raw_std < 1e-9, 1.0, raw_std)
    xs_train = (x_train - mean) / std
    xs_test = (x_test - mean) / std if x_test.size > 0 else x_test

    w = np.zeros((xs_train.shape[1],), dtype=np.float64)
    b = 0.0

    lr = float(args.lr)
    steps = int(args.steps)
    l2 = float(args.l2)
    threshold = float(args.threshold)

    print(
        f"train: n={x_train.shape[0]} pos={n_pos} neg={n_neg} features={len(feature_keys)}",
        file=sys.stderr,
    )
    if x_test.size > 0:
        print(
            f"holdout: n={x_test.shape[0]} pos={int(y_test.sum())} "
            f"neg={int(len(y_test) - y_test.sum())} cases_seed={args.holdout_seed}",
            file=sys.stderr,
        )

    for step in range(steps):
        logits = xs_train @ w + b
        p = sigmoid(logits)
        err = p - y_train
        grad_w = (xs_train.T @ err) / xs_train.shape[0] + l2 * w
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b

        if step in (0, 9, 49, 99, 199, 399, steps - 1):
            train_m = metrics_from(p, y_train, threshold)
            line = f"step={step} train: loss={train_m['loss']:.4f} acc={train_m['acc']:.3f} f1={train_m['f1']:.3f}"
            if x_test.size > 0:
                p_test = sigmoid(xs_test @ w + b)
                test_m = metrics_from(p_test, y_test, threshold)
                line += (
                    f"  ||  holdout: loss={test_m['loss']:.4f} acc={test_m['acc']:.3f}"
                    f" f1={test_m['f1']:.3f} prec={test_m['precision']:.3f}"
                    f" rec={test_m['recall']:.3f}"
                )
            print(line, file=sys.stderr)

    model = {
        "version": "1.1",
        "kind": "learned-quality-gate",
        "threshold": threshold,
        "bias": float(b),
        "features": feature_keys,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": w.tolist(),
    }

    if x_test.size > 0:
        p_test = sigmoid(xs_test @ w + b)
        final = metrics_from(p_test, y_test, threshold)
        model["holdout_metrics"] = final
        model["holdout_size"] = int(x_test.shape[0])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
