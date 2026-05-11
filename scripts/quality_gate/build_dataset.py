#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _norm_path(p: str) -> str:
    # Portability: reports emit forward slashes; Windows sidecars emit backslashes.
    # Normalize both sides before lookup so cross-OS corpora match.
    s = p.replace("\\", "/").strip()
    while s.startswith("./"):
        s = s[2:]
    return s


def find_metadata_sidecars(search_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in search_dir.rglob("*.sub-zero.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue
        output_file = str(payload.get("output_file") or "").strip()
        if not output_file:
            continue
        # Keep the first-seen payload for a given key; in corpus mode we avoid name collisions
        # by using full run-relative output paths, but defensive behavior is still better.
        out.setdefault(_norm_path(output_file), payload)
    return out


def extract_features(meta: dict[str, Any]) -> dict[str, float]:
    q = meta.get("quality") or {}
    structural = (q.get("structural") or {}) if isinstance(q, dict) else {}
    semantic = (q.get("semantic") or {}) if isinstance(q, dict) else {}

    def fget(obj: dict[str, Any], key: str) -> float:
        v = obj.get(key)
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    features = {
        "cue_count": float(q.get("cue_count") or 0.0),
        "top_line_ratio": fget(structural, "top_line_ratio"),
        "overlap_ratio": fget(structural, "overlap_ratio"),
        "non_empty_ratio": fget(structural, "non_empty_ratio"),
        "anomaly_ratio": fget(semantic, "anomaly_ratio"),
        "malformed_contraction_ratio": fget(semantic, "malformed_contraction_ratio"),
        "low_function_word_ratio": fget(semantic, "low_function_word_ratio"),
        "adjacent_repeat_ratio": fget(semantic, "adjacent_repeat_ratio"),
        "scene_low_quality_ratio": fget(semantic, "scene_low_quality_ratio"),
        "scene_count": fget(semantic, "scene_count"),
        "name_inconsistency_ratio": fget(semantic, "name_inconsistency_ratio"),
    }

    speaker = meta.get("speaker") or {}
    if isinstance(speaker, dict):
        register = speaker.get("register") or {}
        if isinstance(register, dict):
            features["register_speakers_observed"] = fget(register, "speakers_observed")
            features["register_speakers_formal"] = fget(register, "speakers_formal")
            features["register_cues_labeled"] = fget(register, "cues_labeled")

        diar = speaker.get("audio_diarization") or {}
        if isinstance(diar, dict):
            features["diar_speakers"] = fget(diar, "speakers")
            features["diar_used_segments"] = fget(diar, "used_segments")
            features["diar_assigned_cues"] = fget(diar, "assigned_cues")

    return features


# Report-side metrics worth using as pairwise features. These are the numeric
# signals the benchmark itself uses to rank candidates, so deltas from the
# case median are highly discriminative for "is this the best variant".
PAIRWISE_METRIC_KEYS = (
    "time_coverage_ratio",
    "weighted_timing_iou",
    "non_empty_rate",
    "reference_char_similarity",
    "line_char_similarity",
    "token_overlap_f1",
    "weighted_token_f1",
    "cue_density_ratio",
    "japanese_char_ratio",
)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def compute_case_medians(candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Median of each pairwise metric across all candidates in a single report."""
    medians: dict[str, float] = {}
    for key in PAIRWISE_METRIC_KEYS:
        vals: list[float] = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            metrics = cand.get("metrics") or {}
            v = metrics.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        medians[key] = _median(vals)
    return medians


def compute_case_stats(candidates: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-metric (median, max, min, sorted-values) across candidates of one case."""
    stats: dict[str, dict[str, float]] = {}
    for key in PAIRWISE_METRIC_KEYS:
        vals: list[float] = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            metrics = cand.get("metrics") or {}
            v = metrics.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if not vals:
            stats[key] = {"median": 0.0, "max": 0.0, "min": 0.0, "n": 0.0}
            continue
        vals_sorted = sorted(vals)
        stats[key] = {
            "median": _median(vals),
            "max": float(vals_sorted[-1]),
            "min": float(vals_sorted[0]),
            "n": float(len(vals_sorted)),
            # Pre-compute ranks for O(1) lookup later.
            "_sorted": vals_sorted,  # type: ignore[dict-item]
        }
    return stats


def _rank_in_case(value: float, sorted_vals: list[float]) -> float:
    """Fractional rank in [0, 1]; 1.0 = best in case, 0.0 = worst."""
    n = len(sorted_vals)
    if n <= 1:
        return 0.5
    # Number of strictly-smaller values; ties get a midpoint score.
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    rank_low = lo
    while hi < n and sorted_vals[hi] == value:
        hi += 1
    rank_high = hi
    midrank = (rank_low + rank_high - 1) / 2.0
    return midrank / (n - 1)


def add_pairwise_features(
    features: dict[str, float],
    metrics: dict[str, Any],
    case_stats: dict[str, dict[str, float]],
) -> None:
    """Append delta_<metric> = candidate value - case median.

    Note: a richer feature set (delta_max, delta_min, fractional rank)
    was evaluated and overfit on a 175-row dataset (held-out F1 dropped
    from 0.868 to 0.829 across 5 seeds). Reintroduce those features
    only when the corpus crosses ~500 rows.
    """
    for key in PAIRWISE_METRIC_KEYS:
        v = metrics.get(key)
        if not isinstance(v, (int, float)):
            continue
        median = float((case_stats.get(key) or {}).get("median", 0.0))
        features[f"delta_{key}"] = float(v) - median


def add_variant_onehot(features: dict[str, float], variant: str) -> None:
    """One-hot encode the candidate variant string. Unknown variants are ignored."""
    # Sanitise the variant string into a feature-key-safe slug.
    slug = "".join(ch if ch.isalnum() else "_" for ch in variant.lower())
    if slug:
        features[f"variant_{slug}"] = 1.0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report",
        action="append",
        default=[],
        help="benchmarks/reports/*.json (repeatable)",
    )
    ap.add_argument(
        "--reports-dir",
        help="directory containing benchmark reports (*.json); combines with any --report args",
    )
    ap.add_argument(
        "--search-dir",
        default=".",
        help="directory to scan for *.sub-zero.json metadata sidecars (default: .)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="output dataset JSONL path",
    )
    ap.add_argument(
        "--label",
        default="pass",
        choices=["pass", "ranking_top1", "ranking_topk"],
        help=(
            "labeling mode (default: pass). "
            "pass uses report metrics.pass; "
            "ranking_top1 labels the top-ranked candidate that has a matching metadata sidecar; "
            "ranking_topk labels the top-k ranked candidates (filtered to those with matching metadata sidecars)"
        ),
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="k for ranking_topk (default: 1)",
    )
    ap.add_argument(
        "--require-positive",
        action="store_true",
        help="fail if the dataset has 0 positive rows (recommended for training)",
    )
    args = ap.parse_args(argv)

    report_paths: list[Path] = [Path(p) for p in (args.report or [])]
    if args.reports_dir:
        reports_dir = Path(args.reports_dir)
        if not reports_dir.is_dir():
            print(f"error: reports dir does not exist: {reports_dir}", file=sys.stderr)
            return 2
        report_paths.extend(sorted(reports_dir.glob("*.json")))

    if not report_paths:
        print("error: at least one --report or --reports-dir must be provided", file=sys.stderr)
        return 2

    search_dir = Path(args.search_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    positive_set: set[str] = set()

    meta_by_output = find_metadata_sidecars(search_dir)
    if not meta_by_output:
        print(
            f"error: no metadata sidecars found under {search_dir} (expected *.sub-zero.json)",
            file=sys.stderr,
        )
        return 2

    rows = 0
    misses = 0
    positives = 0
    with out_path.open("w", encoding="utf-8") as f:
        for report_path in report_paths:
            report = load_json(report_path)
            candidates = report.get("candidates") or []
            if not isinstance(candidates, list) or not candidates:
                continue

            # Compute per-report stats once so pairwise features are O(C) not O(C^2).
            case_stats = compute_case_stats(candidates)

            ranking = report.get("ranking") or []
            positive_set = set()
            if args.label in ("ranking_top1", "ranking_topk"):
                if not isinstance(ranking, list) or not ranking:
                    continue

                top_k = int(args.top_k)
                if top_k <= 0:
                    print("error: --top-k must be >= 1", file=sys.stderr)
                    return 2

                def has_matching_metadata(candidate_name: str) -> bool:
                    norm = _norm_path(candidate_name)
                    if norm in meta_by_output:
                        return True
                    return any(k.endswith(norm) for k in meta_by_output.keys())

                available_ranked: list[str] = []
                for item in ranking:
                    if not isinstance(item, str):
                        continue
                    name = item.strip()
                    if not name:
                        continue
                    if has_matching_metadata(name):
                        available_ranked.append(name)

                if available_ranked:
                    if args.label == "ranking_top1":
                        positive_set = {available_ranked[0]}
                    else:
                        positive_set = set(
                            available_ranked[: min(top_k, len(available_ranked))]
                        )

            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                name = str(cand.get("name") or "").strip()
                metrics = cand.get("metrics") or {}
                if not name or not isinstance(metrics, dict):
                    continue

                # Match by exact output_file first, then suffix match for portability.
                norm_name = _norm_path(name)
                meta = meta_by_output.get(norm_name)
                if meta is None:
                    meta = next(
                        (m for k, m in meta_by_output.items() if k.endswith(norm_name)),
                        None,
                    )
                if meta is None:
                    misses += 1
                    continue

                if args.label in ("ranking_top1", "ranking_topk"):
                    label = 1 if name in positive_set else 0
                else:
                    label_bool = metrics.get("pass")
                    if not isinstance(label_bool, bool):
                        continue
                    label = 1 if label_bool else 0

                features = extract_features(meta)
                add_pairwise_features(features, metrics, case_stats)
                variant = str(cand.get("variant") or "").strip()
                if variant:
                    add_variant_onehot(features, variant)

                row = {
                    "candidate": name,
                    "label": int(label),
                    "features": features,
                    "report_metrics": metrics,
                    "variant": variant,
                    "report": str(report_path),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
                positives += int(label)

    if args.require_positive and positives == 0:
        print(
            "error: dataset has 0 positives; use --label ranking_top1/ranking_topk or adjust the benchmark pass thresholds",
            file=sys.stderr,
        )
        return 2

    print(
        f"wrote {rows} rows (positives {positives}, missed {misses} candidates)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
