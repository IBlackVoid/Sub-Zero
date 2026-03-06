#!/usr/bin/env python3
"""Evaluate subtitle output quality against a reference SRT.

The evaluator is intentionally segmentation-agnostic:
- cues are aligned by timeline overlap, not by index equality
- metrics are overlap-weighted to remain stable across different chunking styles
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class Cue:
    start: float
    end: float
    timing: str
    text: str


def parse_timestamp(ts: str) -> float:
    hms, ms = ts.strip().split(",")
    hours, minutes, seconds = hms.split(":")
    return (
        int(hours) * 3600.0
        + int(minutes) * 60.0
        + int(seconds)
        + int(ms) / 1000.0
    )


def parse_timing_range(timing: str) -> Tuple[float, float]:
    start_raw, end_raw = timing.split("-->")
    start = parse_timestamp(start_raw.strip())
    end = parse_timestamp(end_raw.strip())
    if end < start:
        start, end = end, start
    return start, end


def parse_srt(path: Path) -> List[Cue]:
    content = path.read_text(encoding="utf-8-sig")
    normalized = content.replace("\r\n", "\n")
    cues: List[Cue] = []
    for block in normalized.split("\n\n"):
        trimmed = block.strip()
        if not trimmed:
            continue
        lines = [line.rstrip() for line in trimmed.splitlines()]
        if len(lines) < 3:
            continue
        try:
            start, end = parse_timing_range(lines[1].strip())
        except Exception:
            continue
        cues.append(
            Cue(
                start=start,
                end=end,
                timing=lines[1].strip(),
                text=" ".join(line.strip() for line in lines[2:] if line.strip()),
            )
        )
    return cues


def jp_ratio(text: str) -> float:
    total = 0
    jp = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        is_jp = (
            "\u3040" <= ch <= "\u309F"
            or "\u30A0" <= ch <= "\u30FF"
            or "\u4E00" <= ch <= "\u9FFF"
        )
        if is_jp:
            jp += 1
    return 0.0 if total == 0 else jp / total


def token_f1(reference: str, hypothesis: str) -> float:
    token_re = re.compile(r"[a-z0-9']+")
    ref_tokens = token_re.findall(reference.lower())
    hyp_tokens = token_re.findall(hypothesis.lower())
    if not ref_tokens and not hyp_tokens:
        return 1.0
    ref_counts = {}
    hyp_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in hyp_tokens:
        hyp_counts[token] = hyp_counts.get(token, 0) + 1
    common = 0
    for token, ref_count in ref_counts.items():
        common += min(ref_count, hyp_counts.get(token, 0))
    precision = common / max(1, len(hyp_tokens))
    recall = common / max(1, len(ref_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end <= start:
        return 0.0
    return end - start


def duration(cue: Cue) -> float:
    return max(0.001, cue.end - cue.start)


def align_by_time(
    reference: List[Cue], hypothesis: List[Cue]
) -> List[Tuple[Cue, Cue, float, float]]:
    """Return aligned pairs: (ref, hyp, overlap_secs, iou)."""
    pairs: List[Tuple[Cue, Cue, float, float]] = []
    if not reference or not hypothesis:
        return pairs

    ref_idx = 0
    for hyp in hypothesis:
        while ref_idx < len(reference) and reference[ref_idx].end <= hyp.start:
            ref_idx += 1

        best: Optional[Tuple[Cue, float, float]] = None
        probe_idx = ref_idx
        while probe_idx < len(reference) and reference[probe_idx].start < hyp.end:
            ref = reference[probe_idx]
            ov = overlap_seconds(ref.start, ref.end, hyp.start, hyp.end)
            if ov > 0.0:
                union = max(ref.end, hyp.end) - min(ref.start, hyp.start)
                iou = ov / max(0.001, union)
                if best is None or ov > best[1]:
                    best = (ref, ov, iou)
            probe_idx += 1

        if best is None:
            # If no overlap exists, align to nearest center cue for loose comparison.
            hyp_center = (hyp.start + hyp.end) * 0.5
            nearest = min(
                reference,
                key=lambda ref: abs(((ref.start + ref.end) * 0.5) - hyp_center),
            )
            best = (nearest, 0.0, 0.0)

        ref, ov, iou = best
        pairs.append((ref, hyp, ov, iou))
    return pairs


def evaluate(reference_path: Path, hypothesis_path: Path) -> dict:
    reference = parse_srt(reference_path)
    hypothesis = parse_srt(hypothesis_path)

    non_empty = sum(1 for cue in hypothesis if cue.text.strip())
    pairs = align_by_time(reference, hypothesis)
    aligned = len(pairs)

    ref_joined = " ".join(cue.text for cue in reference)
    hyp_joined = " ".join(cue.text for cue in hypothesis)

    overlap_weight_sum = 0.0
    overlap_line_similarity = 0.0
    overlap_token_f1 = 0.0
    overlap_iou = 0.0
    overlap_seconds_total = 0.0
    hyp_duration_total = sum(duration(cue) for cue in hypothesis)
    for ref, hyp, ov, iou in pairs:
        weight = max(0.05, ov / duration(hyp))
        overlap_weight_sum += weight
        overlap_seconds_total += ov
        overlap_line_similarity += SequenceMatcher(
            None, ref.text.lower(), hyp.text.lower()
        ).ratio() * weight
        overlap_token_f1 += token_f1(ref.text, hyp.text) * weight
        overlap_iou += iou * weight

    weighted_line_similarity = (
        overlap_line_similarity / max(0.001, overlap_weight_sum)
    )
    weighted_token_f1 = overlap_token_f1 / max(0.001, overlap_weight_sum)
    weighted_timing_iou = overlap_iou / max(0.001, overlap_weight_sum)
    time_coverage_ratio = overlap_seconds_total / max(0.001, hyp_duration_total)

    ref_cues_per_hour = len(reference) / max(1.0, (reference[-1].end / 3600.0 if reference else 1.0))
    hyp_cues_per_hour = len(hypothesis) / max(1.0, (hypothesis[-1].end / 3600.0 if hypothesis else 1.0))
    cue_density_ratio = hyp_cues_per_hour / max(1.0, ref_cues_per_hour)

    result = {
        "reference_cues": len(reference),
        "hypothesis_cues": len(hypothesis),
        "cue_count_match": len(reference) == len(hypothesis),
        "aligned_pairs": aligned,
        "time_coverage_ratio": time_coverage_ratio,
        "weighted_timing_iou": weighted_timing_iou,
        "non_empty_rate": non_empty / max(1, len(hypothesis)),
        "reference_char_similarity": SequenceMatcher(
            None,
            ref_joined.lower(),
            hyp_joined.lower(),
        ).ratio(),
        "line_char_similarity": weighted_line_similarity,
        "token_overlap_f1": token_f1(ref_joined, hyp_joined),
        "weighted_token_f1": weighted_token_f1,
        "cue_density_ratio": cue_density_ratio,
        "japanese_char_ratio": jp_ratio(hyp_joined),
    }
    return result


def passes(
    metrics: dict,
    min_line_similarity: float,
    max_japanese_ratio: float,
    min_weighted_token_f1: float,
    min_time_coverage: float,
) -> bool:
    return (
        metrics["non_empty_rate"] >= 0.999
        and metrics["time_coverage_ratio"] >= min_time_coverage
        and metrics["line_char_similarity"] >= min_line_similarity
        and metrics["weighted_token_f1"] >= min_weighted_token_f1
        and metrics["japanese_char_ratio"] <= max_japanese_ratio
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Subtitle quality evaluator")
    parser.add_argument("--reference", required=True, help="Reference SRT path")
    parser.add_argument("--hypothesis", required=True, help="Hypothesis SRT path")
    parser.add_argument(
        "--min-line-similarity",
        type=float,
        default=0.20,
        help="Minimum line-level similarity threshold",
    )
    parser.add_argument(
        "--max-japanese-ratio",
        type=float,
        default=0.20,
        help="Maximum allowed Japanese character ratio in hypothesis",
    )
    parser.add_argument(
        "--min-weighted-token-f1",
        type=float,
        default=0.35,
        help="Minimum overlap-weighted token F1 threshold",
    )
    parser.add_argument(
        "--min-time-coverage",
        type=float,
        default=0.75,
        help="Minimum hypothesis-vs-reference timeline coverage ratio",
    )
    parser.add_argument(
        "--fail-on-low-quality",
        action="store_true",
        help="Exit non-zero when quality gates fail",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference)
    hypothesis_path = Path(args.hypothesis)
    if not reference_path.is_file():
        print(f"missing reference: {reference_path}", file=sys.stderr)
        return 2
    if not hypothesis_path.is_file():
        print(f"missing hypothesis: {hypothesis_path}", file=sys.stderr)
        return 2

    metrics = evaluate(reference_path, hypothesis_path)
    metrics["pass"] = passes(
        metrics,
        min_line_similarity=args.min_line_similarity,
        max_japanese_ratio=args.max_japanese_ratio,
        min_weighted_token_f1=args.min_weighted_token_f1,
        min_time_coverage=args.min_time_coverage,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.fail_on_low_quality and not metrics["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
