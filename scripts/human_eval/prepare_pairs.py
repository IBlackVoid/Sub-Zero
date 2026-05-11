#!/usr/bin/env python3
"""Sample blinded side-by-side comparison pairs for human evaluation.

For each case directory under --runs-dir, find all `*.en.srt` outputs
(one per pipeline variant). Build all unordered variant pairs, sample
N aligned cues per pair, randomise the A/B assignment, and emit:

    pairs.jsonl       — what the rater sees
    .truth/mapping.jsonl — A/B -> variant truth (for the aggregator)

The two files MUST live in different directories so a rater never
accidentally opens the truth file.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


SRT_TIMING = re.compile(
    r"^\s*(\d{2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{1,3})\s*$"
)


@dataclass(frozen=True)
class Cue:
    index: int
    start: str
    end: str
    text: str


def parse_srt(text: str) -> list[Cue]:
    """Forgiving SRT parser. Tolerates blank lines and missing indices."""
    cues: list[Cue] = []
    blocks = re.split(r"\r?\n\r?\n", text.strip())
    for block in blocks:
        lines = [ln.rstrip("\r") for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        # Allow the first line to be a numeric index OR the timing line.
        if SRT_TIMING.match(lines[0]):
            timing_line = lines[0]
            text_lines = lines[1:]
            index = len(cues) + 1
        else:
            try:
                index = int(lines[0].strip())
            except ValueError:
                continue
            if len(lines) < 2 or not SRT_TIMING.match(lines[1]):
                continue
            timing_line = lines[1]
            text_lines = lines[2:]
        m = SRT_TIMING.match(timing_line)
        if not m:
            continue
        cues.append(
            Cue(
                index=index,
                start=m.group(1).replace(".", ","),
                end=m.group(2).replace(".", ","),
                text="\n".join(text_lines).strip(),
            )
        )
    return cues


def find_variant_outputs(case_dir: Path) -> dict[str, Path]:
    """Map variant name -> *.en.srt path. Variant is the basename stem.

    Excludes `reference.en.srt`, which is the ground-truth subtitle, not
    a candidate variant.
    """
    out: dict[str, Path] = {}
    for srt in sorted(case_dir.glob("*.en.srt")):
        if srt.name == "reference.en.srt":
            continue
        # Strip trailing ".en" so e.g. "input.fast.en.srt" -> "input.fast".
        name = srt.name[: -len(".en.srt")]
        # Drop a leading "input." prefix to keep variant names short.
        if name.startswith("input."):
            name = name[len("input."):]
        if not name:
            name = srt.stem
        out[name] = srt
    return out


def align_cue_indices(
    cues_by_variant: dict[str, list[Cue]], k: int, rng: random.Random
) -> list[int]:
    """Pick k cue indices that exist in every variant's cue list."""
    if not cues_by_variant:
        return []
    common = None
    for cues in cues_by_variant.values():
        idxs = {c.index for c in cues}
        common = idxs if common is None else (common & idxs)
    if not common:
        return []
    common_sorted = sorted(common)
    if len(common_sorted) <= k:
        return common_sorted
    return sorted(rng.sample(common_sorted, k))


def cue_at(cues: list[Cue], target: int) -> Cue | None:
    for c in cues:
        if c.index == target:
            return c
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cues-per-pair", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = no cap")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    truth_dir = out_dir / ".truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    pairs_out = out_dir / "pairs.jsonl"
    truth_out = truth_dir / "mapping.jsonl"

    n_pairs = 0
    n_skipped = 0

    with pairs_out.open("w", encoding="utf-8") as pf, truth_out.open(
        "w", encoding="utf-8"
    ) as tf:
        for case_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            variants = find_variant_outputs(case_dir)
            if len(variants) < 2:
                continue

            cues_by_variant: dict[str, list[Cue]] = {}
            for name, path in variants.items():
                try:
                    cues_by_variant[name] = parse_srt(
                        path.read_text(encoding="utf-8", errors="replace")
                    )
                except OSError:
                    cues_by_variant[name] = []

            # Drop variants that yielded zero cues (failed runs).
            cues_by_variant = {k: v for k, v in cues_by_variant.items() if v}
            if len(cues_by_variant) < 2:
                n_skipped += 1
                continue

            names = sorted(cues_by_variant.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if args.max_pairs and n_pairs >= args.max_pairs:
                        break
                    var_a, var_b = names[i], names[j]
                    aligned = align_cue_indices(
                        {var_a: cues_by_variant[var_a], var_b: cues_by_variant[var_b]},
                        args.cues_per_pair,
                        rng,
                    )
                    if not aligned:
                        n_skipped += 1
                        continue

                    flip = rng.random() < 0.5
                    truth_a, truth_b = (var_b, var_a) if flip else (var_a, var_b)

                    pair_id = uuid.UUID(
                        int=rng.getrandbits(128), version=4
                    ).hex
                    samples = []
                    for idx in aligned:
                        ca = cue_at(cues_by_variant[truth_a], idx)
                        cb = cue_at(cues_by_variant[truth_b], idx)
                        if ca is None or cb is None:
                            continue
                        samples.append(
                            {
                                "cue_index": idx,
                                "start": ca.start,
                                "end": ca.end,
                                "a_text": ca.text,
                                "b_text": cb.text,
                            }
                        )
                    if not samples:
                        n_skipped += 1
                        continue

                    pf.write(
                        json.dumps(
                            {
                                "pair_id": pair_id,
                                "case": case_dir.name,
                                "samples": samples,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    tf.write(
                        json.dumps(
                            {
                                "pair_id": pair_id,
                                "case": case_dir.name,
                                "a_variant": truth_a,
                                "b_variant": truth_b,
                            }
                        )
                        + "\n"
                    )
                    n_pairs += 1
                if args.max_pairs and n_pairs >= args.max_pairs:
                    break
            if args.max_pairs and n_pairs >= args.max_pairs:
                break

    print(
        f"wrote {n_pairs} pairs to {pairs_out} (skipped {n_skipped} unalignable)"
    )
    print(f"wrote truth mapping to {truth_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
