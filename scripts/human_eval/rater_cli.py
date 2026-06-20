#!/usr/bin/env python3
"""Interactive blind side-by-side rater for VoiDex outputs.

Resumable: ratings are appended to --out as soon as they are entered,
and pairs already rated by this rater are skipped on restart.

Per pair the rater enters:
    1. Preference: a / b / t (tie / can't tell).
    2. Optional Likert 1-5 for each candidate (blank to skip).

Commands:
    s       skip the current pair (no rating recorded)
    q       save and quit
    h       show help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


HELP_TEXT = """\
Commands:
  preference: a (A is better) / b (B is better) / t (tie or can't tell)
  Likert: 1..5 (5 = best). Blank = no Likert rating.
  s = skip pair, q = quit, h = help.
"""


def load_pairs(path: Path) -> list[dict]:
    pairs: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def already_rated(out_path: Path, rater: str) -> set[str]:
    if not out_path.exists():
        return set()
    rated: set[str] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("rater") == rater and isinstance(row.get("pair_id"), str):
                rated.add(row["pair_id"])
    return rated


def prompt(label: str, allowed: set[str], allow_blank: bool = False) -> str:
    while True:
        raw = input(label).strip().lower()
        if allow_blank and raw == "":
            return ""
        if raw in allowed:
            return raw
        print(f"  invalid input; expected one of {sorted(allowed)}")


def render_pair(pair: dict) -> None:
    print()
    print("=" * 72)
    print(f"pair_id: {pair['pair_id']}")
    print(f"case   : {pair['case']}")
    print("=" * 72)
    for i, s in enumerate(pair["samples"], 1):
        print(f"\n  cue {i}/{len(pair['samples'])}  [{s['start']} --> {s['end']}]")
        print(f"  A: {s['a_text']}")
        print(f"  B: {s['b_text']}")


def append_rating(out_path: Path, row: dict) -> None:
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="pairs.jsonl from prepare_pairs.py")
    parser.add_argument("--out", required=True, help="output ratings.jsonl (appended)")
    parser.add_argument("--rater", required=True, help="rater identifier (initials are fine)")
    args = parser.parse_args(argv)

    pairs_path = Path(args.pairs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(pairs_path)
    if not pairs:
        print("no pairs to rate", file=sys.stderr)
        return 2

    done = already_rated(out_path, args.rater)
    remaining = [p for p in pairs if p["pair_id"] not in done]

    print(f"loaded {len(pairs)} pairs; {len(done)} already rated; {len(remaining)} to go")
    print(HELP_TEXT)

    n_recorded = 0
    for pair in remaining:
        render_pair(pair)
        choice = prompt("\n  preference (a/b/t/s/q/h): ", {"a", "b", "t", "s", "q", "h"})
        if choice == "h":
            print(HELP_TEXT)
            choice = prompt("  preference (a/b/t/s/q): ", {"a", "b", "t", "s", "q"})
        if choice == "q":
            print(f"saved {n_recorded} new ratings; quitting")
            return 0
        if choice == "s":
            continue

        likert_a = prompt(
            "  A Likert 1-5 (blank=skip): ",
            {"1", "2", "3", "4", "5"},
            allow_blank=True,
        )
        likert_b = prompt(
            "  B Likert 1-5 (blank=skip): ",
            {"1", "2", "3", "4", "5"},
            allow_blank=True,
        )

        row = {
            "pair_id": pair["pair_id"],
            "case": pair["case"],
            "rater": args.rater,
            "preference": choice,  # 'a' | 'b' | 't'
            "likert_a": int(likert_a) if likert_a else None,
            "likert_b": int(likert_b) if likert_b else None,
            "rated_at_epoch_secs": int(time.time()),
        }
        append_rating(out_path, row)
        n_recorded += 1

    print(f"\ndone. recorded {n_recorded} new ratings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
