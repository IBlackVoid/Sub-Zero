#!/usr/bin/env python3
"""Aggregate blind ratings into paper-ready statistics.

Inputs
------
--truth      mapping.jsonl produced by prepare_pairs.py
--ratings    one or more ratings.jsonl files (one per rater)

Outputs
-------
- summary.json next to the first ratings file
- a markdown table on stdout, ready to paste into the paper

Statistics
----------
- Per-variant pairwise win-rate with Wilson 95 % CI.
- Two-sided sign test (exact binomial) on each variant pair.
- Krippendorff's alpha across raters (interval scale, Likert).
- Per-variant Likert mean with bootstrapped 95 % CI for each pair's
  delta.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


# --------------------------------------------------------------------- #
# Statistics primitives — pure stdlib, no scipy.                        #
# --------------------------------------------------------------------- #


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    """Cumulative P(X <= k) for X ~ Binomial(n, p). Stable for small n."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    log_p = math.log(p) if p > 0 else float("-inf")
    log_q = math.log(1.0 - p) if p < 1 else float("-inf")
    log_factorial = [0.0] * (n + 1)
    for i in range(1, n + 1):
        log_factorial[i] = log_factorial[i - 1] + math.log(i)

    def log_binom(i: int) -> float:
        return log_factorial[n] - log_factorial[i] - log_factorial[n - i]

    # Sum exp(log_binom(i) + i*log_p + (n-i)*log_q) for i in [0, k]
    terms = [log_binom(i) + i * log_p + (n - i) * log_q for i in range(0, k + 1)]
    m = max(terms)
    return math.exp(m) * sum(math.exp(t - m) for t in terms)


def sign_test_two_sided(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p-value, ties already split out."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    one_sided = binom_cdf(k, n, 0.5)
    return min(1.0, 2.0 * one_sided)


def bootstrap_delta_ci(
    deltas: list[float], n_resamples: int = 10_000, seed: int = 1337
) -> tuple[float, float, float]:
    """Mean delta and percentile bootstrap 95 % CI."""
    if not deltas:
        return (0.0, 0.0, 0.0)
    if len(deltas) == 1:
        d = deltas[0]
        return (d, d, d)
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return (statistics.fmean(deltas), lo, hi)


def krippendorff_alpha_interval(
    ratings_by_unit: dict[str, dict[str, float]]
) -> float | None:
    """Krippendorff's alpha for interval data.

    Each unit (key of outer dict) maps {rater_id: rating}.
    Returns None if there is too little overlap to compute alpha.
    """
    # Flatten to list of (unit_id, rater_id, value) and compute the
    # required sums per Krippendorff (2011) §3.2.
    units = []
    all_values: list[float] = []
    for unit_id, ratings in ratings_by_unit.items():
        if len(ratings) < 2:
            continue
        units.append((unit_id, list(ratings.values())))
        all_values.extend(ratings.values())

    if len(units) < 2 or len(all_values) < 2:
        return None

    # Observed disagreement: weighted by 1 / (m_u - 1) per unit.
    obs_num = 0.0
    obs_den = 0.0
    for _, vals in units:
        m_u = len(vals)
        if m_u < 2:
            continue
        # Sum of squared pairwise differences within the unit.
        unit_sq_sum = 0.0
        for i in range(m_u):
            for j in range(i + 1, m_u):
                unit_sq_sum += (vals[i] - vals[j]) ** 2
        obs_num += unit_sq_sum / (m_u - 1)
        obs_den += m_u

    if obs_den == 0:
        return None
    d_o = obs_num / obs_den * 2.0  # factor 2 because each pair counted once

    # Expected disagreement: across-unit, all pair-distances.
    n = len(all_values)
    if n < 2:
        return None
    exp_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            exp_sum += (all_values[i] - all_values[j]) ** 2
    d_e = (2.0 * exp_sum) / (n * (n - 1))

    if d_e == 0:
        return 1.0 if d_o == 0 else None
    return 1.0 - d_o / d_e


# --------------------------------------------------------------------- #
# Aggregation pipeline                                                  #
# --------------------------------------------------------------------- #


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def aggregate(truth_rows: list[dict], rating_rows: list[dict]) -> dict:
    truth_by_pair: dict[str, dict] = {row["pair_id"]: row for row in truth_rows}

    # Pairwise win/loss/tie counts: variant_x vs variant_y -> wins for x.
    duels: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"x_wins": 0, "y_wins": 0, "ties": 0}
    )

    likert_by_unit: dict[str, dict[str, float]] = defaultdict(dict)
    likert_pairs: dict[tuple[str, str], list[float]] = defaultdict(list)

    for r in rating_rows:
        pid = r.get("pair_id")
        truth = truth_by_pair.get(pid)
        if truth is None:
            continue
        a_var = truth["a_variant"]
        b_var = truth["b_variant"]
        # Canonicalise the variant pair (alphabetical) so duel keys are stable.
        if a_var <= b_var:
            x, y = a_var, b_var
            a_is_x = True
        else:
            x, y = b_var, a_var
            a_is_x = False

        pref = r.get("preference")
        bucket = duels[(x, y)]
        if pref == "t":
            bucket["ties"] += 1
        elif pref == "a":
            bucket["x_wins" if a_is_x else "y_wins"] += 1
        elif pref == "b":
            bucket["y_wins" if a_is_x else "x_wins"] += 1

        la = r.get("likert_a")
        lb = r.get("likert_b")
        rater = r.get("rater") or "anon"
        if isinstance(la, (int, float)):
            unit_a = f"{pid}::A"
            likert_by_unit[unit_a][rater] = float(la)
        if isinstance(lb, (int, float)):
            unit_b = f"{pid}::B"
            likert_by_unit[unit_b][rater] = float(lb)

        if isinstance(la, (int, float)) and isinstance(lb, (int, float)):
            delta_x_minus_y = (
                (float(la) - float(lb)) if a_is_x else (float(lb) - float(la))
            )
            likert_pairs[(x, y)].append(delta_x_minus_y)

    # Build per-duel summary.
    duel_summary: list[dict] = []
    for (x, y), b in sorted(duels.items()):
        total = b["x_wins"] + b["y_wins"] + b["ties"]
        decisive = b["x_wins"] + b["y_wins"]
        # Split ties evenly when computing win-rate (standard practice).
        x_credit = b["x_wins"] + 0.5 * b["ties"]
        win_rate_x = (x_credit / total) if total else 0.0
        # Wilson uses integer counts; allocate ties half-and-half by rounding
        # on the side of x.
        x_int = b["x_wins"] + b["ties"] // 2 + (b["ties"] % 2)
        wilson = wilson_interval(x_int, total) if total else (0.0, 0.0)
        p_value = sign_test_two_sided(b["x_wins"], b["y_wins"])

        deltas = likert_pairs.get((x, y), [])
        mean_d, lo_d, hi_d = bootstrap_delta_ci(deltas)

        duel_summary.append(
            {
                "variant_x": x,
                "variant_y": y,
                "n_total": total,
                "n_decisive": decisive,
                "x_wins": b["x_wins"],
                "y_wins": b["y_wins"],
                "ties": b["ties"],
                "win_rate_x": win_rate_x,
                "wilson_95_ci": [wilson[0], wilson[1]],
                "sign_test_p_two_sided": p_value,
                "likert_delta_x_minus_y_mean": mean_d,
                "likert_delta_x_minus_y_95_ci": [lo_d, hi_d],
                "likert_delta_n": len(deltas),
            }
        )

    alpha = krippendorff_alpha_interval(likert_by_unit)
    return {
        "duels": duel_summary,
        "krippendorff_alpha_interval": alpha,
        "n_truth_pairs": len(truth_by_pair),
        "n_ratings": len(rating_rows),
    }


def render_markdown(summary: dict) -> str:
    lines: list[str] = []
    alpha = summary.get("krippendorff_alpha_interval")
    alpha_disp = "n/a" if alpha is None else f"{alpha:.3f}"
    lines.append(
        f"## Human evaluation summary  (n_pairs={summary['n_truth_pairs']}, "
        f"n_ratings={summary['n_ratings']}, Krippendorff alpha={alpha_disp})"
    )
    lines.append("")
    lines.append(
        "| variant_x | variant_y | n | win_rate_x | 95% CI | sign-test p | "
        "Likert delta (x-y) | 95% CI |"
    )
    lines.append(
        "|-----------|-----------|---|------------|--------|-------------|"
        "--------------------|--------|"
    )
    for d in summary["duels"]:
        ci = d["wilson_95_ci"]
        dci = d["likert_delta_x_minus_y_95_ci"]
        delta_disp = (
            f"{d['likert_delta_x_minus_y_mean']:+.2f} (n={d['likert_delta_n']})"
            if d["likert_delta_n"]
            else "n/a"
        )
        lines.append(
            f"| {d['variant_x']} | {d['variant_y']} | {d['n_total']} | "
            f"{d['win_rate_x']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{d['sign_test_p_two_sided']:.4f} | {delta_disp} | "
            f"[{dci[0]:+.2f}, {dci[1]:+.2f}] |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--ratings", required=True, nargs="+")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    truth_path = Path(args.truth)
    truth_rows = load_jsonl(truth_path)
    if not truth_rows:
        print(f"error: no truth rows in {truth_path}", file=sys.stderr)
        return 2

    rating_rows: list[dict] = []
    for p in args.ratings:
        rating_rows.extend(load_jsonl(Path(p)))
    if not rating_rows:
        print("error: no ratings loaded", file=sys.stderr)
        return 2

    summary = aggregate(truth_rows, rating_rows)
    md = render_markdown(summary)
    print(md)

    out_path = Path(args.out) if args.out else Path(args.ratings[0]).parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
