#!/usr/bin/env python3
"""Generate a synthetic recommender-system click-through corpus for the
F.3 third-domain case study.

The motivation: subtitle QE (case study 1) and medical mortality
(case study 2) are both "evaluate the output of a model" problems.
Recsys is structurally different — it's a *ranking* problem, the label
is downstream user behaviour, and the canonical leakage pattern is
**post-click features bleeding into the pre-click scoring model**. A
naive ML pipeline that pulls features from the production event stream
will include features (dwell time, scroll depth, conversion) that
*by definition* can only be observed after the click decision was
made. Training on them inflates offline metrics; at inference these
features default to zero and the model collapses.

This case study constructs exactly that scenario with documented
ground truth, then runs the F.3 diagnostic — *unchanged* from the
subtitle and medical pipelines — and verifies it catches the
post-click features at 100% precision/recall.

Domain
------
Predicting whether a user clicks a recommended item, at scoring time.

**Pre-click features** (8 — available the moment the recommendation is
scored, "what we know about the user and the item"):
  user_age_bucket, user_lifetime_days, user_prior_session_count,
  item_category_match_score, item_price_bucket, item_freshness_days,
  item_popularity_log, session_dwell_time_so_far.

**Post-click features** (5 — only observable AFTER the user clicked;
**the canonical time-traveling leakers**):
  click_dwell_time, scroll_depth, downstream_conversion,
  add_to_cart, subsequent_purchase_amount.

Generation
----------
We synthesise a candidate impression via:

    latent_interest ~ Beta(2, 4)            # ground-truth user-item fit
    clicked = Bernoulli(σ(6 * latent_interest − 2.5))
    (and add per-feature noise)

Pre-click features are weak-to-moderate functions of `latent_interest`.
Post-click features only have a distribution conditional on `clicked=1`
(zero otherwise, mirroring the "no event no feature" reality).

The corpus is balanced toward the click-positive class (~30 % click
rate is roughly aligned with realistic ad/recommendation CTR for
in-domain items).

Output: `corpus.jsonl` in the same schema the F.3 audit and v2
training harness already consume — no schema invention, no
recsys-specific tooling needed at the audit layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


PRE_CLICK_FEATURES = (
    "user_age_bucket",
    "user_lifetime_days",
    "user_prior_session_count",
    "item_category_match_score",
    "item_price_bucket",
    "item_freshness_days",
    "item_popularity_log",
    "session_dwell_time_so_far",
)

# These 5 features only exist *after* a click happens. Including them
# at training time is the canonical recsys leakage bug (Kaufman et al.
# 2012). At inference (scoring time) they are forced to zero, and the
# model trained on rich training-time values fits to noise.
POST_CLICK_FEATURES = (
    "click_dwell_time",
    "scroll_depth",
    "downstream_conversion",
    "add_to_cart",
    "subsequent_purchase_amount",
)


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def gen_impression(rng: random.Random) -> dict:
    # Latent ground-truth user-item interest in [0, 1].
    a, b = 2.0, 4.0
    g_a = sum(-math.log(1.0 - rng.random()) for _ in range(int(a)))
    g_b = sum(-math.log(1.0 - rng.random()) for _ in range(int(b)))
    latent_interest = g_a / (g_a + g_b) if (g_a + g_b) > 0 else 0.5
    latent_interest = max(0.0, min(1.0, latent_interest + rng.gauss(0.0, 0.05)))

    p_click = sigmoid(6.0 * latent_interest - 2.5)
    clicked = 1 if rng.random() < p_click else 0
    label = clicked  # 1 = clicked, 0 = did not click

    # Pre-click features — moderate signal.
    user_age_bucket = max(0, min(7, int(round(2 + 3 * latent_interest + rng.gauss(0, 1)))))
    user_lifetime_days = max(0, int(round(120 + 200 * latent_interest + rng.gauss(0, 50))))
    user_prior_session_count = max(0, int(round(3 + 18 * latent_interest + rng.gauss(0, 4))))
    item_category_match_score = round(
        max(0.0, min(1.0, 0.2 + 0.6 * latent_interest + rng.gauss(0, 0.08))), 3
    )
    item_price_bucket = max(0, min(9, int(round(4 - 2 * latent_interest + rng.gauss(0, 1.5)))))
    item_freshness_days = max(0, int(round(30 - 22 * latent_interest + rng.gauss(0, 8))))
    item_popularity_log = round(
        max(0.0, 2.0 + 4.0 * latent_interest + rng.gauss(0, 0.5)), 3
    )
    session_dwell_time_so_far = round(
        max(0.0, 30.0 + 100.0 * latent_interest + rng.gauss(0, 15)), 2
    )

    # Post-click features — only meaningful when clicked. The leakage
    # path: they correlate strongly with the realised label because
    # they only fire on the positive class. At inference these values
    # are unavailable (the click decision is what we're predicting),
    # so they default to zero — and the model's reliance on them at
    # training time becomes the leakage symptom.
    if clicked == 1:
        click_dwell_time = round(
            max(0.5, 5.0 + 25.0 * latent_interest + rng.gauss(0, 5)), 2
        )
        scroll_depth = round(
            max(0.0, min(1.0, 0.3 + 0.65 * latent_interest + rng.gauss(0, 0.08))), 3
        )
        downstream_conversion = (
            1 if rng.random() < sigmoid(4.0 * latent_interest - 1.5) else 0
        )
        add_to_cart = (
            1 if rng.random() < sigmoid(3.0 * latent_interest - 1.0) else 0
        )
        subsequent_purchase_amount = round(
            max(0.0, downstream_conversion * (10.0 + 80.0 * latent_interest + rng.gauss(0, 12))),
            2,
        )
    else:
        click_dwell_time = 0.0
        scroll_depth = 0.0
        downstream_conversion = 0
        add_to_cart = 0
        subsequent_purchase_amount = 0.0

    return {
        "impression_id": rng.randrange(10**8, 10**9),
        "label": int(label),
        "features": {
            "user_age_bucket": float(user_age_bucket),
            "user_lifetime_days": float(user_lifetime_days),
            "user_prior_session_count": float(user_prior_session_count),
            "item_category_match_score": float(item_category_match_score),
            "item_price_bucket": float(item_price_bucket),
            "item_freshness_days": float(item_freshness_days),
            "item_popularity_log": float(item_popularity_log),
            "session_dwell_time_so_far": float(session_dwell_time_so_far),
            "click_dwell_time": float(click_dwell_time),
            "scroll_depth": float(scroll_depth),
            "downstream_conversion": float(downstream_conversion),
            "add_to_cart": float(add_to_cart),
            "subsequent_purchase_amount": float(subsequent_purchase_amount),
        },
        "latent_interest_truth": round(latent_interest, 4),
        "label_meaning": "1 = clicked, 0 = no click",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    rows = [gen_impression(rng) for _ in range(args.n)]

    pos = sum(r["label"] for r in rows)
    neg = args.n - pos
    print(f"Generated {args.n} impressions: pos (clicked) = {pos}, neg (no click) = {neg}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    h = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"corpus sha256: {h}")
    print(f"wrote -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
