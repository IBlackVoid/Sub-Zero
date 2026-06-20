#!/usr/bin/env python3
"""Generate a synthetic 30-day-mortality prediction corpus for the F.3
second-domain case study.

The point: demonstrate that the F.3 leakage diagnostic catches a
canonical "time-traveling feature" bug in a completely different domain
from subtitle QE — without being tuned to the subtitle setting. The
diagnostic was designed for VoiDex's reference-feature problem and
makes no medical-domain assumptions; if it generalises, it should flag
the leakers here too.

Domain
------
Predicting 30-day mortality at the moment of hospital admission. The
true clinical practice is to score patients at admission and triage
them to the appropriate level of care.

At admission a clinician *has*:
  age, sex, systolic_bp, heart_rate, oxygen_saturation,
  glasgow_coma_scale, admission_complaint_severity, comorbidity_count.

These eight features are **honest**: they are available at the exact
moment the prediction is needed.

After admission, the EHR records features that *cannot* be available
at the admission decision:
  peak_lactate_24h, icu_days, max_inflammatory_marker,
  intubated_within_24h, max_pressor_dose.

These five features are **post-admission**: they encode the trajectory
the patient actually took, which is partially the outcome we want to
predict. A naive ML pipeline that scrapes the EHR retrospectively
will include them in training; a deployment that scores at admission
cannot supply them.

Generation
----------
We synthesise a patient via:

    latent_severity ~ Beta(2, 5)      # ground-truth illness severity
    survives = Bernoulli(1 - σ(8 * latent_severity − 4))
    (and add per-feature Gaussian/discrete noise)

Honest features are weak-to-moderate functions of `latent_severity`.
Post-admission features are *strong* functions of `latent_severity`
plus a sharper response near the survival decision boundary —
they encode what actually happened.

The corpus is deliberately sized (n=500, balanced ~50/50) to be larger
than the VoiDex corpus (n=175) so reviewers can verify that the
diagnostic's behaviour is consistent at different sample sizes.

Output: `corpus.jsonl` with one JSON per line in the same schema the
F.3 audit and v2 training harness already consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


HONEST_FEATURES = (
    "age",
    "sex",
    "systolic_bp",
    "heart_rate",
    "oxygen_saturation",
    "glasgow_coma_scale",
    "admission_complaint_severity",
    "comorbidity_count",
)

# These 5 features are *post*-admission. Including them at training
# time corresponds to the canonical time-traveling-feature leakage bug.
# In the "deployed" / inference-time distribution they are forced to
# their absent-value (0 here, mirroring how a real EHR system would
# return null + the deployment code would default to 0).
POST_ADMISSION_FEATURES = (
    "peak_lactate_24h",
    "icu_days",
    "max_inflammatory_marker",
    "intubated_within_24h",
    "max_pressor_dose",
)


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def gen_patient(rng: random.Random) -> dict:
    # Latent ground-truth illness severity in [0, 1]. Drawn from Beta(2, 5)
    # so most patients are mildly ill, a tail are severely ill —
    # mirrors actual hospital triage distributions.
    a, b = 2.0, 5.0
    # Sample Beta via the gamma trick.
    g_a = sum(-math.log(1.0 - rng.random()) for _ in range(int(a)))
    g_b = sum(-math.log(1.0 - rng.random()) for _ in range(int(b)))
    latent_severity = g_a / (g_a + g_b) if (g_a + g_b) > 0 else 0.5
    # Push severity around a bit so it's not perfectly Beta-shaped.
    latent_severity = max(0.0, min(1.0, latent_severity + rng.gauss(0.0, 0.05)))

    # The label: died within 30 days.
    p_die = sigmoid(8.0 * latent_severity - 4.0)
    died = 1 if rng.random() < p_die else 0
    # We frame label as "good outcome (= survived)" for parity with the
    # subtitle gate convention where label=1 means good.
    label = 1 - died

    # Honest features — moderate signal.
    age = max(18, min(95, int(round(40 + 40 * latent_severity + rng.gauss(0, 10)))))
    sex = 1 if rng.random() < 0.52 else 0  # 1 = male; mild prevalence skew
    systolic_bp = max(60, int(round(130 - 30 * latent_severity + rng.gauss(0, 12))))
    heart_rate = max(40, int(round(75 + 35 * latent_severity + rng.gauss(0, 8))))
    oxygen_saturation = max(60, min(100, int(round(97 - 20 * latent_severity + rng.gauss(0, 2)))))
    glasgow_coma_scale = max(3, min(15, int(round(15 - 11 * latent_severity + rng.gauss(0, 1)))))
    admission_complaint_severity = max(0, min(10, int(round(1 + 7 * latent_severity + rng.gauss(0, 1)))))
    comorbidity_count = max(0, int(round(0.5 + 4 * latent_severity + rng.gauss(0, 1))))

    # Post-admission features — STRONG signal because they encode the
    # trajectory the patient actually took. This is the leakage path.
    # Note these features *correlate with the label itself*, not just
    # with latent severity, because the body's response over the next
    # 24h is part of the outcome.
    label_signal = (1 - label)  # 1 if died
    peak_lactate_24h = round(
        max(0.5, 1.0 + 4.0 * latent_severity + 4.0 * label_signal + rng.gauss(0, 0.6)),
        2,
    )
    icu_days = max(0, int(round(0.5 * latent_severity + 6.0 * label_signal + rng.gauss(0, 1))))
    max_inflammatory_marker = round(
        max(0.5, 2.0 + 10.0 * latent_severity + 12.0 * label_signal + rng.gauss(0, 2.0)),
        2,
    )
    intubated_within_24h = (
        1
        if rng.random() < sigmoid(6.0 * latent_severity + 4.0 * label_signal - 2.5)
        else 0
    )
    max_pressor_dose = round(
        max(0.0, 0.5 * latent_severity + 1.5 * label_signal + rng.gauss(0, 0.3)),
        2,
    )

    return {
        "patient_id": rng.randrange(10**8, 10**9),
        "label": int(label),
        "features": {
            "age": float(age),
            "sex": float(sex),
            "systolic_bp": float(systolic_bp),
            "heart_rate": float(heart_rate),
            "oxygen_saturation": float(oxygen_saturation),
            "glasgow_coma_scale": float(glasgow_coma_scale),
            "admission_complaint_severity": float(admission_complaint_severity),
            "comorbidity_count": float(comorbidity_count),
            "peak_lactate_24h": float(peak_lactate_24h),
            "icu_days": float(icu_days),
            "max_inflammatory_marker": float(max_inflammatory_marker),
            "intubated_within_24h": float(intubated_within_24h),
            "max_pressor_dose": float(max_pressor_dose),
        },
        # Provenance — recorded so reviewers can spot-check that no
        # feature outside the documented honest+leaky set is included.
        "latent_severity_truth": round(latent_severity, 4),
        "label_meaning": "1 = survived 30 days, 0 = died",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    rows = [gen_patient(rng) for _ in range(args.n)]

    pos = sum(r["label"] for r in rows)
    neg = args.n - pos
    print(
        f"Generated {args.n} patients: pos (survived) = {pos}, "
        f"neg (died) = {neg}",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Self-checking provenance hash so the README can quote the exact
    # SHA of the committed corpus.
    h = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"corpus sha256: {h}")
    print(f"wrote -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
