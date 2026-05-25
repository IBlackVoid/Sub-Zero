#!/usr/bin/env python3
"""LFAS synthetic bandit verification — F.4 theorem experiment.

Runs a K=4 synthetic bandit experiment that verifies LFAS-UCB achieves
sublinear regret against uniform-random, ε-greedy, and oracle baselines.

Usage:
    python scripts/bench/lfas_synthetic_bandit.py [--T 5000] [--seed 42]

Output:
    - scripts/bench/lfas_results.json  — per-step arm choices, regret
    - scripts/bench/lfas_regret.png    — cumulative regret plot (if matplotlib)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Arm configuration ───────────────────────────────────────────────

@dataclass
class Arm:
    """A synthetic arm with true F.3 ΔI̅ distribution."""
    name: str
    true_mean: float  # E[ΔI̅]
    true_sigma: float  # std dev of ΔI̅

    def sample(self, rng: random.Random) -> float:
        """Draw a noisy F.3 observation, clipped to [0, B]."""
        return max(0.0, min(B, rng.gauss(self.true_mean, self.true_sigma)))


B = math.log(2)  # reward bound (binary gate)

ARMS = [
    Arm("chunk=180s/fast/batch=32/w=4",   true_mean=0.35, true_sigma=0.12),
    Arm("chunk=240s/balanced/batch=24/w=3", true_mean=0.18, true_sigma=0.08),
    Arm("chunk=300s/balanced/batch=16/w=2", true_mean=0.15, true_sigma=0.06),  # best
    Arm("chunk=360s/strict/batch=16/w=1",  true_mean=0.22, true_sigma=0.09),
]

K = len(ARMS)
BEST_ARM = min(range(K), key=lambda a: ARMS[a].true_mean)
R_STAR = ARMS[BEST_ARM].true_mean


# ── C-BHC transfer function ────────────────────────────────────────

def phi(x: float) -> float:
    """C-BHC: φ(x) = √(1 − e^{−x})."""
    if x <= 0.0:
        return 0.0
    return math.sqrt(1.0 - math.exp(-x))


# ── LFAS-UCB ────────────────────────────────────────────────────────

@dataclass
class ArmState:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def variance(self) -> float:
        return self.m2 / self.n if self.n >= 2 else 1.0

    def update(self, value: float) -> None:
        self.n += 1
        d1 = value - self.mean
        self.mean += d1 / self.n
        d2 = value - self.mean
        self.m2 += d1 * d2


class LfasUCB:
    """LFAS-UCB with Bernstein confidence intervals."""

    def __init__(self, k: int, b: float, delta: float = 0.05):
        self.k = k
        self.b = b
        self.delta = delta
        self.arms = [ArmState() for _ in range(k)]
        self.t = 0

    def pick_arm(self) -> int:
        if self.t < self.k:
            return self.t

        best_arm = 0
        best_lcb = float("inf")

        for a in range(self.k):
            state = self.arms[a]
            if state.n == 0:
                return a
            var = state.variance()
            n = float(state.n)
            log_term = math.log(3.0 * (self.t ** 2) / self.delta)
            exploration = math.sqrt(var) * math.sqrt(2.0 * log_term / n) + \
                          3.0 * self.b * log_term / n
            lcb = state.mean - exploration
            if lcb < best_lcb:
                best_lcb = lcb
                best_arm = a

        return best_arm

    def record(self, arm: int, reward: float) -> None:
        self.t += 1
        r = max(0.0, min(self.b, reward))
        self.arms[arm].update(r)


# ── Baselines ───────────────────────────────────────────────────────

class UniformRandom:
    def __init__(self, k: int, rng: random.Random):
        self.k = k
        self.rng = rng

    def pick_arm(self) -> int:
        return self.rng.randint(0, self.k - 1)

    def record(self, arm: int, reward: float) -> None:
        pass


class EpsilonGreedy:
    def __init__(self, k: int, b: float, eps: float, rng: random.Random):
        self.k = k
        self.b = b
        self.eps = eps
        self.rng = rng
        self.arms = [ArmState() for _ in range(k)]
        self.t = 0

    def pick_arm(self) -> int:
        if self.t < self.k:
            return self.t
        if self.rng.random() < self.eps:
            return self.rng.randint(0, self.k - 1)
        return min(range(self.k), key=lambda a: self.arms[a].mean if self.arms[a].n > 0 else float("inf"))

    def record(self, arm: int, reward: float) -> None:
        self.t += 1
        self.arms[arm].update(max(0.0, min(self.b, reward)))


# ── Simulation ──────────────────────────────────────────────────────

@dataclass
class RunResult:
    name: str
    arms_played: list[int] = field(default_factory=list)
    cumulative_f3_regret: list[float] = field(default_factory=list)
    cumulative_cov_regret: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_f3_regret": self.cumulative_f3_regret[-1] if self.cumulative_f3_regret else 0.0,
            "total_cov_regret": self.cumulative_cov_regret[-1] if self.cumulative_cov_regret else 0.0,
            "arm_distribution": {
                str(a): self.arms_played.count(a) for a in range(K)
            },
        }


def simulate(policy, name: str, T: int, rng: random.Random) -> RunResult:
    result = RunResult(name=name)
    cum_f3 = 0.0
    cum_cov = 0.0

    for t in range(T):
        arm = policy.pick_arm()
        reward = ARMS[arm].sample(rng)
        policy.record(arm, reward)

        # F.3 regret: r_t - r*
        cum_f3 += ARMS[arm].true_mean - R_STAR
        # Coverage regret: phi(r*(a_t)) - phi(r*)
        cum_cov += phi(ARMS[arm].true_mean) - phi(R_STAR)

        result.arms_played.append(arm)
        result.cumulative_f3_regret.append(cum_f3)
        result.cumulative_cov_regret.append(cum_cov)

    return result


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LFAS synthetic bandit verification")
    parser.add_argument("--T", type=int, default=5000, help="Horizon (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    T = args.T
    seed = args.seed

    print(f"LFAS synthetic bandit: K={K}, T={T}, seed={seed}")
    print(f"Arms: {[(a.name, a.true_mean, a.true_sigma) for a in ARMS]}")
    print(f"Best arm: {BEST_ARM} (r* = {R_STAR:.3f})")
    print()

    results: list[RunResult] = []

    # Oracle (always best arm)
    rng = random.Random(seed)
    oracle_policy = type("Oracle", (), {
        "pick_arm": lambda self: BEST_ARM,
        "record": lambda self, arm, reward: None,
    })()
    results.append(simulate(oracle_policy, "Oracle", T, rng))

    # Uniform random
    rng = random.Random(seed)
    results.append(simulate(UniformRandom(K, rng), "Uniform", T, rng))

    # Epsilon-greedy (eps=0.1)
    rng = random.Random(seed)
    results.append(simulate(EpsilonGreedy(K, B, 0.1, rng), "EpsGreedy(0.1)", T, rng))

    # LFAS-UCB
    rng = random.Random(seed)
    lfas = LfasUCB(K, B, delta=1.0 / T)
    results.append(simulate(lfas, "LFAS-UCB", T, rng))

    # Print summary
    print(f"{'Policy':<20} {'F3 Regret':>12} {'Cov Regret':>12} {'Best Arm %':>12}")
    print("-" * 60)
    for r in results:
        best_pct = 100.0 * r.arms_played.count(BEST_ARM) / T
        print(f"{r.name:<20} {r.cumulative_f3_regret[-1]:>12.1f} "
              f"{r.cumulative_cov_regret[-1]:>12.1f} {best_pct:>11.1f}%")

    # Sublinearity check: LFAS regret at T/2 should be > 50% of regret at T
    # (linear would be exactly 50%; sublinear is > 50%)
    lfas_result = results[-1]
    half = T // 2
    if half > 0 and lfas_result.cumulative_f3_regret[-1] > 0:
        ratio = lfas_result.cumulative_f3_regret[half - 1] / lfas_result.cumulative_f3_regret[-1]
        sublinear = ratio > 0.55  # sublinear: more than 55% of regret in first half
        print(f"\nSublinearity check: regret(T/2)/regret(T) = {ratio:.3f} "
              f"({'PASS: sublinear' if sublinear else 'FAIL: possibly linear'})")

    # Improvement ratio
    uniform_regret = results[1].cumulative_cov_regret[-1]
    lfas_regret = lfas_result.cumulative_cov_regret[-1]
    if lfas_regret > 0:
        improvement = uniform_regret / lfas_regret
        print(f"Coverage regret improvement vs Uniform: {improvement:.1f}x")

    # Save results
    out_dir = Path(__file__).parent
    json_path = out_dir / "lfas_results.json"
    summary = {
        "config": {"K": K, "T": T, "seed": seed, "B": B, "r_star": R_STAR},
        "results": [r.to_dict() for r in results],
    }
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {json_path}")

    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        steps = list(range(1, T + 1))

        for r in results:
            ax1.plot(steps, r.cumulative_f3_regret, label=r.name)
        ax1.set_xlabel("Step t")
        ax1.set_ylabel("Cumulative F.3 Regret")
        ax1.set_title("F.3 Regret (Theorem 1)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for r in results:
            ax2.plot(steps, r.cumulative_cov_regret, label=r.name)
        ax2.set_xlabel("Step t")
        ax2.set_ylabel("Cumulative Coverage Regret")
        ax2.set_title("Coverage Regret (Theorem 2)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Reference: sqrt(T) curve for sublinearity
        sqrt_t = [math.sqrt(t) * 0.5 for t in steps]
        ax1.plot(steps, sqrt_t, "--", color="gray", alpha=0.5, label="O(√T) ref")
        ax1.legend()

        fig.tight_layout()
        png_path = out_dir / "lfas_regret.png"
        fig.savefig(str(png_path), dpi=150)
        print(f"Plot saved to {png_path}")
        plt.close(fig)
    except ImportError:
        print("(matplotlib not available — skipping plot)")


if __name__ == "__main__":
    main()
