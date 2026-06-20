#!/usr/bin/env python3
"""Verify the M/D/1 tandem latency bound (docs/F1_latency.md, Theorem 1).

Reads `*.voidex.trace.json` sidecars produced by `--trace-runtime`.
For each trace it estimates per-stage mean service time and global
throughput, computes the Pollaczek-Khinchine M/D/1 bound, and checks

    total_elapsed_secs <= N_chunks * W_total + (K - 1) * max_k S_k

A non-zero exit code is returned if any trace violates the bound.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class StageStat:
    name: str
    service_time_secs: float
    rho: float
    response_time_secs: float


@dataclass(frozen=True)
class TraceVerdict:
    path: Path
    n_chunks: int
    total_observed_secs: float
    arrival_rate: float
    stages: tuple[StageStat, ...]
    predicted_upper_bound_secs: float
    bound_holds: bool


def load_trace(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_n_chunks(trace: dict) -> int | None:
    for key in ("chunk_count", "num_chunks", "chunks"):
        v = trace.get(key)
        if isinstance(v, int) and v > 0:
            return v
    plan = trace.get("plan_used") or trace.get("plan") or {}
    if isinstance(plan, dict):
        v = plan.get("chunks") or plan.get("chunk_count")
        if isinstance(v, int) and v > 0:
            return v
    stages = trace.get("stages") or []
    counts = [
        s.get("chunks")
        for s in stages
        if isinstance(s, dict) and isinstance(s.get("chunks"), int)
    ]
    if counts:
        return max(counts)
    return None


# Stages known to run with the pipeline's plan-level worker count. Every
# other stage is treated as serial (workers=1). This mapping is the
# verifier's load-bearing assumption — if a new parallel stage is added
# to the engine, add its trace name here. See `src/engine/pipeline.rs`
# `record_runtime_stage(... "<stage_name>", ...)` for the truth set.
_PARALLEL_STAGE_NAMES = frozenset(
    {
        # New async streaming pipeline.
        "stream_transcribe_translate",
        # Original sync parallel pipeline.
        "parallel_pipeline",
        # Translation pass uses the MT daemon's batched workers.
        "translate",
        # Legacy alias retained for old traces.
        "transcribe",
    }
)


def per_stage_workers(trace: dict, name: str) -> int:
    # 1) Prefer an explicit per-stage worker count embedded in the trace
    #    when the engine has emitted one (forward-compat path for newer
    #    trace formats that promote workers out of `details`).
    for stage in trace.get("stages") or []:
        if not isinstance(stage, dict) or stage.get("name") != name:
            continue
        if isinstance(stage.get("workers"), int) and stage["workers"] > 0:
            return int(stage["workers"])
        details = stage.get("details")
        if (
            isinstance(details, dict)
            and isinstance(details.get("workers"), int)
            and details["workers"] > 0
        ):
            return int(details["workers"])
    # 2) Otherwise, only the known parallel stages take the plan-level
    #    worker count; serial stages keep workers=1. The previous version
    #    of this script only checked `name == "transcribe"`, which never
    #    matched any real stage name and so silently treated every stage
    #    as serial — underestimating service time on parallel stages.
    if name not in _PARALLEL_STAGE_NAMES:
        return 1
    plan = trace.get("plan_used") or trace.get("plan") or {}
    if isinstance(plan, dict) and isinstance(plan.get("workers"), int):
        return max(int(plan["workers"]), 1)
    return 1


def verify_one(path: Path) -> TraceVerdict | None:
    trace = load_trace(path)
    total = trace.get("total_elapsed_secs")
    stages_raw = trace.get("stages") or []
    if not isinstance(total, (int, float)) or total <= 0 or not stages_raw:
        return None

    n_chunks = extract_n_chunks(trace) or 1
    arrival_rate = float(n_chunks) / float(total)

    stages: list[StageStat] = []
    for stage in stages_raw:
        if not isinstance(stage, dict):
            continue
        name = stage.get("name")
        elapsed = stage.get("elapsed_secs")
        if not isinstance(name, str) or not isinstance(elapsed, (int, float)):
            continue
        workers = per_stage_workers(trace, name)
        # Mean service time per chunk, accounting for parallel workers.
        service_time = max(float(elapsed) * workers / float(n_chunks), 0.0)
        rho = arrival_rate * service_time
        if rho >= 1.0:
            # Saturated stage: bound diverges; clip and flag downstream.
            response_time = float("inf")
        elif service_time == 0.0:
            response_time = 0.0
        else:
            # M/D/1 mean response: S * (2 - rho) / (2 * (1 - rho))
            response_time = service_time * (2.0 - rho) / (2.0 * (1.0 - rho))
        stages.append(
            StageStat(
                name=name,
                service_time_secs=service_time,
                rho=rho,
                response_time_secs=response_time,
            )
        )

    if not stages:
        return None

    w_total = sum(s.response_time_secs for s in stages)
    max_service = max(s.service_time_secs for s in stages)
    k_minus_one = max(len(stages) - 1, 0)
    predicted_upper = float(n_chunks) * w_total + k_minus_one * max_service

    bound_holds = float(total) <= predicted_upper if predicted_upper != float("inf") else False

    return TraceVerdict(
        path=path,
        n_chunks=n_chunks,
        total_observed_secs=float(total),
        arrival_rate=arrival_rate,
        stages=tuple(stages),
        predicted_upper_bound_secs=predicted_upper,
        bound_holds=bound_holds,
    )


def fmt_secs(value: float) -> str:
    if value == float("inf"):
        return "    inf"
    return f"{value:8.3f}s"


def print_verdict(v: TraceVerdict) -> None:
    print(f"\ntrace: {v.path}")
    print(
        f"  n_chunks={v.n_chunks}  observed_total={fmt_secs(v.total_observed_secs)}"
        f"  lambda={v.arrival_rate:.4f} chunks/s"
    )
    print(f"  {'stage':28} {'S':>10} {'rho':>8} {'W_k':>10}")
    print(f"  {'-' * 28} {'-' * 10} {'-' * 8} {'-' * 10}")
    for s in v.stages:
        rho_disp = f"{s.rho:.3f}" if s.rho < 1.0 else ">=1 SAT"
        print(
            f"  {s.name:28} {fmt_secs(s.service_time_secs):>10} {rho_disp:>8}"
            f" {fmt_secs(s.response_time_secs):>10}"
        )
    bound_label = "PASS" if v.bound_holds else "FAIL"
    print(
        f"  predicted upper bound = {fmt_secs(v.predicted_upper_bound_secs)}"
        f"   bound holds: {bound_label}"
    )


def aggregate(verdicts: Iterable[TraceVerdict]) -> None:
    verdicts = list(verdicts)
    if not verdicts:
        return
    holds = sum(1 for v in verdicts if v.bound_holds)
    slacks = [
        v.predicted_upper_bound_secs - v.total_observed_secs
        for v in verdicts
        if v.predicted_upper_bound_secs != float("inf")
    ]
    print("\n=== summary ===")
    print(f"traces verified: {len(verdicts)}")
    print(f"bound holds:     {holds}/{len(verdicts)}")
    if slacks:
        print(
            f"slack (bound - observed): mean={statistics.fmean(slacks):.3f}s "
            f"p50={statistics.median(slacks):.3f}s "
            f"min={min(slacks):.3f}s"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="*.voidex.trace.json files")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any trace violates the bound",
    )
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.traces]
    verdicts: list[TraceVerdict] = []
    for path in paths:
        if not path.is_file():
            print(f"error: not a file: {path}", file=sys.stderr)
            return 2
        v = verify_one(path)
        if v is None:
            print(f"warning: skipping unrecognised trace: {path}", file=sys.stderr)
            continue
        print_verdict(v)
        verdicts.append(v)

    aggregate(verdicts)

    if args.strict and any(not v.bound_holds for v in verdicts):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
