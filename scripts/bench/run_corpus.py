#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(args: list[str], cwd: Path) -> int:
    proc = subprocess.run(args, cwd=str(cwd))
    return int(proc.returncode)


def ensure_release_bin(repo_root: Path) -> Path:
    exe = repo_root / "target" / "release" / ("sub-zero.exe" if os.name == "nt" else "sub-zero")
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=str(repo_root),
        check=True,
    )
    if not exe.is_file():
        raise RuntimeError(f"release binary not found after build: {exe}")
    return exe


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases-dir",
        default="benchmarks/cases",
        help="directory containing case JSON files (default: benchmarks/cases)",
    )
    ap.add_argument(
        "--run-dir-base",
        default="benchmarks/runs",
        help="base directory for run folders (default: benchmarks/runs)",
    )
    ap.add_argument(
        "--reports-dir",
        default="benchmarks/reports",
        help="report output directory (default: benchmarks/reports)",
    )
    ap.add_argument("--sub-zero-bin", help="path to sub-zero executable (optional)")
    ap.add_argument(
        "--timeout-secs",
        type=float,
        default=0.0,
        help="per-case timeout passed through to run_srt_benchmark.py (default: 0)",
    )
    ap.add_argument(
        "--keep-going",
        action="store_true",
        help="continue running remaining cases even if some fail; exit non-zero if any failed",
    )
    ap.add_argument(
        "--train-gate",
        action="store_true",
        help="after running cases, rebuild dataset from reports and retrain learned gate",
    )
    ap.add_argument(
        "--gate-label",
        default="ranking_top1",
        choices=["pass", "ranking_top1", "ranking_topk"],
        help="labeling mode for learned gate dataset (default: ranking_top1)",
    )
    ap.add_argument(
        "--gate-top-k",
        type=int,
        default=1,
        help="k for ranking_topk (default: 1)",
    )
    ap.add_argument(
        "--gate-out",
        default="models/learned_gate.json",
        help="output learned gate model path (default: models/learned_gate.json)",
    )
    ap.add_argument(
        "--dataset-out",
        default="datasets/quality_gate.jsonl",
        help="output dataset path (default: datasets/quality_gate.jsonl)",
    )
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]

    cases_dir_raw = Path(args.cases_dir)
    cases_dir = cases_dir_raw if cases_dir_raw.is_absolute() else (repo_root / cases_dir_raw)
    if not cases_dir.is_dir():
        print(f"error: missing cases dir: {cases_dir}", file=sys.stderr)
        return 2

    cases = sorted(cases_dir.rglob("*.json"))
    if not cases:
        print(f"error: no cases found under {cases_dir}", file=sys.stderr)
        return 2

    bench_script = repo_root / "scripts" / "bench" / "run_srt_benchmark.py"
    sub_zero_bin = Path(args.sub_zero_bin) if args.sub_zero_bin else ensure_release_bin(repo_root)

    failures: list[Path] = []
    for case_path in cases:
        cmd = [
            sys.executable,
            str(bench_script),
            "--case",
            str(case_path),
            "--run-dir-base",
            str(args.run_dir_base),
            "--reports-dir",
            str(args.reports_dir),
            "--sub-zero-bin",
            str(sub_zero_bin),
        ]
        if args.timeout_secs and args.timeout_secs > 0:
            cmd.extend(["--timeout-secs", str(args.timeout_secs)])

        code = run(cmd, cwd=repo_root)
        if code != 0:
            failures.append(case_path)
            if not args.keep_going:
                return code

    if not args.train_gate:
        if failures:
            print(f"finished with {len(failures)} failures", file=sys.stderr)
            for p in failures[:25]:
                print(f"- {p}", file=sys.stderr)
            if len(failures) > 25:
                print(f"- ... ({len(failures) - 25} more)", file=sys.stderr)
            return 1
        return 0

    dataset_script = repo_root / "scripts" / "quality_gate" / "build_dataset.py"
    train_script = repo_root / "scripts" / "quality_gate" / "train_gate.py"
    dataset_cmd = [
        sys.executable,
        str(dataset_script),
        "--reports-dir",
        str(args.reports_dir),
        "--search-dir",
        str(args.run_dir_base),
        "--out",
        str(args.dataset_out),
        "--label",
        str(args.gate_label),
        "--top-k",
        str(int(args.gate_top_k)),
        "--require-positive",
    ]
    code = run(dataset_cmd, cwd=repo_root)
    if code != 0:
        return code

    train_cmd = [
        sys.executable,
        str(train_script),
        "--data",
        str(args.dataset_out),
        "--out",
        str(args.gate_out),
    ]
    code = run(train_cmd, cwd=repo_root)
    if code != 0:
        return code

    if failures:
        print(f"finished with {len(failures)} failures (gate retrained anyway)", file=sys.stderr)
        for p in failures[:25]:
            print(f"- {p}", file=sys.stderr)
        if len(failures) > 25:
            print(f"- ... ({len(failures) - 25} more)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
