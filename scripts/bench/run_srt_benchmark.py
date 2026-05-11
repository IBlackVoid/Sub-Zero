#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


def safe_text(text: str) -> str:
    # Keep logs ASCII-safe on Windows consoles.
    return text.encode("unicode_escape").decode("ascii")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def today_str() -> str:
    return date.today().isoformat()


def slugify(name: str) -> str:
    out = []
    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "."):
            out.append("_")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s or "case"


def pick_run_dir(base_dir: Path, case_name: str) -> Path:
    base = base_dir / f"{today_str()}_{slugify(case_name)}"
    if not base.exists():
        return base
    for i in range(1, 10_000):
        cand = base_dir / f"{today_str()}_{slugify(case_name)}_{i:03d}"
        if not cand.exists():
            return cand
    raise RuntimeError("failed to pick unique run dir")


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


def run_cmd(
    args: list[str],
    cwd: Path,
    *,
    timeout_secs: float | None = None,
    env_extra: dict[str, str] | None = None,
) -> tuple[int, float, str]:
    started = time.time()
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_secs,
            env=env,
        )
        elapsed = time.time() - started
        return proc.returncode, elapsed, proc.stdout
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - started
        out = (e.stdout or "") + "\n[timeout]\n"
        return 124, elapsed, out


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return
    except Exception:
        shutil.copy2(src, dst)


def extract_reference_from_video(video: Path, stream_index: int, out_srt: Path) -> None:
    # Extract and convert to SRT (ASS->SRT often carries <font> tags; evaluator strips them).
    video_abs = video.resolve()
    out_abs = out_srt.resolve()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(video_abs),
        "-map",
        f"0:{int(stream_index)}",
        "-c:s",
        "srt",
        str(out_abs),
    ]
    code, _elapsed, out = run_cmd(cmd, cwd=Path("."), timeout_secs=120.0)
    if code != 0 or not out_abs.is_file():
        raise RuntimeError(
            f"failed to extract reference stream {stream_index} from {video}: {out.strip()}"
        )


def eval_quality(
    repo_root: Path,
    reference: Path,
    hypothesis: Path,
    *,
    min_line_similarity: float,
    max_japanese_ratio: float,
    min_weighted_token_f1: float,
    min_time_coverage: float,
) -> dict[str, Any]:
    script = repo_root / "scripts" / "evaluate_sub_quality.py"
    code, _, out = run_cmd(
        [
            sys.executable,
            str(script),
            "--reference",
            str(reference),
            "--hypothesis",
            str(hypothesis),
            "--min-line-similarity",
            str(min_line_similarity),
            "--max-japanese-ratio",
            str(max_japanese_ratio),
            "--min-weighted-token-f1",
            str(min_weighted_token_f1),
            "--min-time-coverage",
            str(min_time_coverage),
        ],
        cwd=repo_root,
        timeout_secs=None,
    )
    if code != 0:
        raise RuntimeError(f"quality eval failed (code={code}): {out.strip()}")
    return json.loads(out)


def candidate_score(metrics: dict[str, Any]) -> float:
    # Deterministic, "reasonably aligned" ranking:
    # prioritize content overlap, then timing alignment, then coverage.
    wtf1 = float(metrics.get("weighted_token_f1") or 0.0)
    tiou = float(metrics.get("weighted_timing_iou") or 0.0)
    cov = float(metrics.get("time_coverage_ratio") or 0.0)
    return 0.55 * wtf1 + 0.25 * tiou + 0.20 * cov


def gap_summary_from_best(metrics: dict[str, Any]) -> dict[str, Any]:
    cue_match = bool(metrics.get("cue_count_match"))
    density = float(metrics.get("cue_density_ratio") or 0.0)
    tiou = float(metrics.get("weighted_timing_iou") or 0.0)
    wtf1 = float(metrics.get("weighted_token_f1") or 0.0)

    if not cue_match or abs(density - 1.0) > 0.12:
        largest = "cue segmentation and timing style mismatch vs reference"
        next_focus = [
            "optional style-matching segmentation mode",
            "time-warp post-pass to align cue boundaries to reference-like pacing",
            "cue-density regularizer (avoid over-fragmentation)",
        ]
    elif tiou < 0.55:
        largest = "timing mismatch vs reference"
        next_focus = [
            "time-warp post-pass to align cue boundaries to reference-like pacing",
            "VAD-boundary snapping on short cues",
            "cue overlap minimization pass",
        ]
    elif wtf1 < 0.30:
        largest = "lexical mismatch vs reference"
        next_focus = [
            "phrase-level style normalization for short interjections",
            "speaker/register-conditioned decoding for conversational cues",
            "short-utterance lexical normalization pass",
        ]
    else:
        largest = "residual stylistic differences vs reference"
        next_focus = [
            "learned gate trained on your reference corpus (more labels)",
            "speaker/relationship graph conditioning strengthened across scenes",
            "post-edit pass for flagged cues only",
        ]

    secondary = "lexical style divergence in short conversational cues"
    return {
        "largest_gap": largest,
        "secondary_gap": secondary,
        "next_focus": next_focus,
    }


@dataclass(frozen=True)
class Variant:
    name: str
    input_basename: str
    args: list[str]


def parse_case_variants(case: dict[str, Any], input_path: Path) -> list[Variant]:
    variants_raw = case.get("variants") or []
    if not isinstance(variants_raw, list) or not variants_raw:
        # Default: a single variant with the original basename and no extra args.
        return [
            Variant(
                name="default",
                input_basename=input_path.name,
                args=[],
            )
        ]

    out: list[Variant] = []
    for v in variants_raw:
        if not isinstance(v, dict):
            continue
        name = str(v.get("name") or "").strip()
        input_basename = str(v.get("input_basename") or "").strip()
        args = v.get("args") or []
        if not name or not input_basename or not isinstance(args, list):
            continue
        out.append(Variant(name=name, input_basename=input_basename, args=[str(a) for a in args]))
    if not out:
        raise RuntimeError("case variants are empty/invalid")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="benchmarks/cases/*.json")
    ap.add_argument("--input", help="source SRT input")
    ap.add_argument("--reference", help="reference SRT path")
    ap.add_argument("--case-name", help="case name used for run/report naming")
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
    ap.add_argument(
        "--sub-zero-bin",
        help="path to sub-zero executable (default: build target/release)",
    )
    ap.add_argument(
        "--timeout-secs",
        type=float,
        default=0.0,
        help="per-variant timeout (0 disables; default: 0)",
    )

    # Evaluator thresholds (must match evaluate_sub_quality defaults unless you have a reason).
    ap.add_argument("--min-line-similarity", type=float, default=0.20)
    ap.add_argument("--max-japanese-ratio", type=float, default=0.20)
    ap.add_argument("--min-weighted-token-f1", type=float, default=0.35)
    ap.add_argument("--min-time-coverage", type=float, default=0.75)

    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    timeout_secs: float | None = None
    if args.timeout_secs and args.timeout_secs > 0:
        timeout_secs = float(args.timeout_secs)

    if args.case:
        case_path = Path(args.case)
        case = load_json(case_path)
        case_name = str(case.get("case_name") or case_path.stem)
        input_path = Path(str(case.get("input") or "")).expanduser()
        reference_raw = case.get("reference")
        base_args = case.get("base_args") or []
        if not isinstance(base_args, list):
            raise RuntimeError("case.base_args must be a list")
        base_args_list = [str(a) for a in base_args]
        source_lang = str(case.get("source_lang") or "ja")
        target_lang = str(case.get("target_lang") or "en")
        variants = parse_case_variants(case, input_path)
    else:
        if not args.input or not args.reference:
            print("error: either --case or (--input and --reference) must be provided", file=sys.stderr)
            return 2
        case_name = args.case_name or Path(args.input).stem
        input_path = Path(args.input).expanduser()
        reference_raw = args.reference
        base_args_list = []
        source_lang = "ja"
        target_lang = "en"
        variants = [
            Variant(name="default", input_basename=input_path.name, args=[]),
        ]

    if not input_path.is_file():
        print(f"error: missing input: {safe_text(str(input_path))}", file=sys.stderr)
        return 2

    run_dir = pick_run_dir(Path(args.run_dir_base), case_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    sub_zero = Path(args.sub_zero_bin) if args.sub_zero_bin else ensure_release_bin(repo_root)

    candidates: list[dict[str, Any]] = []
    candidate_names_in_order: list[str] = []

    report_reference: Any = reference_raw
    for variant in variants:
        requested = Path(variant.input_basename)
        variant_input = run_dir / f"{requested.stem}.{variant.name}{requested.suffix}"
        link_or_copy(input_path, variant_input)

        if isinstance(reference_raw, dict):
            from_video_raw = reference_raw.get("from_video")
            stream_index = reference_raw.get("stream_index")
            if not from_video_raw or not isinstance(stream_index, int):
                raise RuntimeError(
                    "case reference must be either a file path string or "
                    "a dict {from_video: <path>, stream_index: <int>}"
                )
            from_video = Path(str(from_video_raw)).expanduser()
            # If the reference points at the same file as input, use the run-dir link.
            video_for_ref = variant_input if from_video.resolve() == input_path.resolve() else from_video
            reference_path = run_dir / "reference.en.srt"
            extract_reference_from_video(video_for_ref, int(stream_index), reference_path)
            report_reference = {
                "from_video": str(from_video).replace("\\", "/"),
                "stream_index": int(stream_index),
            }
        else:
            reference_path = Path(str(reference_raw or "")).expanduser()
            if not reference_path.is_file():
                print(f"error: missing reference: {safe_text(str(reference_path))}", file=sys.stderr)
                return 2
            report_reference = str(reference_path).replace("\\", "/")

        cmd = [
            str(sub_zero),
            str(variant_input),
            "--source-lang",
            source_lang,
            "--lang",
            target_lang,
            *base_args_list,
            *variant.args,
        ]
        candidate_names_in_order.append(variant.name)

        sub_zero_home = run_dir / ".sub-zero"
        sub_zero_home.mkdir(parents=True, exist_ok=True)
        code, elapsed, out = run_cmd(
            cmd,
            cwd=repo_root,
            timeout_secs=timeout_secs,
            env_extra={
                "SUB_ZERO_HOME": str(sub_zero_home),
            },
        )
        if code != 0:
            print(
                f"[{variant.name}] sub-zero failed code={code}: {safe_text(out.strip())}",
                file=sys.stderr,
            )
            candidates.append(
                {
                    "name": str(run_dir / f"{variant_input.stem}.{target_lang}.srt").replace(
                        "\\", "/"
                    ),
                    "variant": variant.name,
                    "run": {
                        "command": " ".join(cmd),
                        "wall_time_seconds": round(elapsed, 3),
                        "status": "failed",
                        "output": out[-4000:],
                    },
                    "metrics": {"pass": False},
                }
            )
            continue

        hypothesis = run_dir / f"{variant_input.stem}.{target_lang}.srt"
        if not hypothesis.is_file():
            print(
                f"[{variant.name}] missing hypothesis output: {safe_text(str(hypothesis))}",
                file=sys.stderr,
            )
            candidates.append(
                {
                    "name": str(hypothesis).replace("\\", "/"),
                    "variant": variant.name,
                    "run": {
                        "command": " ".join(cmd),
                        "wall_time_seconds": round(elapsed, 3),
                        "status": "failed",
                        "output": out[-4000:],
                    },
                    "metrics": {"pass": False},
                }
            )
            continue

        metrics = eval_quality(
            repo_root,
            reference_path,
            hypothesis,
            min_line_similarity=float(args.min_line_similarity),
            max_japanese_ratio=float(args.max_japanese_ratio),
            min_weighted_token_f1=float(args.min_weighted_token_f1),
            min_time_coverage=float(args.min_time_coverage),
        )

        candidates.append(
            {
                "name": str(hypothesis).replace("\\", "/"),
                "variant": variant.name,
                "run": {
                    "command": " ".join(cmd),
                    "wall_time_seconds": round(elapsed, 3),
                    "status": "success",
                },
                "metrics": metrics,
            }
        )

    # Rank only successful candidates with usable metrics.
    ranked = []
    for cand in candidates:
        metrics = cand.get("metrics") or {}
        if isinstance(metrics, dict) and "weighted_token_f1" in metrics:
            ranked.append((cand["name"], metrics, candidate_score(metrics)))

    ranked.sort(key=lambda t: (-bool(t[1].get("pass")), -t[2], t[0]))
    ranking = [name for (name, _m, _s) in ranked]

    best_metrics = ranked[0][1] if ranked else {}
    report = {
        "generated_at": today_str(),
        "input": case_name,
        "reference": report_reference,
        "source": {"path": str(input_path).replace("\\", "/")},
        "run_dir": str(run_dir).replace("\\", "/"),
        # Keep full candidate records (command/status + truncated output tail on failure)
        # so corpus reports are self-contained for debugging and gate dataset generation.
        "candidates": candidates,
        "ranking": ranking,
        "gap_summary": gap_summary_from_best(best_metrics) if best_metrics else {},
        "notes": {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "base_args": base_args_list,
            "variants": [
                {"name": v.name, "input_basename": v.input_basename, "args": v.args}
                for v in variants
            ],
        },
    }

    # Use the run dir name for uniqueness; multiple runs per day are expected.
    report_path = Path(args.reports_dir) / f"{run_dir.name}.json"
    dump_json(report_path, report)
    print(f"wrote report: {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
