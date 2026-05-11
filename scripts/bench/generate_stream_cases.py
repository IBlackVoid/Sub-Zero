#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def safe_text(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha1_8(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:8]


def ffprobe_streams(video: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name:stream_tags=language,title",
        "-of",
        "json",
        str(video),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def norm_lang(lang: str) -> str:
    v = (lang or "").strip().lower()
    if v in ("ja", "jpn", "jp"):
        return "jpn"
    if v in ("en", "eng"):
        return "eng"
    return v


def pick_audio_lang(streams: list[dict[str, Any]]) -> str | None:
    audio_langs = []
    for s in streams:
        if s.get("codec_type") != "audio":
            continue
        tags = s.get("tags") or {}
        audio_langs.append(norm_lang(str(tags.get("language") or "")))
    if "jpn" in audio_langs:
        return "jpn"
    if "eng" in audio_langs:
        return "eng"
    return None


def pick_reference_subtitle_stream(
    streams: list[dict[str, Any]], *, lang: str = "eng"
) -> int | None:
    # Only text tracks; skip image-based PGS.
    want = norm_lang(lang)
    candidates = []
    for s in streams:
        if s.get("codec_type") != "subtitle":
            continue
        codec = str(s.get("codec_name") or "")
        if codec in ("hdmv_pgs_subtitle", "dvd_subtitle"):
            continue
        tags = s.get("tags") or {}
        slang = norm_lang(str(tags.get("language") or ""))
        if want and slang != want:
            continue
        title = str(tags.get("title") or "")
        idx = int(s.get("index"))
        # Rank: prefer subrip, then dialogue-ish, then ass.
        score = 0
        if codec == "subrip":
            score += 10
        if "dialogue" in title.lower():
            score += 5
        if codec in ("ass", "ssa"):
            score += 2
        candidates.append((score, idx))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--streams-dir",
        default="datasets/streams",
        help="root directory containing video files (default: datasets/streams)",
    )
    ap.add_argument(
        "--out-dir",
        default="benchmarks/cases/generated/streams",
        help="output directory for generated case JSON files (default: benchmarks/cases/generated/streams)",
    )
    ap.add_argument(
        "--max",
        type=int,
        default=0,
        help="maximum number of cases to generate (0 = no limit)",
    )
    ap.add_argument(
        "--source-lang",
        default="ja",
        help="source language code for Sub-Zero (default: ja)",
    )
    ap.add_argument(
        "--target-lang",
        default="en",
        help="target language code for Sub-Zero (default: en)",
    )
    ap.add_argument(
        "--reference-lang",
        default="eng",
        help="subtitle language code used as reference (default: eng)",
    )
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    streams_dir = (repo_root / args.streams_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()

    if not streams_dir.is_dir():
        print(f"error: missing streams dir: {streams_dir}", file=sys.stderr)
        return 2

    videos = sorted(streams_dir.rglob("*.mkv"))
    if not videos:
        print(f"error: no mkv files under {streams_dir}", file=sys.stderr)
        return 2

    written = 0
    skipped = 0
    for video in videos:
        if args.max and written >= int(args.max):
            break

        rel = video.relative_to(repo_root)
        try:
            probe = ffprobe_streams(video)
        except Exception as e:
            skipped += 1
            print(f"skip ffprobe: {safe_text(str(rel))}: {e}", file=sys.stderr)
            continue

        streams = probe.get("streams") or []
        if not isinstance(streams, list):
            skipped += 1
            continue

        ref_idx = pick_reference_subtitle_stream(streams, lang=str(args.reference_lang))
        if ref_idx is None:
            skipped += 1
            continue

        audio_lang = pick_audio_lang(streams)
        # Prefer GPU when available; `--gpu` falls back to CPU, so it's safe on non-CUDA hosts.
        #
        # On Windows/CUDA, spawning many parallel python-whisper GPU workers is a common crash vector
        # (driver/VRAM pressure). Start conservative for corpus runs.
        base_args = [
            "--offline",
            "--transcribe",
            "--gpu",
            "--workers",
            "1",
            "--mt-force-cpu",
            "--mt-no-quality-floor",
        ]
        if audio_lang:
            base_args.extend(["--audio-lang", audio_lang])

        # Keep run-dir filenames short to avoid Windows path-length pain.
        video_ext = video.suffix.lower()
        case_id = sha1_8(str(rel).replace("\\", "/"))
        case_name = f"stream_{case_id}"
        input_basename = f"input{video_ext}"

        payload = {
            "case_name": case_name,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "input": str(rel).replace("\\", "/"),
            "reference": {
                "from_video": str(rel).replace("\\", "/"),
                "stream_index": int(ref_idx),
            },
            "base_args": base_args,
            "variants": [
                {
                    "name": "fast",
                    "input_basename": input_basename,
                    "args": ["--profile", "fast"],
                },
                {
                    "name": "fast_speaker_diarize",
                    "input_basename": input_basename,
                    "args": [
                        "--profile",
                        "fast",
                        "--speaker-diarize",
                        "--speaker-max-speakers",
                        "6",
                    ],
                },
            ],
        }

        out_path = out_dir / f"{case_name}.json"
        dump_json(out_path, payload)
        written += 1

    print(f"generated {written} cases (skipped {skipped})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
