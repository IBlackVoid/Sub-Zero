#!/usr/bin/env python3
"""Preview a single frame of a Sub-Zero TUI animation as ANSI.

Usage:
  python scripts/tui/preview.py assets/ascii/idle.jsonl
  python scripts/tui/preview.py assets/ascii/running.jsonl --frame 12
  python scripts/tui/preview.py assets/ascii/  # iterates every state
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows consoles default to cp1252 which cannot encode braille glyphs.
# Force stdout/stderr to UTF-8 so the preview survives on Git Bash, conhost
# without UTF-8 mode, and PowerShell prior to enabling beta UTF-8 support.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def render_frame(frame: dict, *, no_color: bool = False) -> str:
    width = int(frame.get("width", 0))
    height = int(frame.get("height", 0))
    rows = frame.get("rows") or []
    out: list[str] = []
    for row in rows[:height]:
        line: list[str] = []
        for cell in row[:width]:
            if not isinstance(cell, list) or len(cell) < 4:
                line.append(" ")
                continue
            ch, r, g, b = cell[0], int(cell[1]), int(cell[2]), int(cell[3])
            if no_color:
                line.append(str(ch))
            else:
                line.append(f"\x1b[38;2;{r};{g};{b}m{ch}")
        out.append("".join(line) + ("" if no_color else "\x1b[0m"))
    return "\n".join(out)


def preview_file(path: Path, frame_idx: int, *, no_color: bool) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"error: {path}: {e}", file=sys.stderr)
        return 1
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        print(f"error: {path} has no frames", file=sys.stderr)
        return 1
    if frame_idx >= len(lines):
        frame_idx = len(lines) - 1
    try:
        frame = json.loads(lines[frame_idx])
    except json.JSONDecodeError as e:
        print(f"error: {path} frame {frame_idx}: {e}", file=sys.stderr)
        return 1

    header = f"== {path.name} :: frame {frame_idx + 1}/{len(lines)} ({frame.get('width')}x{frame.get('height')}) =="
    print(header)
    print(render_frame(frame, no_color=no_color))
    print()
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="JSONL file or directory containing *.jsonl")
    parser.add_argument("--frame", type=int, default=0, help="frame index (default: 0)")
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="skip ANSI color escapes (useful for environments that don't render them)",
    )
    args = parser.parse_args(argv)

    p = Path(args.path)
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
        if not files:
            print(f"error: no .jsonl in {p}", file=sys.stderr)
            return 1
        rc = 0
        for f in files:
            rc |= preview_file(f, args.frame, no_color=args.no_color)
        return rc
    return preview_file(p, args.frame, no_color=args.no_color)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
