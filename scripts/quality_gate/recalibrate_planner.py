#!/usr/bin/env python3
"""Roll up per-run DOOM-QLOCK history into the global planner cache.

Each VoiDex run writes its own `.voidex/history.json` next to its
output directory. Those records are never merged into the global
`~/.voidex/history.json` the planner reads from at startup, so the
planner stays cold even after a 70-stream corpus run.

This script scans a search root (default: `benchmarks/runs/`), unions
every per-run history record, deduplicates by `(device_fingerprint,
content_profile_hash, timestamp_epoch_secs)`, and writes the merged set
to the global path. The pre-merge global file is backed up next to it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


HISTORY_VERSION = 1
HISTORY_MAX_RECORDS = 400  # mirrors src/engine/doom_qlock/history.rs:9


@dataclass
class MergeStats:
    sources_scanned: int
    records_in_sources: int
    records_in_global_before: int
    records_added: int
    records_in_global_after: int
    records_dropped_by_cap: int


def default_global_history_path() -> Path:
    voidex_home = os.environ.get("VOIDEX_HOME")
    if voidex_home:
        return Path(voidex_home) / "history.json"
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME") or "."
    return Path(home) / ".voidex" / "history.json"


def load_history(path: Path) -> dict:
    if not path.is_file():
        return {"version": HISTORY_VERSION, "records": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: could not read {path}: {e}", file=sys.stderr)
        return {"version": HISTORY_VERSION, "records": []}
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        return {"version": HISTORY_VERSION, "records": []}
    return data


def record_key(rec: dict) -> tuple[str, str, int]:
    return (
        str(rec.get("device_fingerprint") or ""),
        str(rec.get("content_profile_hash") or ""),
        int(rec.get("timestamp_epoch_secs") or 0),
    )


def is_record_usable(rec: dict) -> bool:
    """Filter to records that will actually inform the planner.

    The planner's `best_plan_exact` path filters on `success=True` and
    requires a non-empty input_kind plus an elapsed_secs measurement.
    """
    if not isinstance(rec, dict):
        return False
    if not isinstance(rec.get("elapsed_secs"), (int, float)):
        return False
    if not isinstance(rec.get("input_kind"), str) or not rec["input_kind"]:
        return False
    return True


def merge(
    search_root: Path,
    global_path: Path,
    successful_only: bool,
    dry_run: bool,
) -> MergeStats:
    sources = sorted(search_root.rglob(".voidex/history.json"))
    sources_scanned = 0
    records_in_sources = 0
    new_records: list[dict] = []
    seen: set[tuple[str, str, int]] = set()

    global_before = load_history(global_path)
    for rec in global_before.get("records", []):
        seen.add(record_key(rec))

    for source in sources:
        if source.resolve() == global_path.resolve():
            continue
        sources_scanned += 1
        data = load_history(source)
        for rec in data.get("records", []):
            records_in_sources += 1
            if not is_record_usable(rec):
                continue
            if successful_only and not bool(rec.get("success")):
                continue
            key = record_key(rec)
            if key in seen:
                continue
            seen.add(key)
            new_records.append(rec)

    merged = list(global_before.get("records", []))
    merged.extend(new_records)
    # Keep newest first, then enforce the cap.
    merged.sort(key=lambda r: int(r.get("timestamp_epoch_secs") or 0), reverse=True)
    dropped = max(0, len(merged) - HISTORY_MAX_RECORDS)
    if dropped > 0:
        merged = merged[:HISTORY_MAX_RECORDS]

    if not dry_run:
        if global_path.is_file():
            backup = global_path.with_suffix(
                f".json.bak-{int(time.time())}"
            )
            shutil.copy2(global_path, backup)
        global_path.parent.mkdir(parents=True, exist_ok=True)
        out_payload = {"version": HISTORY_VERSION, "records": merged}
        global_path.write_text(
            json.dumps(out_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return MergeStats(
        sources_scanned=sources_scanned,
        records_in_sources=records_in_sources,
        records_in_global_before=len(global_before.get("records", [])),
        records_added=len(new_records),
        records_in_global_after=len(merged),
        records_dropped_by_cap=dropped,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root",
        default="benchmarks/runs",
        help="directory to scan for per-run .voidex/history.json files",
    )
    parser.add_argument(
        "--global-path",
        default=str(default_global_history_path()),
        help="path to the global planner history.json",
    )
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="only merge records with success=True (recommended for the planner)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing",
    )
    args = parser.parse_args(argv)

    search_root = Path(args.search_root)
    if not search_root.is_dir():
        print(f"error: search-root does not exist: {search_root}", file=sys.stderr)
        return 2
    global_path = Path(args.global_path)

    stats = merge(
        search_root=search_root,
        global_path=global_path,
        successful_only=args.successful_only,
        dry_run=args.dry_run,
    )

    print(f"sources scanned          : {stats.sources_scanned}")
    print(f"records in sources       : {stats.records_in_sources}")
    print(f"records in global before : {stats.records_in_global_before}")
    print(f"records added            : {stats.records_added}")
    print(f"records dropped by cap   : {stats.records_dropped_by_cap}")
    print(f"records in global after  : {stats.records_in_global_after}")
    print(f"global path              : {global_path}")
    if args.dry_run:
        print("(dry run — no file written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
