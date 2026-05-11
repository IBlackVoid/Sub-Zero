#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

cp "$root_dir/clip_10s.ja.srt" "$tmp_dir/clip_10s.ja.srt"

cd "$root_dir"

# Phrase-table + offline avoids pulling in any Python MT deps on CI runners.
cargo run --release -- \
  "$tmp_dir/clip_10s.ja.srt" \
  --offline \
  --phrase-table \
  --no-doom-qlock \
  --no-transcribe \
  --trace-runtime

trace_path="$tmp_dir/clip_10s.ja.sub-zero.trace.json"
test -f "$trace_path"

python3 - <<'PY' "$trace_path"
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    trace = json.load(f)

assert trace.get("trace_kind") == "runtime-performance"
assert isinstance(trace.get("stages"), list)
assert trace.get("total_elapsed_secs") is not None
assert math.isfinite(float(trace["total_elapsed_secs"]))

stage_names = [stage.get("name") for stage in trace["stages"]]
required = {
    "resolve_subtitle_source",
    "sidecar_health_and_rescue",
    "source_quality_gate",
    "translate",
    "translated_quality_gate",
    "write_output_srt",
    "write_metadata_sidecar",
}
missing = sorted(required.difference(stage_names))
assert not missing, f"missing stages: {missing}"

total = float(trace["total_elapsed_secs"])
assert total < 10.0, f"perf regression? total_elapsed_secs={total}"
print("perf_trace_smoke: ok")
PY

