#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[integration] checking ffmpeg/ffprobe availability"
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null

echo "[integration] checking neural MT python dependencies"
python3 - <<'PY'
import importlib.util
for name in ("ctranslate2", "sentencepiece"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"missing python dependency: {name}")
print("python deps ok")
PY

echo "[integration] ffprobe smoke"
ffprobe -hide_banner -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 clip_10s.wav >/tmp/sub_zero_clip_duration.txt
python3 - <<'PY'
from pathlib import Path
duration = float(Path("/tmp/sub_zero_clip_duration.txt").read_text().strip())
if duration <= 0.5:
    raise SystemExit(f"invalid clip duration: {duration}")
print(f"clip duration ok: {duration:.3f}s")
PY

echo "[integration] rust integration tests (fast path)"
cargo test --test cli_integration help_flag_exits_success -- --nocapture
cargo test --test cli_integration phrase_table_cli_smoke -- --nocapture

echo "[integration] rust integration tests (ffmpeg + neural smoke)"
SUB_ZERO_RUN_FFMPEG_SMOKE=1 cargo test --test cli_integration ffmpeg_ffprobe_smoke -- --nocapture
SUB_ZERO_RUN_NEURAL_SMOKE=1 cargo test --test cli_integration neural_mt_subtitle_quality_smoke -- --nocapture

echo "[integration] generating neural translation for quality evaluation"
cargo run --quiet -- -i clip_10s.ja.srt --source-lang ja --lang en --offline

echo "[integration] subtitle quality evaluation"
python3 scripts/evaluate_sub_quality.py \
  --reference clip_10s.en.srt \
  --hypothesis clip_10s.ja.en.srt \
  --min-line-similarity 0.20 \
  --max-japanese-ratio 0.20 \
  --fail-on-low-quality

echo "[integration] smoke checks completed"
