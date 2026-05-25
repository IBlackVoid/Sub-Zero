# Sub-Zero

**offline subtitle engine. nothing leaves your machine. ever.**

[![CI](https://github.com/IBlackVoid/Sub-Zero/actions/workflows/ci.yml/badge.svg)](https://github.com/IBlackVoid/Sub-Zero/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

---

## what it does

takes a video or `.srt` file. transcribes it locally (whisper). translates it
locally (NLLB-200). runs a quality gate against the result. emits a sidecar
with every metric it computed so you can audit every claim.

no cloud. no API keys. no telemetry. no data leaves your machine.

**in one line:**

```bash
sub-zero -i movie.mkv --source-lang ja --lang en --gpu --parallel
```

output: `movie.en.srt` + `movie.sub-zero.json` (quality audit sidecar).

---

## why this exists

every subtitle tool either:
- sends your content to a cloud API (privacy violation)
- gives you zero quality metrics (hope-based engineering)
- can't scale past a single file without manual babysitting

Sub-Zero does none of that. it runs entirely offline, tells you exactly how
confident it is, and self-heals when quality drops below the floor.

---

## features

| feature | what it does |
|---------|-------------|
| **DOOM-QLOCK** | adaptive runtime scheduler. probes your hardware, picks the optimal plan, learns from history |
| **parallel chunked transcription** | splits audio at silence boundaries, transcribes in parallel, stitches with overlap dedup |
| **LFAS (Label-Free Adaptive Scheduling)** | bandit algorithm that minimizes quality regret without requiring ground-truth labels |
| **C-BHC coverage bound** | provable coverage guarantee via Bretagnolle-Huber inequality |
| **Rényi-Hellinger sharpening** | tighter coverage bound computed directly from streaming histograms |
| **learned quality gate** | logistic + isotonic + conformal prediction — PASS/REJECT/ABSTAIN with calibrated confidence |
| **speaker-aware translation** | per-character voice priors, discourse consistency, register tagging |
| **character glossary** | persistent name canonicalization across episodes |
| **scene rescue** | auto-retries low-quality scenes with halved batch size |
| **live TUI dashboard** | terminal UI with real-time progress, three visual modes, waveform display |

---

## quick start

### install

```bash
git clone https://github.com/IBlackVoid/Sub-Zero
cd Sub-Zero
cargo build --release
```

binaries land in `target/release/`:
- `sub-zero` — the engine CLI
- `sub-zero-tui` — the live dashboard

### prerequisites

- **Rust** 1.75+ (`rustup default stable`)
- **ffmpeg** on PATH (audio extraction + media probing)
- **Python 3.10+** with `openai-whisper` installed (transcription backend)
- (optional) **whisper.cpp** binary for 5-10x faster transcription
- (optional) **NLLB-200** in CTranslate2 format for neural translation

### first run

```bash
# transcribe + translate a video (GPU accelerated)
sub-zero -i video.mp4 --source-lang ja --lang en --gpu --parallel

# just transcribe english audio
sub-zero -i podcast.mp4 --source-lang en --lang en --gpu

# translate an existing SRT
sub-zero -i existing.ja.srt --lang en

# fire up the dashboard
sub-zero-tui
```

### models (one-time setup)

```bash
# whisper model (auto-downloaded by openai-whisper on first run)
pip install openai-whisper

# NLLB-200 for neural translation (optional)
./scripts/release/bootstrap_models.sh
```

without NLLB, falls back to a phrase-table backend. enough to smoke-test.

---

## how it works

```
video.mkv
    │
    ├─── ffmpeg extract audio ──► mono 16kHz WAV
    │
    ├─── SBOD chunking ──► split at silence gaps (never mid-sentence)
    │
    ├─── parallel whisper (1 GPU worker, full VRAM) ──► per-chunk SRTs
    │
    ├─── stitcher ──► merge + Levenshtein dedup at boundaries
    │
    ├─── coverage gate ──► verify all chunks produced output
    │
    ├─── neural MT (NLLB-200) ──► batched translation with quality floor
    │
    ├─── post-processing ──► discourse consistency, scene rescue, compaction
    │
    ├─── quality gate ──► structural + semantic + learned gate
    │
    └─── output: video.en.srt + video.sub-zero.json (audit sidecar)
```

---

## the dashboard

```bash
cargo run -p sub-zero-tui --release
```

a live terminal dashboard. shows the pipeline as it runs, cues as they
translate, per-character voice priors the engine learns.

three running-screen modes (cycle with `g`):
- **original** — the pre-recorded braille animation
- **emerge** — reveals cell-by-cell as chunks complete
- **generative** — flow-field particle system, fresh artwork per run

keybinds: `Enter` pick file, `r` re-run, `p` cycle profile, `g` cycle
visual, `Tab` path completion, `:help` full reference.

---

## quality metrics

every claim is checkable. the fidelity bound is the empirical mutual
information between machine and reference, estimated with a Kraskov
k-NN estimator:

```bash
python scripts/quality_gate/verify_fidelity_bound.py \
  --machine  out.en.srt \
  --baseline baseline.en.srt \
  --reference reference.en.srt \
  --strict
```

real-corpus results (998 aligned cues, JP gameplay stream, human reference):

| metric | value |
|--------|-------|
| I(machine ; reference) | **0.6985 nats** |
| I(baseline ; reference) | 0.2282 nats |
| fidelity gap | **+0.4703 nats** (3x the mutual information) |
| name inconsistency | 0.00% |
| adjacent repeats | 0.37% |
| scene low-quality | 1.13% |

---

## CLI reference

```
sub-zero -i <file> [options]

Core:
  --source-lang <code>    source language (default: ja)
  -l, --lang <code>       target language (default: en)
  --profile <name>        fast | balanced | strict
  --gpu                   use CUDA when available
  --offline               offline-only backends

Speed:
  --parallel              parallel chunked transcription
  --stream                progressive output (chunk-by-chunk)
  --workers <N>           max whisper workers (auto-capped for GPU VRAM)
  --chunk-duration <sec>  target chunk size (default: 300)

Quality:
  --speaker-aware         per-character voice priors + discourse consistency
  --speaker-diarize       audio-based speaker diarization
  --verify                check output timing against audio VAD
  --trace-runtime         emit per-stage timing sidecar
  --lfas-control          let LFAS override DOOM-QLOCK's plan selection

Backends:
  --whisper-bin <path>    path to whisper.cpp binary (fastest)
  --whisper-model <path>  path to GGML model file
  --mt-model <name>       neural MT model override
  --phrase-table          skip neural MT, use phrase-table fallback
```

---

## project structure

```
src/engine/              the pipeline + DOOM-QLOCK + quality gates
src/engine/lfas.rs       LFAS scheduler (4 theorems)
src/engine/f3_stream.rs  streaming MI + Hellinger estimator
src/engine/parallel.rs   chunked transcription worker pool
src/engine/pipeline.rs   dual-path convergence orchestrator
tui/src/                 ratatui dashboard
scripts/                 benchmarks, model bootstrap, quality verifier
examples/                case studies (medical NLP, recsys)
```

---

## status

218 tests. zero clippy warnings. CI on Linux, macOS, Windows.

```bash
cargo test --workspace --locked
```

---

## license

MIT.

---

*control is an illusion. determinism isn't.*
