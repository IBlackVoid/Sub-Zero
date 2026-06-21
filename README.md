# VoiDex

**offline subtitle engine. nothing leaves your machine. ever.**

[![CI](https://github.com/IBlackVoid/VoiDex/actions/workflows/ci.yml/badge.svg)](https://github.com/IBlackVoid/VoiDex/actions)
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
voidex -i movie.mkv --source-lang ja --lang en --gpu --parallel
```

output: `movie.en.srt` + `movie.voidex.json` (quality audit sidecar).

---

## why this exists

every subtitle tool either:
- sends your content to a cloud API (privacy violation)
- gives you zero quality metrics (hope-based engineering)
- can't scale past a single file without manual babysitting

VoiDex does none of that. it runs entirely offline, tells you exactly how
confident it is per scene, and when quality drops below the floor it retries
the weak scenes at a stronger model — and is honest in the sidecar when it
still can't reach the bar.

---

## features

| feature | what it does |
|---------|-------------|
| **DOOM-QLOCK** | adaptive runtime scheduler. probes your hardware, picks a plan from history, learns from each run |
| **parallel chunked transcription** | splits audio at silence boundaries, transcribes in parallel, stitches with overlap dedup |
| **LFAS adaptive scheduling** | empirical-Bernstein bandit (UCB-V) driven by a reference-free quality signal, mapped to a conformal coverage floor — no reference subtitle needed at inference |
| **C-BHC coverage bound** | provable coverage guarantee via Bretagnolle-Huber inequality |
| **Rényi-Hellinger sharpening** | tighter coverage bound computed directly from streaming histograms |
| **learned quality gate** | logistic + isotonic + conformal prediction — PASS/REJECT/ABSTAIN with calibrated confidence |
| **speaker-aware translation** | per-character voice priors, discourse consistency, register tagging |
| **character glossary** | persistent name canonicalization across episodes |
| **per-segment MT escalation** | retries only low-quality scenes at a stronger model (600M→1.3B), bounded by a per-profile budget, with live telemetry |
| **live TUI dashboard** | terminal UI with real-time progress, three visual modes, waveform display |

---

## quick start

### install

```bash
git clone https://github.com/IBlackVoid/VoiDex
cd VoiDex
cargo build --release
```

binaries land in `target/release/`:
- `voidex` — opens the live dashboard with no arguments; runs the engine CLI when passed input/options
- `voidex-tui` — direct dashboard entry point when building the whole workspace

### prerequisites

- **Rust** 1.80+ (`rustup default stable`)
- **ffmpeg** on PATH (audio extraction + media probing)
- **Python 3.10+** with `openai-whisper` installed (transcription backend)
- (optional) **whisper.cpp** binary for 5-10x faster transcription
- (optional) **NLLB-200** in CTranslate2 format for neural translation

### first run

```bash
# transcribe + translate a video (GPU accelerated)
voidex -i video.mp4 --source-lang ja --lang en --gpu --parallel

# just transcribe english audio
voidex -i podcast.mp4 --source-lang en --lang en --gpu

# translate an existing SRT
voidex -i existing.ja.srt --lang en

# fire up the dashboard
voidex
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
    └─── output: video.en.srt + video.voidex.json (audit sidecar)
```

---

## the dashboard

```bash
voidex
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

VoiDex doesn't ask you to trust it — every run writes a sidecar
(`<name>.voidex.json`) with a calibrated verdict, per-scene scores, and
hallucination signals, so you see exactly where the output is trustworthy.

```jsonc
"verdict":  { "pass": true, "reason": "quality gate passed" },
"quality":  {
  "per_scene": [ { "scene": 1, "score": 1.0, "floor": 0.8, "pass": true } ],
  "semantic": { "adjacent_repeat_ratio": 0.004, "anomaly_ratio": 0.012 }
}
```

### where it's strong, where it isn't

This is the honest version. VoiDex's flagship JA→EN path is built for
**dialogue-driven content** — narration, interviews, lectures, documentaries,
scripted drama, meetings. There it delivers production-quality subtitles.

**Rapid casual conversational speech** (gameplay/stream banter dominated by very
short, context-poor utterances) is a known limitation: offline sentence-level
NMT degrades on it. VoiDex does not hide this — the quality gate scores the
output, and on the speed profiles it emits best-effort subtitles with
`verdict.pass=false` and the reason retained, while Strict refuses rather than
ship low-quality output. (Active research on casual-speech MT is tracked in
`docs/`.)

| content class | JA→EN quality | gate behavior |
|---|---|---|
| dialogue / narration / lecture / documentary | strong | `verdict.pass=true` |
| rapid casual / stream banter | degrades | gate fires; Fast/Balanced emit best-effort (`pass=false`), Strict hard-fails |

### performance

Real measured throughput and resource use (RTX 4070 Laptop, release build) are
in [BENCHMARKS.md](BENCHMARKS.md) — Fast 12–22× real-time, Balanced 6–16×,
Strict 1–8×, with the full method and raw rows. Robustness checks (checkpoint
resume, GPU-absent CPU fallback, profile-aware gates) are in [EVALS.md](EVALS.md).

The mutual-information fidelity tool (`scripts/quality_gate/verify_fidelity_bound.py`,
Kraskov k-NN estimator) ships for users who have a trusted human reference and
want to quantify machine-vs-reference fidelity on their own corpus.

---

## CLI reference

```
voidex -i <file> [options]

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
