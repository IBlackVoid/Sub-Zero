# Architecture

## Overview

Sub-Zero is a local subtitle pipeline. The CLI owns orchestration, input
resolution, transcription, translation, post-processing, quality gates,
metadata, and optional event streams. The TUI is a companion process that
spawns the CLI and renders events, previews, progress, and hidden local assets.
Python scripts are boundary adapters for model inference, evaluation, benchmark
generation, and human-rating workflows.

## Invariants

- User media and subtitle text stay local unless a user explicitly invokes
  external tools outside Sub-Zero.
- SRT output is parseable, sequentially indexed, and monotonic after stitch and
  compaction.
- Quality claims are represented as metrics, sidecars, or explicit warnings.
- Runtime policy changes are recorded in traces or checkpoint metadata.
- Event sidecars are observability surfaces, not authority surfaces.

## Module Ownership

| Module | Owns | Must Not Own |
| --- | --- | --- |
| `src/main.rs` | CLI parsing, top-level run orchestration | Translation or ASR internals |
| `src/engine/pipeline.rs` | Main pipeline state machine and quality gates | UI rendering |
| `src/engine/transcribe.rs` | Whisper/ffmpeg transcription boundary | Translation policy |
| `src/engine/translate.rs` | Backend selection, phrase-table fallback, neural MT policy | Media probing |
| `src/engine/neural_mt.rs` | Python MT process/daemon protocol | CLI argument parsing |
| `src/engine/chunker.rs` | Audio chunk boundaries and chunk export | Subtitle quality scoring |
| `src/engine/stitcher.rs` | Chunk timestamp offset and deduplication | ASR or MT backend choice |
| `src/engine/doom_qlock/*` | Hardware/content probing, adaptive plan, history | User-facing claims |
| `src/engine/pipeline/*` | Focused pipeline helpers for scenes, speaker, trace, verify | TUI state |
| `tui/src/*` | Terminal UI, runner, local preferences, encrypted assets | Core translation logic |
| `scripts/*` | Offline evaluation, benchmark, release, and model helper tooling | Runtime core invariants |

## Data Flow

1. Creation: user supplies media or SRT through CLI or TUI.
2. Validation: CLI validates flags; pipeline checks input existence and source
   mode; parsers reject unusable SRTs.
3. Transformation: media may be probed/extracted, chunked, transcribed,
   stitched, translated, post-processed, compacted, and quality-gated.
4. Persistence: output SRT, metadata sidecar, optional trace sidecar,
   checkpoint state, voice priors, glossary, and TUI preferences are written to
   local files.
5. Observation: JSON events may be printed, appended to a file, or forwarded to
   local HTTP/WS sidecars.
6. Deletion/retention: temporary work and checkpoints are local; release policy
   must define which generated artifacts stay out of git.

## Boundaries

External systems:

- ffmpeg/ffprobe.
- whisper.cpp or Python whisper.
- Python CTranslate2/SentencePiece MT scripts.
- Local terminal, filesystem, and optional event clients.

Trust boundaries:

- User input to CLI parser.
- User media/SRT to parser and subprocess boundaries.
- Model and script files to local process execution.
- Event clients to sidecar streams.
- TUI save/snapshot paths to filesystem writes.

Sync boundaries:

- CLI pipeline stages are mostly synchronous with explicit subprocess waits.
- Parallel transcription uses Rayon worker pools.
- Streaming modes use bounded channels.

Async boundaries:

- HTTP/WS sidecar threads.
- TUI process runner polling.
- MT daemon subprocess frames.

## API Contracts

Public CLI:

- Inputs are paths.
- Output naming is `<stem>.<target>.srt`.
- `--offline` is a privacy contract and must not introduce cloud calls.
- `--http-events` and `--ws-events` are loopback-only by default.

Internal API:

- `SubtitlePipeline::process_input` returns the final output path or a typed
  pipeline error.
- `Translator::translate_all_with_extra_tags` preserves cue count and timing
  unless the caller applies later compaction.
- `Transcriber` returns an SRT path and an audio path suitable for verification.

Error model:

- User-facing errors should name the failed boundary and path/flag involved.
- Optional observability failures may warn but must not corrupt output.
- Strict profile may fail on quality gates where balanced/fast warn.

Versioning/migration:

- Metadata sidecars include a version.
- Learned quality gate schemas must list accepted versions in code.
- Public release artifacts must not require local private state.

## Failure Modes

| Failure | Expected Behavior | Verification |
| --- | --- | --- |
| Missing input | Exit with clear path error | CLI test |
| Missing ffmpeg/ffprobe | Exit with dependency hint | Integration smoke |
| Missing whisper model | Exit with model path hint | Unit/integration test |
| Neural MT unavailable | Fall back or fail according to flags | Translator tests |
| CUDA OOM | Retry/fallback when allowed | Transcribe/MT tests |
| Degenerate sidecar SRT | Rescue or fail depending on flags/profile | Pipeline tests |
| Remote event bind by default | Refuse unless explicitly allowed | Sidecar tests |
| Invalid learned gate schema | Warn and disable learned gate | Unit tests |

## Observability

Logs:

- Human-readable stderr messages today.
- JSON events for integrations when enabled.

Metrics:

- Metadata sidecar quality block.
- Runtime trace per-stage timings.
- Benchmark reports under `benchmarks/reports`.

Traces:

- `<input>.sub-zero.trace.json` when `--trace-runtime` is enabled.

Health checks:

- HTTP sidecar `/health` returns local JSON status.

## Decision Links

- `docs/F1_latency.md`: latency model and verifier.
- `docs/F2_subtitle_information_bottleneck.md`: research framework.
- `docs/PRODUCTION_PLAN.md`: larger production-readiness roadmap.
- `THREAT_MODEL.md`: current defensive review.
- `VERIFY.md`: release verification gates.
