# Benchmarks

## Performance Targets

Latency:

- Existing SRT translation smoke should finish in seconds on a developer
  machine.
- Long media should emit runtime traces so regressions can be inspected by
  stage.

Throughput:

- Chunked transcription should scale with worker count until the bottleneck
  becomes GPU/CPU inference, disk I/O, or memory pressure.

Memory:

- Streaming and async modes should avoid materializing unbounded media or cue
  state.
- Model inference memory is governed by selected ASR/MT models and batch sizes.

Startup:

- Phrase-table mode should start without Python model dependencies.
- Neural modes may pay model load cost; MT daemon mode amortizes that cost.

Rendering/FPS:

- TUI rendering should remain responsive while the engine runs as a child
  process.

Database/query:

- Not applicable; local filesystem only.

## Workload Shape

Expected input size:

- SRT files from short clips through full episodes.
- Media files from short clips through multi-hour streams.

Worst-case input size:

- Long videos with dense speech, poor silence gaps, large model selections, and
  high worker counts.

Distribution:

- Primary quality target is serialized dialogue such as anime, drama, and
  gameplay streams.

Update/query ratio:

- Batch jobs are write-heavy; TUI preview tails generated SRT and event files.

Concurrency:

- Rayon parallel transcription workers.
- Streaming bounded channels for ASR/MT overlap.
- Local HTTP/WS sidecar threads.

## Baselines

| Benchmark | Baseline | Environment | Date |
| --- | --- | --- | --- |
| Workspace unit/integration tests | `cargo test --workspace` passed after one flaky WS rerun | Local Windows workspace | 2026-05-19 |
| Lint/format | `cargo fmt --all -- --check`; `cargo clippy --workspace --all-targets -- -D warnings` passed | Local Windows workspace | 2026-05-19 |
| Learned gate dataset | 175 rows, holdout F1 about 0.865 in `models/learned_gate.json` | Existing local benchmark artifacts | 2026-05-19 |

## Benchmark Commands

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# Optional dependency smoke.
bash scripts/ci/run_integration_smoke.sh

# Phrase-table trace smoke.
bash scripts/ci/run_perf_trace_smoke.sh

# Latency theorem verification over produced traces.
python scripts/bench/verify_latency_bound.py benchmarks/reports/**/*.trace.json
```

## Profiling Notes

CPU:

- Rust parsing/stitching/postprocess is expected to be cheaper than ASR/MT.
- Python whisper and CTranslate2 dominate CPU/GPU work.

Memory:

- Model weights dominate memory.
- MT batch size and token ceilings are the main tunables.

I/O:

- ffmpeg extraction, chunk WAV files, checkpoints, and sidecars create disk
  pressure on long runs.

Frontend/rendering:

- TUI reads tails and event files; it should not parse whole output files on
  every tick.

## Optimization Log

| Change | Hypothesis | Result | Keep/Revert |
| --- | --- | --- | --- |
| DOOM-QLOCK planning | Match worker/batch/chunk settings to workload/hardware | Existing tests cover heuristic behavior; corpus validation still needed | Keep |
| MT daemon | Avoid repeated Python/model startup | Needs benchmark by model/profile | Keep behind flag |
| Runtime trace sidecar | Make hot stages inspectable | Trace smoke exists | Keep |

## Rules

- No public performance claim without a command and representative fixture.
- Benchmarks must distinguish phrase-table, neural CPU, neural GPU, serial,
  parallel, stream, and stream-async modes.
- Generated benchmark reports must not be confused with source fixtures.
