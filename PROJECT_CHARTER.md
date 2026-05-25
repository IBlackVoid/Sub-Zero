# Project Charter

## Mission

Sub-Zero is an offline-first subtitle translation engine that turns local media
or subtitle files into translated SRT outputs with auditable quality metadata.

## Non-Negotiables

- Correctness: SRT timing must remain parseable, monotonic, and tied to the
  source cue or detected audio segment.
- Privacy: user media, subtitles, model inputs, and generated metadata must not
  leave the machine unless the user explicitly runs external tooling.
- Security: untrusted media, SRT text, CLI args, model files, and event clients
  are treated as hostile inputs at process and parser boundaries.
- Performance: long media must be chunkable, resumable, and measurable through
  runtime trace sidecars.
- Reliability: failure modes must return actionable errors instead of silent
  partial success.
- Reproducibility: public quality and latency claims need a command, fixture,
  dataset, or documented blocker.

## Users and Use Cases

Primary users:

- People who want local subtitle translation without cloud APIs.
- Power users processing long videos, anime, streams, or drama content.
- Contributors interested in offline ASR/MT pipelines and subtitle evaluation.

Critical workflows:

- Translate an existing SRT with local phrase-table or neural MT.
- Transcribe media locally, translate, and emit sidecar metadata.
- Run a TUI dashboard over an engine event stream.
- Verify output quality and timing with reproducible scripts.

Failure impact:

- Bad timing makes subtitles unusable.
- Hidden network use violates the core privacy promise.
- Overstated quality claims damage trust.
- Missing reproducibility blocks open-source adoption.

## Scope

In scope:

- Rust CLI engine.
- Rust TUI dashboard.
- Local ffmpeg/ffprobe integration.
- Local whisper and NLLB/CTranslate2 integration.
- Subtitle quality, latency, and human-eval tooling.
- Project artifacts required for public release readiness.

Out of scope for the first public release:

- Cloud translation APIs.
- Hosted user accounts or remote storage.
- Shipping large model weights in the git repository.
- Shipping copyrighted media fixtures.
- Claims that the current implementation beats professional translators.

Explicitly forbidden shortcuts:

- Network access in `--offline` workflows.
- Silent fallback that hides quality-gate failure.
- Committing private media, model weights, logs, or personal notes.
- Marketing claims without a reproducible verification path.

## System Constraints

- Languages: Rust 2021 for product code; Python 3.10+ for scripts.
- Runtime: local desktop/CLI environment.
- External tools: ffmpeg/ffprobe; optional whisper.cpp or Python whisper;
  optional CTranslate2/SentencePiece for neural MT.
- Data stores: local filesystem sidecars and per-user `.sub-zero` state.
- Supported platforms target: Windows, Linux, macOS.

## Success Metrics

Functional:

- Existing SRT translation succeeds with phrase-table fallback.
- Neural MT path succeeds when local dependencies and models are present.
- Video transcription path succeeds when ffmpeg and whisper dependencies exist.

Performance:

- Runtime traces expose per-stage latency.
- Chunked and streaming modes can process long media without unbounded memory.

Security:

- Sidecar event servers bind loopback by default.
- No secrets or private media are required in the public repository.
- Subprocess calls use structured arguments rather than shell strings.

Reliability:

- `cargo fmt`, `cargo clippy`, and `cargo test` pass on a clean clone.
- Integration smoke tests document optional dependency requirements.

Usability:

- A first-time user can run a small smoke test in under 10 minutes after
  installing prerequisites.

## Acceptance Criteria

- [ ] Public-safe repository contents are reviewed.
- [ ] Architecture, threat model, benchmarks, verification, and risk register
  describe the actual system.
- [ ] CI gates pass on all supported OS targets.
- [ ] README separates product facts, research claims, and roadmap items.
- [ ] A legal fixture set supports reproducible smoke and benchmark runs.
- [ ] Release packaging excludes private media, local logs, and model weights.

## Open Questions

- Which small legal media fixture should be used for public demos?
- Should event sidecars support token auth for remote dashboards?
- Should the learned quality gate be enforced by default or metadata-only?
- What is the minimum public benchmark corpus that is legally redistributable?
