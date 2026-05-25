# Contributing to Sub-Zero

thanks for being here. Sub-Zero is an offline subtitle engine that takes
privacy and correctness seriously — it processes real media on real hardware
and makes provable claims about quality. contributions are welcome.

## before you start

```bash
# clone and build
git clone https://github.com/IBlackVoid/Sub-Zero
cd Sub-Zero
cargo build --workspace

# run the gates (must pass before any PR)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
```

if those three pass, you're good to open a PR.

## what you need

- **Rust** stable (check `rust-version` in Cargo.toml for MSRV)
- **ffmpeg + ffprobe** on PATH
- **Python 3.10+** for scripts (optional unless you touch MT/quality paths)

you do NOT need whisper models or NLLB to run `cargo test`. the test suite
uses fixtures and mocks. the phrase-table path works without any downloads.

## where things live

```
src/engine/           ← the core. pipeline, scheduler, quality gates, ASR/MT
src/engine/lfas.rs    ← the LFAS algorithm (the math lives here)
src/engine/parallel.rs← chunked transcription worker pool
src/main.rs           ← CLI parsing
tui/src/              ← terminal dashboard (separate binary)
scripts/              ← python helpers, benchmarks, model tools
docs/                 ← formal proofs, ADRs
benches/              ← criterion benchmarks
examples/             ← case studies showing the theory in other domains
fixtures/             ← test audio + expected outputs for CI
```

**want to add a new MT backend?** → look at `src/engine/translate.rs`
**want to add a quality gate?** → look at `src/engine/pipeline/learned_gate.rs`
**want to improve the scheduler?** → look at `src/engine/doom_qlock.rs`
**want to touch the TUI?** → it's fully independent in `tui/src/`

## good first issues

look for issues tagged `good first issue`. some areas that always need love:

- **more language pairs** in the phrase-table fallback
- **better error messages** when whisper/ffmpeg aren't found
- **TUI polish** — keybinds, color schemes, animations
- **docs** — usage examples, troubleshooting, platform-specific notes

## workflow

keep PRs small. one thing per PR. if it touches engine logic, add a test.
if it's a refactor, prove nothing changed with before/after test output.

## pull requests

your PR description should answer:

1. **what** changed and **why**
2. **how** you verified it (paste the command + output)
3. anything you're **not sure about** (so reviewers know where to focus)

## testing

```bash
cargo test -p sub-zero         # just the engine
cargo test -p sub-zero-tui     # just the dashboard
cargo test --workspace         # everything
```

## style

- `rustfmt` for Rust. clippy clean under `-D warnings`.
- comments explain *why*, not *what*.
- commit messages: `scope: what happened` (e.g. `engine: fix chunk timeout on CPU`)

## do NOT commit

- media files, model weights, or generated sidecars
- API keys, secrets, or machine-specific paths
- anything that would violate someone's privacy

## security issues

see `SECURITY.md`. don't open public issues for exploitable bugs.

## license

MIT. by contributing you agree your code is MIT-licensed.

---

that's it. keep it simple, keep it correct, keep it offline.
