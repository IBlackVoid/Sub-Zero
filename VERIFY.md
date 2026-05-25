# Verify

## One-Command Check

```bash
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
```

On Windows PowerShell, run the same commands separately if shell chaining is
inconvenient.

## Required Gates

- [x] Format/lint: `cargo fmt --all -- --check`
- [x] Typecheck/build: covered by clippy/test builds
- [x] Unit tests: `cargo test --workspace`
- [x] Integration tests: `cargo test --workspace` includes current Rust
  integration tests; optional model/ffmpeg branches need env gates
- [ ] E2E/smoke: run `scripts/ci/run_integration_smoke.sh` with ffmpeg and
  Python neural dependencies available
- [ ] Security checks: run secret scan and sidecar adversarial tests before
  release
- [ ] Performance checks: run `scripts/ci/run_perf_trace_smoke.sh` and latency
  verifier on representative traces
- [ ] Docs/config parse: review public docs for stale claims and private paths

## Critical Flows

| Flow | Check | Expected Result |
| --- | --- | --- |
| Existing SRT, phrase-table fallback | `cargo test --test cli_integration phrase_table_cli_smoke` | Output SRT exists and contains expected translations |
| CLI help | `cargo test --test cli_integration help_flag_exits_success` | Help exits successfully |
| TUI core state | `cargo test -p sub-zero-tui` | TUI unit tests pass |
| Local event streams | `cargo test -p sub-zero ws_sidecar http_sidecar` | Local streams pass, remote defaults rejected |
| Learned gate schema | `cargo test -p sub-zero learned_gate` | Current model schema is accepted by code |
| Runtime tracing | `scripts/ci/run_perf_trace_smoke.sh` | Trace contains required stages |

## Manual Checks

- Run one fresh-clone smoke on Windows, Linux, and macOS.
- Confirm README install path matches release artifacts.
- Confirm no private/copyrighted media or model weights are staged for public
  release.
- Confirm `--offline` workflows do not intentionally invoke network APIs.

## Known Gaps

- Public legal fixture set is not finalized.
- Large local models and media exist in the current working tree.
- Some docs make research/product claims that need clearer labels.
- TUI save/snapshot path policy needs hardening before hostile-user claims.

## Release Go/No-Go

Go criteria:

- Required gates pass on supported OS targets.
- Public repo audit removes private media, local logs, generated outputs, and
  model weights from tracked release content.
- README, architecture, threat model, benchmark, and verify files agree.
- A tagged release can be built from a fresh clone.

No-go criteria:

- CI red.
- Learned gate silently disabled while README claims it is active.
- Event sidecars exposed remotely without explicit user opt-in.
- Private or legally unsafe artifacts staged.

Rollback check:

- Release branch can be reset to the previous tag.
- Generated artifacts are reproducible from scripts or excluded from release.
