# Contributing to Sub-Zero

Sub-Zero is an offline-first subtitle translation engine and TUI. Contributions
are welcome, but the project has a high bar for correctness, privacy, and
reproducibility because it processes user media and local files.

## Quick Start

Run the core gates before opening a pull request:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
```

The full CI surface also includes:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
cargo deny check --all-features --workspace
cargo audit --deny warnings
```

`cargo-deny` and `cargo-audit` are optional local tools, but CI treats them as
release gates.

## Prerequisites

- Rust stable, compatible with the `rust-version` in `Cargo.toml`.
- Python 3.10+ for helper scripts under `scripts/`.
- `ffmpeg` and `ffprobe` on `PATH` for media probing or transcription paths.
- Optional local ASR and MT dependencies, such as whisper.cpp or a local NLLB
  CTranslate2 model, when testing neural paths.

The phrase-table path is the lowest-dependency smoke path and should stay usable
without model downloads.

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/main.rs` | CLI argument parsing and top-level orchestration |
| `src/lib.rs` | Library crate root and engine exports |
| `src/engine/` | Translation pipeline, ASR/MT boundaries, sidecars, quality gates |
| `tui/` | Ratatui dashboard and TUI-only assets |
| `scripts/` | Python tools for MT, verification, evals, and TUI asset prep |
| `docs/` | Design notes, ADRs, production plan, and research docs |
| `.github/workflows/` | CI and release automation |

Project-level artifacts such as `PROJECT_CHARTER.md`, `ARCHITECTURE.md`,
`THREAT_MODEL.md`, `VERIFY.md`, `BENCHMARKS.md`, and `RISK_REGISTER.md` are
part of the engineering record. Update them when a change affects the boundary
or claim they document.

## Development Workflow

Build the workspace:

```bash
cargo build --workspace
```

Run the CLI help:

```bash
cargo run -- --help
```

Run the TUI:

```bash
cargo run -p sub-zero-tui --release
```

Keep changes small and reviewable. A pull request should usually do one thing:
fix one bug, add one feature, improve one subsystem, or update one documented
decision.

## Pull Request Expectations

Every pull request description should include:

1. What changed and why.
2. What was verified, with exact commands.
3. What was not verified and why.
4. Residual risk for reviewers to inspect.

For behavior changes, include a focused test or explain why an existing gate is
the right proof. For docs-only changes, run a targeted grep or review check that
proves links, commands, and referenced files are still accurate.

## Testing Guidance

Prefer the smallest meaningful gate first:

- Formatting only: `cargo fmt --all -- --check`
- One Rust module: focused `cargo test -p sub-zero <name>`
- TUI state or rendering logic: `cargo test -p sub-zero-tui`
- Cross-module engine behavior: `cargo test --workspace --locked`
- Public docs: `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked`
- Dependency changes: `cargo deny check --all-features --workspace` and
  `cargo audit --deny warnings`

Optional smoke scripts under `scripts/ci/` may require external tools or legal
fixtures. Do not make those mandatory unless the dependency footprint is small
and documented in `VERIFY.md`.

## Style

- Rust: use `rustfmt`; keep clippy clean under `-D warnings`.
- Python: use clear CLI-shaped helpers, `pathlib`, and typed public functions.
- YAML/TOML: use two-space indentation.
- Comments should explain why something is non-obvious, not narrate the syntax.
- Avoid committing decorative Unicode, emoji, generated noise, or machine-local
  absolute paths.

Use clear commit subjects with a scope when useful:

```text
engine: reject remote event binds by default
tui: harden encrypted asset loading
docs: add contributor security policy
```

## Privacy and Test Assets

Do not commit private media, copyrighted clips, personal subtitles, model
weights, local benchmark outputs, generated sidecars from private files, API
tokens, encryption keys, passphrases, or machine-specific logs.

Use synthetic fixtures or legally redistributable samples. If a test requires a
large or restricted asset, gate it behind an environment variable and document
the requirement in `VERIFY.md`.

## Security

Report vulnerabilities through `SECURITY.md`. Do not open public issues for
exploitable crashes, path traversal, remote event exposure, secret leakage,
offline bypasses, or encrypted-asset key disclosure.

Security-sensitive patches should include:

- The affected trust boundary.
- The attacker-controlled input.
- The asset being protected.
- A regression test or adversarial verification case.

## Conduct

Be specific, technical, and respectful. Assume contributors are acting in good
faith, but hold every change to the same evidence standard: clear behavior,
clear verification, and no hidden risk.

## License

By contributing, you agree that your contribution is licensed under the MIT
license in `LICENSE`.
