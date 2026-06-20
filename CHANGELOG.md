# Changelog

All notable changes to VoiDex are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **F.4 LFAS theorem system** — three theorems composing F.3
  counterfactual MI, UCB-Bernstein bandits, and C-BHC coverage
  transfer into the first label-free production coverage guarantee
  with sublinear regret (`docs/F4_lfas.md`, `src/engine/lfas.rs`).
- **C-BHC coverage bound** — sharper-than-Pinsker coverage bound
  `c_dep >= (1 - alpha) - sqrt(1 - exp(-Delta_I))`, non-vacuous
  under large drift where Pinsker diverges.
- **LfasScheduler** — `O(K)` memory, `O(1)` per-token adaptive
  scheduler using F.3 as label-free bandit reward. Wired into
  DOOM-QLOCK (`src/engine/doom_qlock.rs`).
- LFAS synthetic verification experiment
  (`scripts/bench/lfas_synthetic_bandit.py`): sublinearity PASS,
  2.3x coverage regret improvement.
- Content-addressed cache pilot (ADR-0001 Phase A) with
  `MemoryContentCache` and `FsContentCache`.
- `AsrPermit` typed token for concurrency control in stream-async.
- `bench-internals` feature gate for benchmark-only exports.
- MSRV CI enforcement (reads `rust-version` from `Cargo.toml`).
- GitHub issue templates (bug report, feature request).
- GitHub pull request template.

### Changed

- Learned quality gate upgraded to v2.0: six reference-derived
  features removed after F.3 leakage audit (100% precision/recall).
- Methods paper extended with C-BHC and LFAS sections.
- MODEL_CARD.md updated with C-BHC + LFAS theorem claims.
- `parse_srt_cached` gated behind `bench-internals` feature to
  prevent downstream performance traps.

### Fixed

- F.3 case studies (medical, recsys) empirical bounds now track
  Pinsker upper bound in direction and order of magnitude.

## [0.1.0] - 2026-05-19

### Added

- Initial release: offline subtitle translation engine.
- ASR via whisper.cpp / py-whisper backends.
- Neural MT via NLLB-200 (CTranslate2).
- DOOM-QLOCK adaptive scheduler with hardware probing and
  history-based plan lookup.
- Split-conformal quality gate (PASS / REJECT / ABSTAIN).
- TUI dashboard with waveform, runner visualization, splash screen.
- Cross-platform CI (Linux, macOS, Windows).
- 4-target release workflow with SHA-256 checksums.
- F.1 M/D/1 tandem latency theorem.
- F.2 subtitle information bottleneck framework.
- F.3 counterfactual MI leakage diagnostic + Lemma 1
  (Mask-Induced Coverage Bound).

[Unreleased]: https://github.com/IBlackVoid/VoiDex/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/IBlackVoid/VoiDex/releases/tag/v0.1.0
