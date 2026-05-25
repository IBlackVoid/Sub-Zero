# Roadmap

> Where Sub-Zero is, where it is going, and what blocks each milestone.
> This file is the single public source of truth for the launch plan; do
> not duplicate items here without retiring them in `RISK_REGISTER.md`,
> `VERIFY.md`, or the relevant ADR.

## Current state (2026-05-20)

Pre-0.1. Engine compiles and tests clean on three OS in CI. Library +
binary split done; `cargo publish --dry-run` is green. Doctrine artifacts
(`PROJECT_CHARTER.md`, `ARCHITECTURE.md`, `THREAT_MODEL.md`, `VERIFY.md`,
`BENCHMARKS.md`, `RISK_REGISTER.md`) describe the actual system.

A ten-specialist read-only audit on 2026-05-19 (architect 78,
security 71, performance 70, ML 62, systems 72, frontend 66,
researcher 62, theory 38, verifier 38, devops 22) sized the gap between
"interesting solo repo" and "credible open-source 0.1." This roadmap
encodes the closure plan.

## Phase 0 — release-blockers

Goal: a tagged 0.1 a Linux/macOS/Windows user can install and run, with
honest claims and a defensible release pipeline.

| # | Item | Status |
| --- | --- | --- |
| 1 | `src/lib.rs` split; `Cargo.toml` `license` / `repository` / `rust-version` / `keywords` / `categories` / `[lib]` / `[[bin]]` | done |
| 2 | `thiserror` derive across all per-module error enums | done |
| 3 | Release pipeline (`.github/workflows/release.yml`) + supply-chain gates (`cargo-deny`, `cargo-audit`) + `deny.toml` | done |
| 4 | Contributor scaffolding — `CONTRIBUTING.md`, `SECURITY.md` (landed); `CODE_OF_CONDUCT.md`, issue/PR templates, `CODEOWNERS` deferred to Phase 1 polish | done (Phase 0 cut) |
| 5 | Model bootstrap script that actually downloads from HuggingFace with pinned SHA-256 | done — `scripts/release/bootstrap_models.sh` pins NLLB `f8d333a0…`, whisper-base `e37978b9…`, whisper-large-v3 `06f233fe…` (2026-05-22 HEADs). Uses `huggingface-cli` with `git+git-lfs` fallback; verifies HEAD matches the pin. |
| 6 | Public legal fixture (CC0 audio + reference SRT under `fixtures/`) | done — `fixtures/clip_10s.wav` (JFK 1961 inaugural excerpt, public domain, US federal government work, 17 U.S.C. § 105) + `clip_10s.en.srt` + `fixtures/README.md` documenting provenance; allowlisted in `repo-hygiene.yml`. English not Japanese; rationale documented inline. |
| 7 | Repository scrub — `.gitignore` + `.github/workflows/repo-hygiene.yml` guard landed (rejects tracked `*.mp4`/`*.mkv`/`*.safetensors`/etc. and any tracked file > 5 MiB); deletion of existing untracked private media from working tree is maintainer-driven | partial — guard landed, working-tree cleanup pending |
| 8 | Re-train learned quality gate with frozen-by-hash held-out case set; embed `training_seed` + `data_sha256` + `git_commit` in `learned_gate.json`; report bootstrap CI + leakage ablation | harness done — `scripts/train_learned_gate.py` emits the v1.1 schema + `training_seed` / `data_sha256` / `git_commit` / `f1_bootstrap_ci95` / `exclude_delta_features` provenance. `--exclude-delta-features` performs the audit's leakage ablation. Sample corpus at `scripts/learned_gate_corpus_sample.jsonl`. **Production model retrain still needs the maintainer's actual labelled corpus.** |
| 9 | F.2 honesty pass — reframed as applied synthesis: T1 → Proposition 1 (variational-IB restatement, no novelty), T2 → Observation 2 (bias-variance proof withdrawn), T3 → Remark (DPI), T4 → Conjecture 4 (informal scaling), T5 → Proposition 5 (additive surrogate). W1–W7 promoted as the doc's main contribution. §15 honestly enumerates prior art vs. what is new. | done |
| 10 | Headline claim honesty — `docs/REVOLUTION.md` §B.2/§B.3/§B.4 and `README.md` test count | done |

Phase 0 closes when items 4–9 land and the existing CI is green on a
fresh tag.

## Phase 1 — community-ready

Goal: external contributors can find their footing, the engine answers
its own performance claims, and the security review's medium findings
are closed.

Status of the 2026-05-22 batch (lanes A, C, B, D, E, F):

- Rust quality drags from the 2026-05-19 audit:
  - **done** — `stream_async` replaced `Arc<Mutex<mpsc::Receiver>>` with
    `crossbeam-channel` MPMC; busy-wait `acquire_asr_slot` replaced with
    a `Condvar`-backed `AdaptiveSemaphore` that parks waiters and
    re-checks the replanner's dynamic limit on a 50 ms timeout.
  - **partial** — `postprocess::remove_filler_phrases` now maintains
    `out` and `lowered` in lockstep instead of relowercasing on every
    match, and the space collapse is single-pass O(n). The wider
    `Arc<str>` cue migration is deferred — too many sites for a single
    PR; tracked separately.
  - **done** — `srt::parse_srt` streams over `&str` with
    `memchr::memchr_iter`, no more `replace("\r\n","\n")` copy. Added
    `MAX_SRT_INPUT_BYTES = 50 MiB` cap and a typed `SrtError::SizeCap`.
- TUI quality drags:
  - **done** — `NO_COLOR` and `SUB_ZERO_TUI_REDUCED_MOTION` honoured
    through `tui::accessibility::Accessibility::from_env()`. Reduced
    motion pins to last frame; idle redraw drops to 250 ms.
  - **kept** — `tui/src/slots.rs` is not dead code (it carries the
    easter-egg asset directory constants). Documented in-place.
  - **deferred** — splitting `tui/src/app.rs` further. Big refactor.
- Security finding closures beyond `THREAT_MODEL.md`:
  - **done** — WS `Origin` allowlist now parses `(scheme, host)` and
    rejects every host-prefix spoof (`localhost.evil.com`,
    `127.evil.com`, `[::1].evil.com`, userinfo tricks). Tests cover
    spoofing, malformed input, and non-HTTP schemes.
  - **done** — TUI `:save` / `:snap` resolve through
    `paths::resolve_write_target` against
    `tui_write_root()` (CWD or `SUB_ZERO_TUI_WRITE_ROOT`). `..`
    traversal rejected. Tests cover all the bad shapes.
  - **done** — MT daemon length-prefixed stdio has a 64 MiB
    `MAX_MT_FRAME_BYTES` cap; oversize frames raise typed I/O errors
    before allocation.
- Benchmark baselines:
  - **done** — Criterion benches under `benches/bench_engine.rs` cover
    `srt::parse_srt`, `stitcher::deduplicate_overlaps`, and
    `chunker::pick_boundaries` via the `bench_internals` shim in
    `lib.rs`.
  - **done** — F.1 verifier now correctly maps parallel stages to the
    plan's worker count (was always returning 1 for every stage).
  - **deferred** — populating `BENCHMARKS.md` with measured numbers;
    requires actual benchmark runs on representative hardware.
- Cross-platform smoke:
  - **done** — `cargo xtask smoke` (under `xtask/`) ports the
    non-fixture portions of `scripts/ci/run_integration_smoke.sh`. It
    runs fmt, clippy (-D warnings), doc-warnings, the whole test
    suite, and the fixture-gated integration cases. Cross-platform
    because it's Rust, not bash.
- Dependency hygiene:
  - **done** — `.github/dependabot.yml` covers cargo, github-actions,
    and pip on a weekly cadence with grouped minor/patch updates.
- Human eval that exists on disk:
  - **deferred** — collecting ≥ 30 pairs × ≥ 3 raters needs maintainer
    time, not code.

## Phase 2 — revolution-grade

Goal: the project becomes the reference offline subtitling engine and
the F.2 framework actually ships its open W5 condition.

- **Mask-Induced Coverage Bound (F.3 §4) — done 2026-05-23.** Composes
  Vovk split-conformal + Barber-Candès-Ramdas-Tibshirani conformal-
  under-shift + Pinsker + F.3 §3 KL lower bound into a single closed-
  form lemma. Empirically verified against the F.3 medical case study.
  Theory-lab review verdict: "the only new lemma available in this
  repo; provable from textbook ingredients; novel as a named result
  in 2026; one section of a methods paper." Methods paper now has
  three sections of content: F.3 diagnostic, two case studies, the
  Mask-Induced Coverage Bound. Drafting the paper is the next step;
  the artifacts are ready.

- **Content-addressed demand-driven build graph — proposed and
  informed by Phase A pilot (ADR-0001).** Phase A landed 2026-05-23:
  `src/engine/cache.rs` ships the content-addressing primitive
  (trait + memory/fs implementations + SHA-256 keys with function-id
  namespacing), wired to `srt::parse_srt_cached` and benched at
  100/1000/10000 cues. Empirical finding (`docs/adr/0001_phase_a_measurements.md`):
  cache HIT is 1.7-1.9× SLOWER than uncached parse on this function
  — content-addressing only pays off above a ~1 ms recompute threshold.
  ADR-0001 stays *proposed* but is now *informed*: the recalibrated
  design caches only the expensive nodes (transcribe, translate)
  and uses bincode (not JSON) for cached payloads. The full
  3-6 month Phase A-E commit remains a maintainer decision; the
  Phase A pilot has produced the kill-criterion-grade measurement
  data needed to inform it. Architect-review 2026-05-23 identified
  this as the single structural move that elevates the engine from
  "solid-senior conventional pipeline" to a category-defining artifact. Recast the
  engine as a Salsa/Bazel-style graph where every intermediate
  artifact is content-addressed; unifies four currently-independent
  ambitions (retry-a-chunk, branched A/B runs, replay debugging,
  byte-identical determinism) into one primitive. Reframes
  DOOM-QLOCK as "a planner observing the graph" and the
  LiveHistogramReplanner as "a controller mutating the graph."
  Estimated effort: 3–6 months, staged Phase A–E with a hard kill
  criterion at end of Phase B (cache hit rate ≥ 40% or revert).
  Read `docs/adr/0001-content-addressed-build-graph.md` before
  starting. If shipped, the headline claim of the project becomes
  "first content-addressed subtitle engine" — defensible against
  the entire 2026 competitive field per the researcher 2026-05-23
  audit.

- Multimodal grounding (F.2 W5). SeamlessM4T-v2 + a small vision encoder
  pulling on-screen referents. Theorem 3 says the bound is monotonic —
  adding it can only widen the fidelity gap. This is the real research
  contribution worth being loud about. Theory-lab review identifies
  this as the *second* most tractable theorem in the project (after
  the Mask-Induced Coverage Bound) but notes it requires a generative
  model of visual referents that does not yet exist in the repo —
  estimated 1–2 months of research work.
- Reproducibility from-scratch. One command on a clean clone produces
  byte-identical SRT on identical input: `SOURCE_DATE_EPOCH`,
  `cargo build --locked`, pinned model SHAs, deterministic decoding.
- Showcase TUI. Cut a ≤ 90-second demo video; reduced-motion variant
  for stills.
- Trait-based backends. `TranslatorBackend` and `Transcriber` become
  traits, not enums. Lets the community drop in `mistral.rs`,
  distil-whisper, SeamlessM4T, Madlad-400 without editing two 1.4 k-LOC
  files.
- Distribution channels. Homebrew tap, AUR, Chocolatey, generated from
  the release workflow. `cargo install sub-zero` working end-to-end.
- Sigstore / cosign keyless signing on release artifacts; CycloneDX
  SBOM per artifact; OSSF Scorecard badge ≥ 7.

## Decisions still open

- **Production corpus for the learned gate** (item 8 finisher). The
  retrain harness is in place; running it against the maintainer's
  real labelled corpus will produce the final `models/learned_gate.json`.
  When the corpus lands, store it outside the repo (it likely contains
  copyrighted reference SRTs), then run:
  `python scripts/train_learned_gate.py --corpus path/to/corpus.jsonl --output models/learned_gate.json --exclude-delta-features`.

## Cross-references

- Current open risks: `RISK_REGISTER.md`
- Release go/no-go: `VERIFY.md` (currently no-go until items 4–9 land)
- Architecture invariants: `ARCHITECTURE.md`
- Threat model: `THREAT_MODEL.md`
- Performance targets: `BENCHMARKS.md`
- The 2026-05-19 specialist audit: documented in this repo's history;
  the integrated rating and per-axis specialist scores are summarised
  in this roadmap's *Current state*.
