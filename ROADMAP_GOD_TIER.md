# Sub-Zero → GOD_TIER Roadmap

> Personal roadmap for IBVoid. Each item pushes a specific dimension
> from current state toward 100/100. Ordered by impact × effort ratio.

---

## Current Baseline (2026-05-25)

| Dimension | Score | Bottleneck |
|-----------|-------|-----------|
| Architecture | 90 | pipeline.rs too long (1500+ lines) |
| Algorithm / Theory | 91 | Missing contextual bandits (10/10 needs minimax proof) |
| Code Quality | 85 | Dead code, some modules over-coupled |
| Testing | 75 | No E2E test with real audio, flaky SSE test |
| Reliability | 65→80 | Fixed: coverage gate, task mode, VRAM cap, timeout realism |
| Performance | 70 | Python whisper is 5-10x slower than whisper.cpp |
| DevOps / CI | 80 | No perf regression gate, no test audio fixture |
| Documentation | 85 | Missing quick-start for strangers |
| UX / Errors | 72 | Progress reporting is raw stderr noise |
| Open-Source | 78 | No `cargo install`, no Docker, no pre-built binaries |

---

## Phase 1: Bulletproof Reliability (→ 95/100 reliability)

### 1.1 — whisper.cpp as primary backend
- **Why:** 5-10x faster than Python whisper, single binary, no VRAM contention,
  deterministic, no Python dependency hell.
- **How:** Auto-detect `whisper-cli` in PATH. If found, prefer it over Python.
  Fall back to Python only when whisper.cpp is absent.
- **Impact:** Eliminates the entire timeout/VRAM class of bugs.

### 1.2 — VRAM budget-aware scheduling
- **Why:** Current fix caps GPU workers to 1. Smart approach: query
  `torch.cuda.mem_get_info()` or `cudaMemGetInfo`, divide available VRAM by
  model size → compute max safe workers.
- **How:** Probe at startup: `available_vram / model_vram_estimate`. Pass as
  ceiling to parallel worker pool.
- **Impact:** Laptop (8GB) = 1 worker. Desktop 4090 (24GB) = 3 workers. Auto.

### 1.3 — Streaming progress (no more raw stderr)
- **Why:** User sees raw whisper progress bars mixed with doom-qlock logs.
  Unreadable.
- **How:** Structured progress events via `--events-json`. TUI already consumes
  these. CLI should render a clean progress bar (indicatif-style) instead of
  forwarding subprocess stderr.
- **Impact:** Professional UX. Quiet by default, verbose with `--verbose`.

### 1.4 — Retry with backoff, not just timeout scale
- **Why:** After a timeout, the adaptive policy scales timeout but re-runs the
  same chunk. Better: split the failed chunk in half and retry each sub-chunk.
- **How:** On chunk failure, if chunk_duration > min_duration, bisect and
  retry each half with fresh timeout.
- **Impact:** Even if one 4-min section is pathological, the surrounding
  content still gets transcribed.

---

## Phase 2: Speed to God Tier (→ 95/100 performance)

### 2.1 — whisper.cpp + GPU (cuBLAS / Metal)
- **Why:** whisper.cpp with cuBLAS on your RTX 4070 would process the 37-min
  video in ~90 seconds (vs. ~5 min with Python).
- **How:** Build whisper.cpp with `-DWHISPER_CUBLAS=1`. Ship pre-built binaries
  in the release workflow (Windows CUDA, Linux CUDA, macOS Metal).
- **Impact:** 3-5x faster than current Python path on same GPU.

### 2.2 — Speculative pipeline overlap
- **Why:** Currently: chunk → transcribe → wait all → stitch → translate.
  Better: start translating chunk 0 while chunk 1 is still transcribing.
- **How:** `--stream-async` already scaffolds this with bounded channels.
  Extend to overlap the stitcher's dedup with ongoing transcription.
- **Impact:** 30-40% wall-clock reduction on long videos.

### 2.3 — Adaptive chunk sizing
- **Why:** Fixed 240s chunks. Some content (silence-heavy intros) could use
  600s chunks (fewer overhead), dense dialogue needs 120s (faster turnaround).
- **How:** After VAD, compute speech density per region. Size chunks
  proportional to speech density. LFAS can learn optimal sizing.
- **Impact:** Better GPU utilization, fewer boundary artifacts.

### 2.4 — Batch-mode pipelining
- **Why:** Processing 100 files sequentially wastes GPU idle time between files.
- **How:** Directory watcher mode. When file N is in translate stage (CPU-bound),
  start transcribing file N+1 (GPU-bound). Double throughput.
- **Impact:** 2x throughput for batch processing workflows.

---

## Phase 3: Algorithm to 10/10 (Paradigm Tier)

### 3.1 — Contextual LFAS
- **Why:** Current LFAS treats arms as independent. But DOOM-QLOCK's arms have
  structure: chunk_duration is ordered, quality profiles form a lattice.
  Contextual bandits exploit this for faster convergence.
- **How:** Implement Kleinberg-Slivkins-Upfal 2008 with Lipschitz assumption
  on the arm space. Context features: speech density, speaker count, noise level.
  Regret improves from O(√(KT)) to O(T^{5/6}).
- **Impact:** Faster adaptation to new content types. Fewer exploration rounds.

### 3.2 — Minimax optimality proof
- **Why:** LFAS achieves sublinear regret. Is it *optimal*? Proving a matching
  lower bound (Ω(√(KT·log T))) would establish that no label-free scheduler
  can do better. This is a paper-level result.
- **How:** Reduction from multi-armed bandits lower bound (Auer et al. 2002)
  + the label-free constraint removes a factor of information.
- **Impact:** Theoretical completeness. Publishable. Knuth tier.

### 3.3 — Rényi-α continuous family
- **Why:** Theorem 4 uses α=½ (Hellinger). Generalizing to arbitrary
  α ∈ (0,1) gives a one-parameter family of bounds. Optimizing α per-arm
  gives the tightest possible bound for each distribution.
- **How:** `phi_renyi(alpha, d_alpha)` — find optimal α via bisection on the
  observed distribution shape (kurtosis hints at which α is tightest).
- **Impact:** Marginal improvement per-sample (~2-5% tighter). Theoretical elegance.

---

## Phase 4: Testing to 100/100

### 4.1 — Real audio E2E fixture
- **Why:** No test exercises the actual whisper → stitch → translate path.
  Current tests mock everything above the SRT parser.
- **How:** 10-second WAV file with known transcript committed to `tests/fixtures/`.
  Integration test runs `sub-zero -i fixture.wav` and asserts output SRT matches
  expected (fuzzy: timing ±0.5s, text Levenshtein > 0.9).
- **Impact:** Catches the exact class of bugs we found today.

### 4.2 — Property-based pipeline tests
- **Why:** Proptest on LFAS/F3 is great. Extend to the stitcher and chunker.
- **How:** Generate random chunk boundaries + random cues. Property: stitch
  output is always monotonic, re-indexed, and deduped.
- **Impact:** Catches edge cases in chunk boundary handling.

### 4.3 — Benchmark regression gate in CI
- **Why:** Performance can silently regress. A 2x regression in the stitcher
  would go unnoticed until someone processes a 4-hour video.
- **How:** `criterion` benchmarks already exist for srt_parse, stitcher, chunker.
  Add a CI job that runs benchmarks and fails if >10% regression vs. baseline.
- **Impact:** Performance is a maintained invariant, not a hope.

### 4.4 — Fuzz the SRT parser
- **Why:** SRT input can be arbitrary user-supplied text. Parser must not panic.
- **How:** `cargo-fuzz` on `parse_srt`. Feed random bytes. Property: never panics,
  returns Ok or Err, no UB (already `forbid(unsafe_code)` but fuzz catches logic bugs).
- **Impact:** Hardened parser. Confidence for open-source contributors fuzzing.

---

## Phase 5: Open-Source to 100/100

### 5.1 — `cargo install sub-zero`
- **Why:** One command install. No git clone needed.
- **How:** Publish to crates.io. Ensure `Cargo.toml` metadata is complete
  (license, description, repository, categories, keywords).
- **Impact:** Discoverability. Ease of adoption.

### 5.2 — Pre-built binaries in GitHub Releases
- **Why:** Most users won't have Rust toolchain.
- **How:** `.github/workflows/release.yml` already exists. Wire it to produce:
  `sub-zero-x86_64-pc-windows-msvc.zip`, `sub-zero-x86_64-unknown-linux-gnu.tar.gz`,
  `sub-zero-aarch64-apple-darwin.tar.gz`. Auto-attach to GitHub Release.
- **Impact:** Download → run. No build step.

### 5.3 — Docker image
- **Why:** Reproducible environment. No dependency issues.
- **How:** `Dockerfile` with CUDA base image + whisper.cpp + sub-zero binary.
  `docker run -v /media:/media ghcr.io/ibvoid/sub-zero -i /media/video.mp4`
- **Impact:** Cloud deployment, CI integration, zero-config.

### 5.4 — Quick Start README section
- **Why:** First thing a contributor sees. Current README lacks a 30-second
  "how to use this" section.
- **How:** 5 lines: install → download model → run → output explained.
- **Impact:** Conversion from visitor to star/contributor.

### 5.5 — Contributor guide with architecture diagram
- **Why:** New contributors don't know where to start.
- **How:** Mermaid diagram of the pipeline stages. "Want to add a new MT backend?
  Look at `src/engine/translate.rs`. Want to add a new quality gate?
  Look at `src/engine/pipeline/learned_gate.rs`."
- **Impact:** PRs from community.

---

## Phase 6: Code Quality to 100/100

### 6.1 — Split pipeline.rs
- **Why:** 1500+ lines. Too many responsibilities in one file.
- **How:** Extract: `pipeline/parallel.rs` (the parallel_transcribe method),
  `pipeline/metadata.rs` (sidecar writing), `pipeline/quality.rs` (gates).
  Keep `pipeline/mod.rs` as the orchestrator (~400 lines).
- **Impact:** Easier navigation, smaller diffs, clearer ownership.

### 6.2 — Remove all `#[allow(dead_code)]`
- **Why:** Dead code is tech debt. Either use it or delete it.
- **How:** Audit each `#[allow(dead_code)]`. If the field/fn is used by benchmarks
  only, gate it behind `#[cfg(feature = "bench-internals")]`. Otherwise delete.
- **Impact:** Cleaner codebase, smaller binary.

### 6.3 — Error type consolidation
- **Why:** Mix of `String` errors and typed errors. Inconsistent.
- **How:** All internal functions return typed errors. Only the CLI boundary
  converts to string for display. Use `thiserror` everywhere (already a dep).
- **Impact:** Better diagnostics, pattern-matchable errors for callers.

---

## Priority Order (Next 5 Actions)

1. **Verify E2E test passes** (run the video, confirm full output)
2. **1.1 whisper.cpp as primary** (biggest bang for buck)
3. **4.1 Real audio E2E fixture** (prevent regression)
4. **5.4 Quick Start README** (open-source readiness)
5. **3.1 Contextual LFAS** (push theory to 10/10)

---

*Last updated: 2026-05-25*
