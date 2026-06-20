//! VoiDex — offline-first subtitle translation engine.
//!
//! VoiDex turns a local media file or `.srt` into a translated SRT with
//! auditable quality metadata, without leaving the machine. The binary at
//! `src/main.rs` is a thin CLI wrapper around this library.
//!
//! # Layout
//!
//! - [`engine`] — the pipeline state machine, transcription and translation
//!   boundaries, the adaptive planner, the learned quality gate, and the
//!   local event sidecars.
//!
//! See `PROJECT_CHARTER.md` and `ARCHITECTURE.md` at the repository root for
//! the project's mission, invariants, and module ownership.
//!
//! # Stability
//!
//! This is a `0.1.x` release. Public types may evolve. Configuration structs
//! are intentionally `#[non_exhaustive]` where added — construct them via
//! their builders or `Default` impl rather than literal initializers.
//!
//! # Privacy contract
//!
//! Nothing in this library performs cloud calls. The `--offline` CLI flag is
//! a user-facing privacy contract enforced at the binary boundary; the
//! library itself never reaches the network except through the explicit
//! local event sidecars in [`engine::http_sidecar`] and [`engine::ws_sidecar`]
//! (loopback-only by default).

#![forbid(unsafe_code)]
#![warn(rust_2018_idioms)]

pub mod engine;

/// One-time, best-effort migration of the legacy `~/.sub-zero` state
/// directory to the rebranded `~/.voidex` location.
///
/// Historically VoiDex shipped as "Sub-Zero" and kept its checkpoints,
/// character glossaries, and DOOM-QLOCK history under `~/.sub-zero`. After
/// the rebrand the canonical location is `~/.voidex` (see
/// [`engine::pipeline`] path resolution). Existing users would otherwise
/// silently lose in-progress checkpoints, so the binary calls this once at
/// startup to move the directory across.
///
/// Behaviour:
/// - If `VOIDEX_HOME` is set the user owns the location explicitly; nothing
///   is moved (the corresponding legacy override was `SUB_ZERO_HOME`, which
///   the rebrand intentionally retired — see the env-var clean break).
/// - The default location is resolved with the same precedence the engine
///   uses: `HOME`, then `USERPROFILE`, then the system temp dir.
/// - The move only happens when the legacy dir exists and the new dir does
///   not, so it is idempotent and never clobbers existing `~/.voidex` state.
/// - Any I/O error is swallowed: a failed migration must never block
///   startup — the engine simply recreates state under the new path.
pub fn migrate_legacy_home() {
    use std::path::PathBuf;

    // Explicit override: the user controls the path, leave it untouched.
    if std::env::var_os("VOIDEX_HOME").is_some() {
        return;
    }
    let parent = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);

    let legacy = parent.join(".sub-zero");
    let current = parent.join(".voidex");
    if legacy.is_dir() && !current.exists() {
        let _ = std::fs::rename(&legacy, &current);
    }
}

// Most commonly used types are re-exported at the crate root so downstream
// callers can `use voidex::PipelineConfig` instead of reaching deep into
// the module tree. Deeper modules remain available under `voidex::engine`.
pub use engine::doom_qlock::DoomQlock;
pub use engine::f3_stream::F3StreamEstimator;
pub use engine::http_sidecar::{start_http_sidecar, HttpSidecarConfig};
pub use engine::lfas::{
    phi, phi_hellinger, phi_sharpened, sharpening_gap, ArmId, F3Sample, LfasConfig, LfasScheduler,
};
pub use engine::pipeline::{PipelineConfig, SubtitlePipeline};
pub use engine::transcribe::QualityProfile;
pub use engine::ws_sidecar::{start_ws_sidecar, WsSidecarConfig};

/// `#[doc(hidden)]` — exposed only so `benches/` can reach the boundary
/// selector and the dedup pass directly. Not part of the public API
/// contract; nothing here is covered by semver. Use the higher-level
/// engine entry points (`PipelineConfig`, `SubtitlePipeline`) in normal
/// downstream code.
///
/// Gated behind the `bench-internals` feature so the published crate
/// does not ship the cached `parse_srt_cached` variant in its public
/// surface. Phase A measurements at
/// `docs/adr/0001_phase_a_measurements.md` showed `parse_srt_cached`
/// is measurably slower than uncached `parse_srt` on cheap functions
/// — a perpetual trap for downstream users if left in the public API.
#[doc(hidden)]
#[cfg(feature = "bench-internals")]
pub mod bench_internals {
    pub use crate::engine::cache::{ContentCache, FsContentCache, MemoryContentCache};
    pub use crate::engine::chunker::{pick_boundaries, SilenceGap};
    pub use crate::engine::srt::{parse_srt, parse_srt_cached, PARSE_SRT_CACHE_ID};
    pub use crate::engine::stitcher::{deduplicate_overlaps, TimedCue};
}
