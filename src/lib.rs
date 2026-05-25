//! Sub-Zero — offline-first subtitle translation engine.
//!
//! Sub-Zero turns a local media file or `.srt` into a translated SRT with
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

// Most commonly used types are re-exported at the crate root so downstream
// callers can `use sub_zero::PipelineConfig` instead of reaching deep into
// the module tree. Deeper modules remain available under `sub_zero::engine`.
pub use engine::doom_qlock::DoomQlock;
pub use engine::f3_stream::F3StreamEstimator;
pub use engine::http_sidecar::{start_http_sidecar, HttpSidecarConfig};
pub use engine::lfas::{
    ArmId, F3Sample, LfasConfig, LfasScheduler,
    phi, phi_hellinger, phi_sharpened, sharpening_gap,
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
