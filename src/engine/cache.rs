//! Content-addressed cache — ADR-0001 Phase A pilot.
//!
//! This module is the proof-of-concept for the proposed structural
//! reinvention of the engine as a content-addressed demand-driven
//! build graph (see `docs/adr/0001-content-addressed-build-graph.md`).
//! It is *deliberately small*: a single trait, one filesystem-backed
//! implementation, an in-memory implementation for tests, and wiring
//! to exactly one pure function (`srt::parse_srt_cached`). The point
//! is to measure cache-hit vs cache-miss overhead on a real workload
//! before committing to the 3–6 month Phase A–E plan.
//!
//! ## What "content-addressed" means here
//!
//! A node is `(function_id, input_hash) -> output_hash` where:
//! - `function_id` is a string naming the function (e.g.
//!   `"srt::parse_srt:v1"`). Versioning the id lets us invalidate
//!   the cache when the function's semantics change.
//! - `input_hash` is `sha256(canonical_input_bytes)`.
//! - `output_hash` is `sha256(output_bytes)`, where the output is
//!   serialised deterministically.
//!
//! The cache stores `(function_id, input_hash) -> output_bytes`. A
//! hit reconstructs the output by deserialising the stored bytes; a
//! miss runs the function, serialises the output, and stores it.
//!
//! ## What this prototype does *not* do (intentionally)
//!
//! - No incremental recomputation graph (Salsa-style). That's Phase B+.
//! - No cross-user cache. Local-only by construction.
//! - No eviction policy. The cache grows; the prototype is for
//!   measurement, not deployment.
//! - No determinism enforcement on the cached function. We rely on
//!   `srt::parse_srt` being already deterministic.
//!
//! ## What this prototype *does* prove
//!
//! - The content-addressing primitive is tractable in idiomatic Rust
//!   without pulling in a heavy framework dep.
//! - Cache-hit overhead is measurable and small compared to recompute.
//! - The architect's hypothesis — *that re-runs with shared upstream
//!   would benefit from caching* — is testable on real workloads via
//!   the bench at `benches/bench_engine.rs`.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

/// 32-byte SHA-256 digest; stored hex in keys for human inspection.
pub type Digest32 = [u8; 32];

/// Public byte-level SHA-256. Exposed under `pub` (not `pub(crate)`)
/// because the cache's content-addressing primitive is meant to be
/// reused by future graph-node implementations; without this helper
/// every caller would import `sha2` directly.
pub fn hash_bytes(input: &[u8]) -> Digest32 {
    let mut h = Sha256::new();
    h.update(input);
    h.finalize().into()
}

pub(crate) fn hash_with_id(function_id: &str, input: &[u8]) -> Digest32 {
    // Domain-separate by function id so two functions with the same
    // input bytes never collide. The id should be bumped whenever the
    // function's semantics change (think `"srt::parse_srt:v1"` vs
    // `"srt::parse_srt:v2"`).
    let mut h = Sha256::new();
    h.update((function_id.len() as u64).to_le_bytes());
    h.update(function_id.as_bytes());
    h.update(input);
    h.finalize().into()
}

pub(crate) fn digest_hex(d: &Digest32) -> String {
    let mut out = String::with_capacity(64);
    for b in d {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Read/write a serialised value into a content-addressed slot. The
/// trait is intentionally byte-shaped so each cached function can
/// choose its own serialisation (bincode, postcard, serde_json — the
/// `srt` pilot uses serde_json for human-readable diffing).
pub trait ContentCache: Send + Sync {
    fn get(&self, key: &Digest32) -> Option<Vec<u8>>;
    fn put(&self, key: &Digest32, value: &[u8]);
}

/// In-memory cache for unit tests + benchmarks. Thread-safe.
#[derive(Default)]
pub struct MemoryContentCache {
    inner: Mutex<HashMap<Digest32, Vec<u8>>>,
}

impl MemoryContentCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ContentCache for MemoryContentCache {
    fn get(&self, key: &Digest32) -> Option<Vec<u8>> {
        self.inner.lock().ok()?.get(key).cloned()
    }
    fn put(&self, key: &Digest32, value: &[u8]) {
        if let Ok(mut g) = self.inner.lock() {
            g.insert(*key, value.to_vec());
        }
    }
}

/// Filesystem-backed cache. Layout: `root/ab/abcd…ef.bin` where
/// `abcd…ef` is the lowercase hex digest. The two-character prefix
/// keeps any single directory's fan-out bounded.
pub struct FsContentCache {
    root: PathBuf,
}

impl FsContentCache {
    pub fn new<P: Into<PathBuf>>(root: P) -> std::io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    fn path_for(&self, key: &Digest32) -> PathBuf {
        let hex = digest_hex(key);
        let (prefix, rest) = hex.split_at(2);
        let mut p = self.root.clone();
        p.push(prefix);
        p.push(format!("{rest}.bin"));
        p
    }
}

impl ContentCache for FsContentCache {
    fn get(&self, key: &Digest32) -> Option<Vec<u8>> {
        fs::read(self.path_for(key)).ok()
    }
    fn put(&self, key: &Digest32, value: &[u8]) {
        let path = self.path_for(key);
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        // Best-effort; cache writes are advisory.
        let _ = fs::write(path, value);
    }
}

/// Drop-in wrapper for any pure `(input bytes) -> output bytes`
/// function. The caller supplies a function-id-versioned namespace
/// so that two unrelated functions with the same input bytes never
/// collide. The closure runs only on cache miss.
pub fn cached<F>(cache: &dyn ContentCache, function_id: &str, input: &[u8], compute: F) -> Vec<u8>
where
    F: FnOnce() -> Vec<u8>,
{
    let key = hash_with_id(function_id, input);
    if let Some(bytes) = cache.get(&key) {
        return bytes;
    }
    let value = compute();
    cache.put(&key, &value);
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_cache_round_trip() {
        let cache = MemoryContentCache::new();
        let key = hash_bytes(b"hello");
        assert!(cache.get(&key).is_none());
        cache.put(&key, b"world");
        assert_eq!(cache.get(&key).as_deref(), Some(&b"world"[..]));
    }

    #[test]
    fn function_id_namespaces_collisions() {
        // Two different function ids on the same input bytes must
        // produce distinct keys — otherwise a cache shared between
        // two functions would conflate their outputs.
        let a = hash_with_id("foo:v1", b"input");
        let b = hash_with_id("foo:v2", b"input");
        let c = hash_with_id("bar:v1", b"input");
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn cached_runs_compute_only_on_miss() {
        // `cached` takes `FnOnce` (single-shot closure) so the
        // "did compute run?" counter has to live in shared state
        // rather than be captured by mutable reference. An atomic is
        // overkill semantically but keeps the test free of `RefCell`
        // ceremony.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let cache = MemoryContentCache::new();
        let compute_calls = AtomicUsize::new(0);
        let id = "test::echo:v1";

        let make_closure = || {
            // A fresh FnOnce per call; both close over the same atomic.
            || {
                compute_calls.fetch_add(1, Ordering::SeqCst);
                b"computed".to_vec()
            }
        };

        // First call: miss → compute runs.
        let out1 = cached(&cache, id, b"input", make_closure());
        assert_eq!(out1, b"computed");
        assert_eq!(compute_calls.load(Ordering::SeqCst), 1);

        // Second call: hit → compute does NOT run.
        let out2 = cached(&cache, id, b"input", make_closure());
        assert_eq!(out2, b"computed");
        assert_eq!(
            compute_calls.load(Ordering::SeqCst),
            1,
            "compute must not be called on cache hit"
        );
    }

    #[test]
    fn fs_cache_persists_across_handles() {
        let dir = std::env::temp_dir().join(format!(
            "sub_zero_cache_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // First handle: write.
        {
            let cache = FsContentCache::new(&dir).expect("new cache");
            let key = hash_bytes(b"persistent");
            cache.put(&key, b"value");
            assert_eq!(cache.get(&key).as_deref(), Some(&b"value"[..]));
        }
        // Second handle: read what the first wrote.
        {
            let cache = FsContentCache::new(&dir).expect("reopen cache");
            let key = hash_bytes(b"persistent");
            assert_eq!(cache.get(&key).as_deref(), Some(&b"value"[..]));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn digest_hex_round_trip_is_lowercase_64_chars() {
        let d = hash_bytes(b"foo");
        let hex = digest_hex(&d);
        assert_eq!(hex.len(), 64);
        assert!(hex
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }
}
