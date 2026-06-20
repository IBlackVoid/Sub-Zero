//! Criterion microbenches for the hot engine paths.
//!
//! Targets: `srt::parse_srt`, `stitcher::deduplicate_overlaps`, and
//! `chunker::pick_boundaries`. These are reached via the
//! `bench_internals` shim in `lib.rs` so the regular public API can stay
//! tight while `cargo bench` still gets at the primitives.
//!
//! Run: `cargo bench` — output lands under `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use voidex::bench_internals::{
    deduplicate_overlaps, parse_srt, parse_srt_cached, pick_boundaries, MemoryContentCache,
    SilenceGap, TimedCue,
};

/// Build a representative SRT body of `cue_count` cues. Long enough to
/// exercise the inner loop without exploding the bench wall-clock.
fn build_srt(cue_count: usize) -> String {
    let mut out = String::with_capacity(cue_count * 80);
    for i in 0..cue_count {
        let start_ms = (i as u64) * 2_000;
        let end_ms = start_ms + 1_500;
        let (sh, sm, ss, smi) = ms_breakdown(start_ms);
        let (eh, em, es, emi) = ms_breakdown(end_ms);
        out.push_str(&format!(
            "{}\n{sh:02}:{sm:02}:{ss:02},{smi:03} --> {eh:02}:{em:02}:{es:02},{emi:03}\nline {i}\nextra line {i}\n\n",
            i + 1
        ));
    }
    out
}

fn ms_breakdown(ms: u64) -> (u64, u64, u64, u64) {
    let millis = ms % 1000;
    let total_s = ms / 1000;
    let s = total_s % 60;
    let m = (total_s / 60) % 60;
    let h = total_s / 3600;
    (h, m, s, millis)
}

fn bench_parse_srt(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_srt");
    for &cue_count in &[100usize, 1_000, 10_000] {
        let input = build_srt(cue_count);
        group.throughput(criterion::Throughput::Bytes(input.len() as u64));
        group.bench_function(format!("cues={cue_count}"), |b| {
            b.iter(|| {
                let cues = parse_srt(black_box(&input)).expect("bench input must parse");
                black_box(cues);
            });
        });
    }
    group.finish();
}

fn bench_deduplicate_overlaps(c: &mut Criterion) {
    let mut group = c.benchmark_group("deduplicate_overlaps");
    for &cue_count in &[100usize, 1_000, 10_000] {
        // Half the cues are duplicates of their predecessor to exercise
        // the dedup branch on every other input.
        let cues: Vec<TimedCue> = (0..cue_count)
            .map(|i| {
                let base = (i / 2) as f64;
                TimedCue {
                    abs_start: base * 2.0,
                    abs_end: base * 2.0 + 1.5,
                    text: format!("line {}", i / 2),
                    chunk_index: i / 32,
                }
            })
            .collect();
        group.bench_function(format!("cues={cue_count}"), |b| {
            b.iter_batched(
                || cues.clone(),
                |c| {
                    let out = deduplicate_overlaps(black_box(&c));
                    black_box(out);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_pick_boundaries(c: &mut Criterion) {
    let mut group = c.benchmark_group("pick_boundaries");
    for &(duration_secs, gap_count) in &[(900.0_f64, 0usize), (3_600.0, 50), (7_200.0, 200)] {
        let gaps: Vec<SilenceGap> = (0..gap_count)
            .map(|i| {
                let centroid = (duration_secs / (gap_count as f64 + 1.0)) * (i as f64 + 1.0);
                SilenceGap {
                    start: centroid - 0.25,
                    end: centroid + 0.25,
                    centroid,
                }
            })
            .collect();
        group.bench_function(format!("duration={duration_secs}s,gaps={gap_count}"), |b| {
            b.iter(|| {
                let bounds =
                    pick_boundaries(black_box(duration_secs), black_box(&gaps), black_box(300.0));
                black_box(bounds);
            });
        });
    }
    group.finish();
}

// ADR-0001 Phase A pilot — measure the cache-hit vs cache-miss
// overhead of content-addressed parse_srt. The architect's hypothesis
// is that a hit is materially faster than a recompute; this bench
// gives the maintainer the numbers to validate it before committing
// 3-6 months to the full build graph.
fn bench_parse_srt_cached(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_srt_cached");
    for &cue_count in &[100usize, 1_000, 10_000] {
        let input = build_srt(cue_count);
        // Reference: uncached parse_srt — the "current engine" number.
        group.bench_function(format!("uncached_cues={cue_count}"), |b| {
            b.iter(|| {
                let cues = parse_srt(black_box(&input)).expect("bench input must parse");
                black_box(cues);
            });
        });
        // Cache MISS — parse_srt + serialize + cache write. This is the
        // *worst-case* cost the content-addressed mode pays vs uncached.
        // Each iter gets a fresh cache so every call is a miss.
        group.bench_function(format!("miss_cues={cue_count}"), |b| {
            b.iter_batched(
                MemoryContentCache::new,
                |cache| {
                    let cues = parse_srt_cached(black_box(&input), &cache)
                        .expect("bench input must parse");
                    black_box(cues);
                },
                BatchSize::SmallInput,
            );
        });
        // Cache HIT — every iter reads from the same warm cache. This
        // is the *steady-state* cost the content-addressed mode pays
        // on re-runs with identical input. The win vs uncached parse
        // is the headline number ADR-0001 needs to justify the move.
        let warm_cache = MemoryContentCache::new();
        // Prime the cache once.
        let _ = parse_srt_cached(&input, &warm_cache).expect("priming");
        group.bench_function(format!("hit_cues={cue_count}"), |b| {
            b.iter(|| {
                let cues =
                    parse_srt_cached(black_box(&input), &warm_cache).expect("warm-cache parse");
                black_box(cues);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_parse_srt,
    bench_deduplicate_overlaps,
    bench_pick_boundaries,
    bench_parse_srt_cached,
);
criterion_main!(benches);
