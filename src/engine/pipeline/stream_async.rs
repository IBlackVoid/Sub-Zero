use crate::engine::chunker::{chunk_audio, ChunkerConfig};
use crate::engine::events::EventSink;
use crate::engine::live_replanner::LiveHistogramReplanner;
use crate::engine::srt::{append_srt_file, parse_srt_file, SubtitleCue};
use crate::engine::transcribe::{
    detect_speech_intervals_from_wav, ffprobe_duration_seconds_pub, Transcriber,
};
use crossbeam_channel::{bounded, Receiver};
use serde_json::json;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};
use std::time::{Duration, Instant};

use super::audio::extract_audio_to_wav_with_selection;
use super::paths::checkpoint_dir_for;
use super::speaker::{build_speaker_tags_with_context, infer_speakers};
use super::time::{format_srt_timing_line, parse_srt_timing_line};
use super::util::now_epoch_secs;
use super::{PipelineConfig, SubtitlePipeline};

const STREAM_DEDUP_SIMILARITY_THRESHOLD: f64 = 0.6;

#[derive(Debug, Clone)]
struct LastCueFingerprint {
    abs_start: f64,
    text: String,
}

#[derive(Debug, Clone)]
struct ChunkJob {
    index: usize,
    start_sec: f64,
    end_sec: f64,
    wav_path: PathBuf,
    chunk_len_secs: f64,
    base_timeout_secs: f64,
}

#[derive(Debug, Clone)]
struct ChunkAsr {
    index: usize,
    start_sec: f64,
    cues: Vec<SubtitleCue>,
    extra_tags: Vec<Vec<String>>,
    elapsed_secs: f64,
}

#[derive(Debug, Clone)]
struct ChunkMt {
    index: usize,
    cues: Vec<SubtitleCue>,
    asr_elapsed_secs: f64,
    mt_elapsed_secs: f64,
}

pub(super) struct StreamAsyncRunResult {
    pub(super) audio_wav_path: PathBuf,
}

// AsrSemaphore — a slot counter whose ceiling is dynamic. The
// `LiveHistogramReplanner` may raise or lower `asr_limit()` at any time
// in response to p95 telemetry; we cannot statically size a `Semaphore`.
//
// Threads call `acquire` with a closure that resolves the current limit;
// on release we `notify_one`. When all slots are full, the waiter parks
// on the condvar and is woken either by a release (immediate) or by a
// 50 ms timeout (to re-evaluate the dynamic limit, since the replanner
// itself does not notify this semaphore).
//
// This replaces the previous busy-wait `acquire_asr_slot` that polled
// `replanner.asr_limit()` with a 10 ms sleep, burning a thread.
#[derive(Debug, Default)]
struct AsrSemaphore {
    state: Mutex<usize>,
    cvar: Condvar,
}

impl AsrSemaphore {
    fn new() -> Self {
        Self::default()
    }

    fn acquire<'sem, F>(&'sem self, get_limit: F, cancel: &AtomicBool) -> Option<AsrPermit<'sem>>
    where
        F: Fn() -> usize,
    {
        let mut count = self.state.lock().ok()?;
        loop {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let limit = get_limit().max(1);
            if *count < limit {
                *count += 1;
                return Some(AsrPermit {
                    sem: self,
                    _not_send: PhantomData,
                });
            }
            // Park until a release or the timeout, then re-check the limit
            // (the replanner may have raised it while we waited).
            let waited = self
                .cvar
                .wait_timeout(count, Duration::from_millis(50))
                .ok()?;
            count = waited.0;
        }
    }
}

/// Compile-time proof that the bearer holds an ASR slot.
///
/// `AsrPermit<'sem>` is constructed *only* by `AsrSemaphore::acquire`
/// (the fields are private). Functions that perform ASR work require
/// `&AsrPermit<'_>` as an argument — see `transcribe_with_permit`
/// below — so the type system refuses any path that tries to start an
/// ASR job without having first taken a slot from the semaphore.
///
/// The permit is **`!Send`** (`PhantomData<*const ()>`): a slot
/// belongs to the worker thread that took it, because the slot is
/// released by `Drop` and that drop must run on the thread that owns
/// the worker's local state. Shipping a permit across threads would
/// silently misroute the release and break the slot accounting.
///
/// `#[must_use]` makes a permit that is never used a compile-time
/// warning — historically the ASR loop wrote `let Some(_slot) = ...`
/// which is *almost* the right shape but leaks the affordance: nothing
/// stops a caller from removing the binding entirely. The permit
/// makes that path impossible to write.
#[must_use = "AsrPermit must be held for the duration of an ASR job; dropping it without doing the work leaks a semaphore slot"]
struct AsrPermit<'sem> {
    sem: &'sem AsrSemaphore,
    _not_send: PhantomData<*const ()>,
}

impl Drop for AsrPermit<'_> {
    fn drop(&mut self) {
        if let Ok(mut count) = self.sem.state.lock() {
            *count = count.saturating_sub(1);
            self.sem.cvar.notify_one();
        }
    }
}

/// Run ASR transcription, requiring at the type level that the caller
/// holds an [`AsrPermit`]. The permit is taken by reference (no
/// transfer of ownership) so the worker's `Drop` semantics fire when
/// the worker scope ends, not when this function returns. The permit
/// is bound but unused inside the function — its purpose is purely to
/// require its existence at the call site.
fn transcribe_with_permit(
    _permit: &AsrPermit<'_>,
    transcriber: &Transcriber,
    wav_path: &Path,
    timeout_secs: f64,
) -> Result<PathBuf, String> {
    transcriber
        .transcribe_wav_to_srt_with_timeout(wav_path, timeout_secs)
        .map_err(|e| e.to_string())
}

pub(super) fn stream_transcribe_translate_video_async(
    pipeline: &SubtitlePipeline,
    video: &Path,
    output_srt: &Path,
    events: &EventSink,
) -> Result<StreamAsyncRunResult, String> {
    if pipeline.transcriber.is_none() {
        return Err("stream-async mode requires transcription to be enabled".to_string());
    }
    let config = &pipeline.config;

    let checkpoint_dir = checkpoint_dir_for(video)?;
    let temp_dir = checkpoint_dir.join("work_stream_async");
    std::fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;

    std::fs::File::create(output_srt).map_err(|e| format!("{}: {e}", output_srt.display()))?;

    let wav_path = temp_dir.join("audio.wav");
    extract_audio_to_wav_with_selection(
        video,
        &wav_path,
        config.audio_stream_index,
        config.audio_lang.as_deref(),
    )?;
    let duration = ffprobe_duration_seconds_pub(&wav_path).map_err(|e| e.to_string())?;
    if events.enabled() {
        let vad_started = Instant::now();
        let speech = detect_speech_intervals_from_wav(
            &wav_path,
            config.vad_threshold_db,
            config.vad_min_silence,
            config.vad_pad,
            Some(duration),
        )?;
        for (idx, seg) in speech.iter().enumerate() {
            events.emit(&json!({
                "event": "vad_segment",
                "input": video.display().to_string(),
                "segment_index": idx,
                "segment_total": speech.len(),
                "start_sec": seg.start,
                "end_sec": seg.end,
                "emitted_at_epoch_secs": now_epoch_secs(),
            }));
        }
        events.emit(&json!({
            "event": "vad_complete",
            "input": video.display().to_string(),
            "segment_total": speech.len(),
            "elapsed_secs": vad_started.elapsed().as_secs_f64(),
            "emitted_at_epoch_secs": now_epoch_secs(),
        }));
    }

    let chunker_config = ChunkerConfig {
        target_chunk_secs: config.chunk_duration_secs,
        min_silence_gap: 0.4,
        overlap_secs: 2.0,
        vad_threshold_db: config.vad_threshold_db,
        vad_min_silence: config.vad_min_silence,
        vad_pad: config.vad_pad,
    };
    let mut chunks = chunk_audio(&wav_path, &temp_dir, duration, &chunker_config)?;
    chunks.sort_by_key(|c| c.index);

    let chunk_total = chunks.len();
    let worker_count = config.max_workers.clamp(1, chunk_total.max(1));
    let speaker_aware = config.speaker_aware;
    let jobs_cap = (worker_count * 2).max(2);
    // `crossbeam_channel::bounded` is natively MPMC — workers clone the
    // receiver directly, no `Arc<Mutex<Receiver>>` shim.
    let (jobs_tx, jobs_rx) = bounded::<ChunkJob>(jobs_cap);
    let (asr_tx, asr_rx) = bounded::<Result<ChunkAsr, String>>(worker_count);
    let (mt_tx, mt_rx) = bounded::<Result<ChunkMt, String>>(worker_count);

    let cancel = Arc::new(AtomicBool::new(false));
    let replanner = Arc::new(LiveHistogramReplanner::new(worker_count));
    let asr_sem = Arc::new(AsrSemaphore::new());
    let cfg_for_workers = SubtitlePipeline::make_transcribe_config(&pipeline.config, true);

    for _ in 0..worker_count {
        let rx: Receiver<ChunkJob> = jobs_rx.clone();
        let asr_tx = asr_tx.clone();
        let cancel = cancel.clone();
        let events = events.clone();
        let replanner = replanner.clone();
        let asr_sem = asr_sem.clone();
        let cfg = cfg_for_workers.clone();
        let video = video.to_path_buf();
        std::thread::spawn(move || {
            let transcriber = match Transcriber::new(cfg).map_err(|e| e.to_string()) {
                Ok(Some(t)) => t,
                Ok(None) => {
                    let _ = asr_tx.send(Err("transcriber disabled explicitly".to_string()));
                    return;
                }
                Err(e) => {
                    let _ = asr_tx.send(Err(format!("failed to init transcriber: {e}")));
                    return;
                }
            };

            loop {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                // crossbeam `recv` blocks without holding any global lock;
                // it returns Err once the sender side has been dropped, at
                // which point the worker exits cleanly.
                let Ok(job) = rx.recv() else { return };

                // Take the typed permit: the compiler will refuse the
                // call to `transcribe_with_permit` below without one.
                let Some(permit) = asr_sem.acquire(|| replanner.asr_limit(), &cancel) else {
                    return;
                };

                let started = Instant::now();
                if events.enabled() {
                    events.emit(&json!({
                        "event": "chunk_started",
                        "input": video.display().to_string(),
                        "chunk_index": job.index,
                        "chunk_total": chunk_total,
                        "start_sec": job.start_sec,
                        "end_sec": job.end_sec,
                        "emitted_at_epoch_secs": now_epoch_secs(),
                    }));
                }

                let timeout_secs =
                    (job.base_timeout_secs * replanner.timeout_scale()).clamp(10.0, 7200.0);
                let srt_path = match transcribe_with_permit(
                    &permit,
                    &transcriber,
                    &job.wav_path,
                    timeout_secs,
                ) {
                    Ok(path) => path,
                    Err(e) => {
                        cancel.store(true, Ordering::Relaxed);
                        let _ = asr_tx.send(Err(e));
                        return;
                    }
                };
                let cues = match parse_srt_file(&srt_path).map_err(|e| e.to_string()) {
                    Ok(c) => c,
                    Err(e) => {
                        cancel.store(true, Ordering::Relaxed);
                        let _ = asr_tx.send(Err(e));
                        return;
                    }
                };

                let extra_tags = if speaker_aware {
                    let (speakers, _) = infer_speakers(&cues);
                    build_speaker_tags_with_context(&cues, &speakers)
                } else {
                    Vec::new()
                };

                let elapsed = started.elapsed().as_secs_f64();
                if events.enabled() {
                    events.emit(&json!({
                        "event": "asr_complete",
                        "input": video.display().to_string(),
                        "chunk_index": job.index,
                        "chunk_total": chunk_total,
                        "cue_count": cues.len(),
                        "elapsed_secs": elapsed,
                        "timeout_secs": timeout_secs,
                        "emitted_at_epoch_secs": now_epoch_secs(),
                    }));
                }

                if let Some(decision) = replanner.note_asr_sample(job.chunk_len_secs, elapsed) {
                    if events.enabled() {
                        events.emit(&json!({
                            "event": "replan",
                            "input": video.display().to_string(),
                            "kind": "asr_concurrency_timeout",
                            "asr_limit": decision.new_asr_limit,
                            "timeout_scale": decision.new_timeout_scale,
                            "asr_p95_secs_per_audio": decision.asr_p95_secs_per_audio,
                            "emitted_at_epoch_secs": now_epoch_secs(),
                        }));
                    }
                }

                let _ = asr_tx.send(Ok(ChunkAsr {
                    index: job.index,
                    start_sec: job.start_sec,
                    cues,
                    extra_tags,
                    elapsed_secs: elapsed,
                }));
            }
        });
    }
    drop(asr_tx);

    let translator = pipeline.translator.clone();
    let video_for_mt = video.to_path_buf();
    let events_for_mt = events.clone();
    std::thread::spawn(move || {
        while let Ok(msg) = asr_rx.recv() {
            let asr = match msg {
                Ok(asr) => asr,
                Err(e) => {
                    let _ = mt_tx.send(Err(e));
                    return;
                }
            };
            let started = Instant::now();
            let translated =
                match translator.translate_all_with_extra_tags(&asr.cues, &asr.extra_tags) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = mt_tx.send(Err(e.to_string()));
                        return;
                    }
                };
            let mut shifted = Vec::with_capacity(translated.len());
            for cue in translated {
                let (start, end) = match parse_srt_timing_line(&cue.timing) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = mt_tx.send(Err(e));
                        return;
                    }
                };
                shifted.push(SubtitleCue {
                    index: cue.index,
                    timing: format_srt_timing_line(start + asr.start_sec, end + asr.start_sec),
                    text: cue.text,
                });
            }
            let mt_elapsed = started.elapsed().as_secs_f64();
            if events_for_mt.enabled() {
                events_for_mt.emit(&json!({
                    "event": "mt_complete",
                    "input": video_for_mt.display().to_string(),
                    "chunk_index": asr.index,
                    "chunk_total": chunk_total,
                    "cue_count": shifted.len(),
                    "elapsed_secs": mt_elapsed,
                    "emitted_at_epoch_secs": now_epoch_secs(),
                }));
            }

            let _ = mt_tx.send(Ok(ChunkMt {
                index: asr.index,
                cues: shifted,
                asr_elapsed_secs: asr.elapsed_secs,
                mt_elapsed_secs: mt_elapsed,
            }));
        }
    });

    // Feed jobs.
    for chunk in chunks {
        let chunk_len_secs = (chunk.end_sec - chunk.start_sec).max(0.0);
        let base_timeout_secs = stream_chunk_timeout_secs(config, chunk_len_secs);
        jobs_tx
            .send(ChunkJob {
                index: chunk.index,
                start_sec: chunk.start_sec,
                end_sec: chunk.end_sec,
                wav_path: chunk.wav_path.clone(),
                chunk_len_secs,
                base_timeout_secs,
            })
            .map_err(|_| "stream-async worker pool stopped unexpectedly".to_string())?;
    }
    drop(jobs_tx);

    // Writer (ordered): buffer out-of-order MT results then append sequentially.
    let mut buffer = BTreeMap::<usize, ChunkMt>::new();
    let mut expected = 0usize;
    let mut next_index = 1usize;
    let mut last_cue = Option::<LastCueFingerprint>::None;

    while let Ok(msg) = mt_rx.recv() {
        let mt = msg?;
        buffer.insert(mt.index, mt);
        while let Some(mt) = buffer.remove(&expected) {
            let write_started = Instant::now();
            let mut cues = mt.cues;

            // Boundary dedup against prior chunk.
            if let Some(first) = cues.first() {
                if let Some(fingerprint) = last_cue.as_ref() {
                    let (start, _) = parse_srt_timing_line(&first.timing)?;
                    let time_gap = (start - fingerprint.abs_start).abs();
                    if time_gap < 3.0 {
                        let sim = strsim::normalized_levenshtein(&fingerprint.text, &first.text);
                        if sim >= STREAM_DEDUP_SIMILARITY_THRESHOLD {
                            cues.remove(0);
                        }
                    }
                }
            }

            if !cues.is_empty() {
                next_index =
                    append_srt_file(output_srt, &cues, next_index).map_err(|e| e.to_string())?;
                if let Some(last) = cues.last() {
                    let (start, _) = parse_srt_timing_line(&last.timing)?;
                    last_cue = Some(LastCueFingerprint {
                        abs_start: start,
                        text: last.text.clone(),
                    });
                }
            }

            if events.enabled() {
                events.emit(&json!({
                    "event": "chunk_complete",
                    "input": video.display().to_string(),
                    "chunk_index": mt.index,
                    "chunk_total": chunk_total,
                    "asr_elapsed_secs": mt.asr_elapsed_secs,
                    "mt_elapsed_secs": mt.mt_elapsed_secs,
                    "write_elapsed_secs": write_started.elapsed().as_secs_f64(),
                    "emitted_at_epoch_secs": now_epoch_secs(),
                }));
            }

            expected += 1;
        }
    }

    // Ensure the output parses (also catches partial writes).
    let _ = parse_srt_file(output_srt).map_err(|e| e.to_string())?;

    Ok(StreamAsyncRunResult {
        audio_wav_path: wav_path,
    })
}

fn stream_chunk_timeout_secs(config: &PipelineConfig, chunk_len_secs: f64) -> f64 {
    match config.quality_profile {
        crate::engine::transcribe::QualityProfile::Fast => {
            (chunk_len_secs * 4.0 + 30.0).clamp(45.0, 900.0)
        }
        crate::engine::transcribe::QualityProfile::Balanced => {
            (chunk_len_secs * 6.0 + 60.0).clamp(90.0, 1800.0)
        }
        crate::engine::transcribe::QualityProfile::Strict => {
            (chunk_len_secs * 8.0 + 90.0).clamp(120.0, 2400.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AsrSemaphore;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn adaptive_semaphore_respects_dynamic_limit() {
        let sem = Arc::new(AsrSemaphore::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let limit = Arc::new(AtomicUsize::new(2));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..6 {
            let sem = sem.clone();
            let cancel = cancel.clone();
            let limit = limit.clone();
            let active = active.clone();
            let peak = peak.clone();
            handles.push(thread::spawn(move || {
                let _g = sem
                    .acquire(|| limit.load(Ordering::Relaxed), &cancel)
                    .expect("acquire");
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(40));
                active.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // Concurrency must never exceed the configured limit.
        assert!(peak.load(Ordering::SeqCst) <= 2);
    }

    #[test]
    fn adaptive_semaphore_yields_to_raised_limit() {
        let sem = Arc::new(AsrSemaphore::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let limit = Arc::new(AtomicUsize::new(1));

        // Hold one slot; another acquirer should block.
        let g = sem
            .acquire(|| limit.load(Ordering::Relaxed), &cancel)
            .unwrap();

        let acquired_after_raise = {
            let sem = sem.clone();
            let cancel = cancel.clone();
            let limit = limit.clone();
            thread::spawn(move || {
                sem.acquire(|| limit.load(Ordering::Relaxed), &cancel)
                    .is_some()
            })
        };

        // Give the spawn time to park on the condvar, then raise the cap.
        thread::sleep(Duration::from_millis(80));
        limit.store(2, Ordering::Relaxed);

        assert!(acquired_after_raise.join().unwrap());
        drop(g);
    }

    #[test]
    fn adaptive_semaphore_cancel_unblocks_waiter() {
        let sem = Arc::new(AsrSemaphore::new());
        let cancel = Arc::new(AtomicBool::new(false));
        let limit = Arc::new(AtomicUsize::new(1));

        let _g = sem
            .acquire(|| limit.load(Ordering::Relaxed), &cancel)
            .unwrap();

        let cancel_clone = cancel.clone();
        let sem_clone = sem.clone();
        let limit_clone = limit.clone();
        let waiter = thread::spawn(move || {
            sem_clone
                .acquire(|| limit_clone.load(Ordering::Relaxed), &cancel_clone)
                .is_none()
        });

        thread::sleep(Duration::from_millis(80));
        cancel.store(true, Ordering::Relaxed);
        // The waiter returns None on next wake (within the 50 ms timeout).
        assert!(waiter.join().unwrap());
    }

    /// `AsrPermit<'_>` must be `!Send` so a worker cannot ship its
    /// slot-release responsibility to another thread. This is
    /// enforced at compile time by the `PhantomData<*const ()>` field
    /// (via Rust's auto-trait rules: any struct containing a `*const T`
    /// is `!Send`).
    ///
    /// The lines marked `// COMPILE-FAIL` below would fail to compile
    /// if `AsrPermit<'_>` ever gained `Send`. They are commented out
    /// because cargo test surfaces compile errors as test failures
    /// rather than asserting "did not compile" — but the documented
    /// invariant is verified by uncommenting either line locally.
    #[test]
    fn asr_permit_is_not_send_documentation() {
        // Sanity ground truth: a `Send` type satisfies `requires_send`.
        fn requires_send<T: Send>() {}
        requires_send::<i32>();
        requires_send::<String>();
        // COMPILE-FAIL when uncommented (because `AsrPermit<'_>: !Send`):
        //     requires_send::<AsrPermit<'_>>();
        //
        // Equivalent compile-fail via `thread::spawn`:
        //     let sem = AsrSemaphore::new();
        //     let cancel = std::sync::atomic::AtomicBool::new(false);
        //     let permit = sem.acquire(|| 1, &cancel).unwrap();
        //     std::thread::spawn(move || drop(permit));
    }
}
