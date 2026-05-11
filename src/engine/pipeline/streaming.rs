use crate::engine::chunker::{chunk_audio, ChunkerConfig};
use crate::engine::events::EventSink;
use crate::engine::srt::{append_srt_file, parse_srt_file, SubtitleCue};
use crate::engine::transcribe::detect_speech_intervals_from_wav;
use crate::engine::transcribe::ffprobe_duration_seconds_pub;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Instant;

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

pub(super) struct StreamRunResult {
    pub(super) audio_wav_path: PathBuf,
}

pub(super) fn stream_transcribe_translate_video(
    pipeline: &SubtitlePipeline,
    video: &Path,
    output_srt: &Path,
    events: &EventSink,
) -> Result<StreamRunResult, String> {
    let transcriber = pipeline
        .transcriber
        .as_ref()
        .ok_or_else(|| "streaming mode requires transcription to be enabled".to_string())?;
    let config = &pipeline.config;

    let checkpoint_dir = checkpoint_dir_for(video)?;
    let temp_dir = checkpoint_dir.join("work_stream");
    std::fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;

    // Make sure the output file exists immediately so players can load it early.
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
    let mut next_index = 1usize;
    let mut last_cue = Option::<LastCueFingerprint>::None;

    for chunk in chunks {
        let chunk_started = Instant::now();
        if events.enabled() {
            events.emit(&json!({
                "event": "chunk_started",
                "input": video.display().to_string(),
                "chunk_index": chunk.index,
                "chunk_total": chunk_total,
                "start_sec": chunk.start_sec,
                "end_sec": chunk.end_sec,
                "emitted_at_epoch_secs": now_epoch_secs(),
            }));
        }

        let chunk_len_secs = (chunk.end_sec - chunk.start_sec).max(0.0);
        let timeout_secs = stream_chunk_timeout_secs(config, chunk_len_secs);
        let chunk_srt = transcriber
            .transcribe_wav_to_srt_with_timeout(&chunk.wav_path, timeout_secs)
            .map_err(|e| e.to_string())?;
        let chunk_cues = parse_srt_file(&chunk_srt).map_err(|e| e.to_string())?;
        let translated = if config.speaker_aware {
            let (speakers, _) = infer_speakers(&chunk_cues);
            let tags = build_speaker_tags_with_context(&chunk_cues, &speakers);
            pipeline
                .translator
                .translate_all_with_extra_tags(&chunk_cues, &tags)
                .map_err(|e| e.to_string())?
        } else {
            pipeline
                .translator
                .translate_all(&chunk_cues)
                .map_err(|e| e.to_string())?
        };

        let mut shifted = Vec::with_capacity(translated.len());
        for cue in translated {
            let (start, end) = parse_srt_timing_line(&cue.timing)?;
            let abs_start = start + chunk.start_sec;
            let abs_end = end + chunk.start_sec;
            shifted.push(SubtitleCue {
                index: cue.index,
                timing: format_srt_timing_line(abs_start, abs_end),
                text: cue.text,
            });
        }

        if let Some(first) = shifted.first() {
            if let Some(fingerprint) = last_cue.as_ref() {
                let (start, _) = parse_srt_timing_line(&first.timing)?;
                let time_gap = (start - fingerprint.abs_start).abs();
                if time_gap < 3.0 {
                    let sim = strsim::normalized_levenshtein(&fingerprint.text, &first.text);
                    if sim >= STREAM_DEDUP_SIMILARITY_THRESHOLD {
                        shifted.remove(0);
                    }
                }
            }
        }

        if !shifted.is_empty() {
            next_index =
                append_srt_file(output_srt, &shifted, next_index).map_err(|e| e.to_string())?;
            if let Some(last) = shifted.last() {
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
                "chunk_index": chunk.index,
                "chunk_total": chunk_total,
                "cue_count": chunk_cues.len(),
                "elapsed_secs": chunk_started.elapsed().as_secs_f64(),
                "emitted_at_epoch_secs": now_epoch_secs(),
            }));
        }
    }

    Ok(StreamRunResult {
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
