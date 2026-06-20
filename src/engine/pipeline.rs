// VoiDex Pipeline — Dual-path convergence.
//
// Composes the speed pipeline (parallel chunked transcription) with the
// accuracy pipeline (context-aware neural translation + post-processing).

mod audio;
mod compact;
mod confidence;
mod consistency;
mod diarization;
mod error;
mod escalation;
mod health;
mod learned_gate;
mod loop_collapse;
mod paths;
mod profile;
mod register;
mod relationship;
mod scenes;
mod sidecar;
mod speaker;
mod stream_async;
mod streaming;
mod text;
mod time;
mod trace;
mod util;
mod verify;

use crate::engine::chunker::{chunk_audio, ChunkerConfig};
use crate::engine::events::EventSink;
use crate::engine::parallel::parallel_transcribe;
use crate::engine::srt::{parse_srt_file, write_srt_file, SubtitleCue};
use crate::engine::stitcher::stitch_chunks;
use crate::engine::transcribe::{
    QualityProfile, TranscribeConfig, Transcriber, TranscriptionResult,
};
use crate::engine::translate::{mt_daemon_script_available, Translator, TranslatorConfig};
use audio::{create_temp_rescue_dir, extract_audio_segment_to_wav, extract_audio_to_wav};
use compact::compact_adjacent_cues;
use confidence::{
    collect_low_confidence_cue_spans, load_cue_asr_confidence_from_whisper_json,
    mean_confidence_score, CueAsrConfidence,
};
use consistency::{apply_source_phrase_consistency, apply_source_phrase_consistency_by_speaker};
pub use error::PipelineError;
use error::PipelineResult;
use health::{
    assess_srt_health, assess_translation_semantics, build_metadata_warnings,
    normalize_health_text, scene_semantic_penalty,
};
use paths::{
    checkpoint_dir_for, is_srt_path, is_video_sidecar_source, looks_like_simulated_placeholder_srt,
    metadata_sidecar_path, output_path_for_target_lang, trace_sidecar_path,
};
use profile::{
    default_mt_batch_for_profile, default_mt_oom_retries_for_profile, default_mt_tokens_for_profile,
};
use scenes::{
    assess_scene_quality, collect_low_quality_scene_ranges,
    collect_low_quality_source_scene_ranges, scene_quality_for_slice, source_scene_quality_score,
};
use sidecar::{
    build_scene_metadata, load_checkpoint_summary, write_parallel_confidence_sidecar,
    write_runtime_trace_sidecar,
};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use text::{
    assess_name_inconsistency, cue_has_adjacent_repeat, cue_has_low_function_word_coverage,
    cue_has_malformed_contraction, token_has_double_apostrophe, tokenize_ascii_words,
};
use time::{format_srt_timing_line, parse_srt_timing_line, scene_time_span, shift_cues_by_offset};
use trace::RuntimeTrace;
use util::{find_in_path, has_repeated_ngram, now_epoch_secs};
use verify::verify_srt_against_audio;

use diarization::{audio_diarize_speakers_for_cues, AudioDiarizationStats};
use learned_gate::evaluate as evaluate_learned_gate;
use loop_collapse::collapse_asr_decode_loops;
use register::infer_speaker_register_tags;
use relationship::write_relationship_graph_sidecar;
use speaker::{build_speaker_tags_with_context, infer_speakers};
use stream_async::stream_transcribe_translate_video_async;
use streaming::stream_transcribe_translate_video;

fn record_runtime_stage(
    events: &EventSink,
    runtime_trace: Option<&mut RuntimeTrace>,
    input: &Path,
    stage: &str,
    started: std::time::Instant,
    details: serde_json::Value,
) {
    if let Some(trace) = runtime_trace {
        trace.record_stage(stage, started, details.clone());
    }
    if !events.enabled() {
        return;
    }

    events.emit(&serde_json::json!({
        "event": "stage_complete",
        "input": input.display().to_string(),
        "stage": stage,
        "elapsed_secs": started.elapsed().as_secs_f64(),
        "details": details,
        "emitted_at_epoch_secs": now_epoch_secs(),
    }));
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub source_lang: String,
    pub target_lang: String,
    pub offline: bool,
    pub transcribe: bool,
    pub whisper_bin: Option<PathBuf>,
    pub whisper_model: Option<PathBuf>,
    pub whisper_args: Vec<String>,
    /// Prefer an audio track by language when extracting audio from video inputs (e.g. "jpn", "eng").
    pub audio_lang: Option<String>,
    /// Force a specific ffmpeg stream index for audio extraction (overrides audio_lang).
    pub audio_stream_index: Option<usize>,
    pub skip_existing: bool,
    pub vad: bool,
    pub vad_threshold_db: f64,
    pub vad_min_silence: f64,
    pub vad_pad: f64,
    pub verify: bool,
    pub verify_min_speech_overlap: f64,
    pub gpu: bool,
    pub require_gpu: bool,

    /// Enable parallel chunked transcription (speed pipeline).
    pub parallel: bool,
    /// Stream chunk-by-chunk transcription+translation while writing the output SRT progressively.
    pub stream: bool,
    /// Stream with bounded channels (overlap transcribe and translate) while still writing progressively.
    pub stream_async: bool,
    /// Max parallel whisper workers.
    pub max_workers: usize,
    /// Target chunk duration in seconds (default 300 = 5 min).
    pub chunk_duration_secs: f64,
    /// Force phrase-table translator instead of neural MT.
    pub force_phrase_table: bool,
    /// Enable speaker-aware discourse translation (Phase E) using deterministic speaker inference.
    pub speaker_aware: bool,
    /// Enable audio-based speaker diarization to fill missing speaker tags (Phase E).
    pub speaker_diarize: bool,
    /// Upper bound for diarization speaker clusters.
    pub speaker_max_speakers: usize,
    /// Override the MT model name (e.g. "nllb-200-distilled-600M").
    pub mt_model: Option<String>,
    /// Override MT decoding batch size.
    pub mt_batch_size: Option<usize>,
    /// Override MT token batch size ceiling.
    pub mt_max_batch_tokens: Option<usize>,
    /// Override MT CUDA OOM retries.
    pub mt_oom_retries: Option<usize>,
    /// Allow CPU fallback when MT CUDA runs out of memory.
    pub mt_allow_cpu_fallback: bool,
    /// Force MT to use CPU even when `--gpu` is set (useful when ASR needs all VRAM).
    pub mt_force_cpu: bool,
    /// Use a persistent Python daemon for neural MT instead of per-run subprocesses.
    pub mt_daemon: bool,
    /// Enforce the neural MT heuristic quality floor (disable for best-effort corpus runs).
    pub mt_enforce_quality_floor: bool,
    /// If sidecar subtitles look degenerate, auto-transcribe from video audio.
    pub auto_repair_sidecar: bool,
    /// Emit a per-run runtime trace sidecar for profiling and benchmarking.
    pub trace_runtime: bool,
    /// Emit JSON-lines events during processing (Phase C scaffolding).
    pub events_json: bool,
    /// If set, append emitted JSON-lines events to this file as well as stdout.
    pub events_file: Option<PathBuf>,
    /// If set, forward emitted JSON-lines events to a local sidecar server.
    pub http_events: Option<Sender<String>>,
    /// If set, forward emitted JSON-lines events to a local websocket sidecar server.
    pub ws_events: Option<Sender<String>>,
    /// Quality/latency operating mode.
    pub quality_profile: QualityProfile,
}

pub struct SubtitlePipeline {
    config: PipelineConfig,
    translator: Translator,
    transcriber: Option<Transcriber>,
}

impl SubtitlePipeline {
    pub fn new(config: PipelineConfig) -> PipelineResult<Self> {
        let transcriber =
            Transcriber::new(Self::make_transcribe_config(&config, config.transcribe)).map_err(
                |err| PipelineError::Initialization {
                    message: err.to_string(),
                },
            )?;
        let translator = Translator::new(TranslatorConfig {
            source_lang: config.source_lang.clone(),
            target_lang: config.target_lang.clone(),
            offline: config.offline,
            force_phrase_table: config.force_phrase_table,
            gpu: config.gpu,
            require_gpu: config.require_gpu,
            mt_model: config.mt_model.clone(),
            mt_batch_size: config.mt_batch_size,
            mt_max_batch_tokens: config.mt_max_batch_tokens,
            mt_oom_retries: config.mt_oom_retries,
            mt_allow_cpu_fallback: config.mt_allow_cpu_fallback,
            mt_force_cpu: config.mt_force_cpu,
            mt_daemon: config.mt_daemon,
            mt_enforce_quality_floor: config.mt_enforce_quality_floor,
            mt_beam_size: None,
            quality_profile: config.quality_profile,
        })
        .map_err(|err| PipelineError::Initialization {
            message: err.to_string(),
        })?;
        Ok(Self {
            config,
            translator,
            transcriber,
        })
    }

    fn target_lang_for_whisper(config: &PipelineConfig) -> String {
        let is_translation = !config.source_lang.eq_ignore_ascii_case(&config.target_lang);
        let use_neural_mt = is_translation && !config.force_phrase_table;
        if use_neural_mt {
            config.source_lang.clone()
        } else {
            config.target_lang.clone()
        }
    }

    fn make_transcribe_config(config: &PipelineConfig, enabled: bool) -> TranscribeConfig {
        TranscribeConfig {
            enabled,
            whisper_bin: config.whisper_bin.clone(),
            whisper_model: config.whisper_model.clone(),
            source_lang: config.source_lang.clone(),
            target_lang: Self::target_lang_for_whisper(config),
            whisper_args: config.whisper_args.clone(),
            audio_lang: config.audio_lang.clone(),
            audio_stream_index: config.audio_stream_index,
            vad: config.vad,
            vad_threshold_db: config.vad_threshold_db,
            vad_min_silence: config.vad_min_silence,
            vad_pad: config.vad_pad,
            gpu: config.gpu,
            require_gpu: config.require_gpu,
            quality_profile: config.quality_profile,
        }
    }

    pub fn process_input(&self, input: &Path) -> PipelineResult<PathBuf> {
        self.process_input_inner(input)
            .map_err(|message| PipelineError::ProcessInput {
                input: input.to_path_buf(),
                message,
            })
    }

    fn process_input_inner(&self, input: &Path) -> Result<PathBuf, String> {
        let output = output_path_for_target_lang(input, &self.config.target_lang)?;
        let trace_path = if self.config.trace_runtime {
            Some(trace_sidecar_path(input)?)
        } else {
            None
        };
        let mut runtime_trace = self.config.trace_runtime.then(RuntimeTrace::new);
        if self.config.skip_existing && output.exists() {
            return Ok(output);
        }

        let events = EventSink::new(
            self.config.events_json,
            self.config.events_file.as_deref(),
            self.config.http_events.clone(),
            self.config.ws_events.clone(),
        )?;

        if self.config.stream && !is_srt_path(input) {
            let streaming_started = std::time::Instant::now();
            let (stage_name, audio_wav_path) = if self.config.stream_async {
                let run = stream_transcribe_translate_video_async(self, input, &output, &events)?;
                ("stream_async_transcribe_translate", run.audio_wav_path)
            } else {
                let run = stream_transcribe_translate_video(self, input, &output, &events)?;
                ("stream_transcribe_translate", run.audio_wav_path)
            };
            record_runtime_stage(
                &events,
                runtime_trace.as_mut(),
                input,
                stage_name,
                streaming_started,
                serde_json::json!({
                    "output": output.display().to_string(),
                    "audio_wav": audio_wav_path.display().to_string(),
                }),
            );

            let metadata_started = std::time::Instant::now();
            let translated = parse_srt_file(&output).map_err(|error| error.to_string())?;
            self.write_metadata_sidecar(
                input,
                &output,
                &translated,
                trace_path.as_deref(),
                None,
                None,
                None,
            )?;
            record_runtime_stage(
                &events,
                runtime_trace.as_mut(),
                input,
                "write_metadata_sidecar",
                metadata_started,
                serde_json::json!({
                    "cue_count": translated.len(),
                    "metadata_file": metadata_sidecar_path(input)?.display().to_string(),
                }),
            );

            let verify_started = std::time::Instant::now();
            let mut verify_ran = false;
            if self.config.verify {
                verify_ran = true;
                let report = verify_srt_against_audio(
                    &output,
                    &audio_wav_path,
                    self.config.vad_threshold_db,
                    self.config.vad_min_silence,
                    self.config.vad_pad,
                    self.config.verify_min_speech_overlap,
                )?;
                let report_path = output.with_extension("verify.txt");
                std::fs::write(&report_path, report)
                    .map_err(|e| format!("{}: {e}", report_path.display()))?;
            }
            record_runtime_stage(
                &events,
                runtime_trace.as_mut(),
                input,
                "verify_output",
                verify_started,
                serde_json::json!({
                    "enabled": self.config.verify,
                    "ran": verify_ran,
                    "audio_available": true,
                }),
            );

            if let (Some(trace), Some(trace_path)) = (runtime_trace.as_ref(), trace_path.as_ref()) {
                let payload = trace.as_json(input, &output, &self.config);
                write_runtime_trace_sidecar(trace_path, &payload)?;
            }

            return Ok(output);
        }

        let resolve_source_started = std::time::Instant::now();
        let (mut source_srt, mut audio_for_verify) =
            self.resolve_subtitle_source(input, &events)?;
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "resolve_subtitle_source",
            resolve_source_started,
            serde_json::json!({
                "source_srt": source_srt.display().to_string(),
                "audio_for_verify": audio_for_verify.as_ref().map(|path| path.display().to_string()),
            }),
        );

        let load_source_started = std::time::Instant::now();
        let mut cues = parse_srt_file(&source_srt).map_err(|error| error.to_string())?;
        let mut source_confidence = load_cue_asr_confidence_from_whisper_json(&source_srt, &cues);
        // Strip whisper decode-loops from the source transcript before any
        // translation: a stuck-decoder run of identical cues is low-entropy
        // and slips past the per-segment non-speech guards, but if left in
        // place it collapses MT quality and fails the quality floor for the
        // entire job. See `loop_collapse`.
        let loop_stats = collapse_asr_decode_loops(&mut cues, &mut source_confidence);
        if loop_stats.collapsed_anything() {
            eprintln!(
                "warning: ibvoid-doom-qlock asr_loop_collapsed runs={} cues_removed={}",
                loop_stats.runs, loop_stats.cues_removed
            );
        }
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "load_source_srt",
            load_source_started,
            serde_json::json!({
                "cue_count": cues.len(),
            }),
        );

        let sidecar_started = std::time::Instant::now();
        let used_video_sidecar = is_video_sidecar_source(input, &source_srt);
        let mut sidecar_rescue_applied = false;
        if used_video_sidecar {
            let health = assess_srt_health(&cues)?;
            if health.is_pathological(self.config.quality_profile) {
                let details = health.summary();
                if !self.config.auto_repair_sidecar {
                    return Err(format!(
                        "sidecar subtitles look degraded ({details}). re-run with --transcribe, or pass --auto-repair-sidecar."
                    ));
                }

                eprintln!(
                    "warning: sidecar subtitles look degraded ({details}); attempting rescue transcription from audio..."
                );
                let rescue = self.rescue_transcribe_video(input, &events)?;
                sidecar_rescue_applied = true;
                source_srt = rescue.srt_path;
                audio_for_verify = Some(rescue.audio_wav_path);
                cues = parse_srt_file(&source_srt).map_err(|error| error.to_string())?;
                source_confidence = load_cue_asr_confidence_from_whisper_json(&source_srt, &cues);
                // Same decode-loop collapse on the rescued transcript.
                let rescue_loop_stats =
                    collapse_asr_decode_loops(&mut cues, &mut source_confidence);
                if rescue_loop_stats.collapsed_anything() {
                    eprintln!(
                        "warning: ibvoid-doom-qlock asr_loop_collapsed runs={} cues_removed={}",
                        rescue_loop_stats.runs, rescue_loop_stats.cues_removed
                    );
                }
                let rescue_health = assess_srt_health(&cues)?;
                if rescue_health.is_pathological(self.config.quality_profile) {
                    if self.config.quality_profile == QualityProfile::Strict {
                        return Err(format!(
                            "rescue transcription failed strict quality gate: {}",
                            rescue_health.summary()
                        ));
                    }
                    eprintln!(
                        "warning: rescue transcription still looks noisy ({}); continuing with best available output.",
                        rescue_health.summary()
                    );
                }
            }
        }
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "sidecar_health_and_rescue",
            sidecar_started,
            serde_json::json!({
                "used_video_sidecar": used_video_sidecar,
                "rescue_applied": sidecar_rescue_applied,
                "source_srt": source_srt.display().to_string(),
                "cue_count_after": cues.len(),
            }),
        );

        let source_rescue_started = std::time::Instant::now();
        self.rescue_low_quality_source_transcription(
            &mut cues,
            audio_for_verify.as_deref(),
            source_confidence.as_deref(),
        )?;
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "source_rescue",
            source_rescue_started,
            serde_json::json!({
                "cue_count_after": cues.len(),
            }),
        );

        let source_gate_started = std::time::Instant::now();
        self.enforce_source_quality_gate(&cues)?;
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "source_quality_gate",
            source_gate_started,
            serde_json::json!({
                "cue_count": cues.len(),
            }),
        );

        // Translate — batched (neural) or per-cue (phrase-table).
        // Speaker-aware discourse (Phase E) is opt-in via config/CLI.
        let mut speakers = Vec::<Option<String>>::new();
        let mut speaker_tags = Vec::<Vec<String>>::new();
        let mut speaker_info = Option::<serde_json::Value>::None;
        let mut register_tags = Vec::<Vec<String>>::new();
        if self.config.speaker_aware {
            let (text_inferred, text_stats) = infer_speakers(&cues);
            if text_stats.labeled_cues > 0 {
                eprintln!(
                    "ibvoid-doom-qlock: speaker-infer labeled={} unique={} cues={}",
                    text_stats.labeled_cues,
                    text_stats.unique_speakers,
                    cues.len()
                );
            }

            let mut merged = text_inferred;
            let mut audio_stats = AudioDiarizationStats::default();
            if self.config.speaker_diarize {
                match audio_diarize_speakers_for_cues(
                    input,
                    audio_for_verify.as_deref(),
                    &cues,
                    self.config.vad_threshold_db,
                    self.config.vad_min_silence,
                    self.config.vad_pad,
                    self.config.speaker_max_speakers,
                ) {
                    Ok((audio_speakers, stats)) => {
                        audio_stats = stats;
                        for (slot, audio) in merged.iter_mut().zip(audio_speakers.into_iter()) {
                            if slot.is_none() && audio.is_some() {
                                *slot = audio;
                            }
                        }
                    }
                    Err(error) => {
                        eprintln!(
                            "warning: speaker diarization failed ({}): {error}",
                            input.display()
                        );
                    }
                }
            }

            let merged_labeled = merged.iter().filter(|s| s.is_some()).count();
            let merged_unique = merged
                .iter()
                .filter_map(|s| s.as_ref())
                .collect::<std::collections::HashSet<_>>()
                .len();

            let (register_infer_tags, _register, register_stats) =
                infer_speaker_register_tags(&cues, &merged);
            let (graph_path, graph_stats) =
                match write_relationship_graph_sidecar(input, &cues, &merged) {
                    Ok(result) => result,
                    Err(error) => {
                        eprintln!(
                            "warning: speaker-aware relationship graph write failed ({}): {error}",
                            input.display()
                        );
                        (None, relationship::RelationshipGraphStats::default())
                    }
                };

            speaker_info = Some(serde_json::json!({
                "enabled": true,
                "labeled_cues": merged_labeled,
                "unique_speakers": merged_unique,
                "text_infer": {
                    "labeled_cues": text_stats.labeled_cues,
                    "unique_speakers": text_stats.unique_speakers,
                },
                "audio_diarization": {
                    "enabled": self.config.speaker_diarize,
                    "audio_available": audio_stats.audio_available,
                    "speakers": audio_stats.speakers,
                    "used_segments": audio_stats.used_segments,
                    "assigned_cues": audio_stats.assigned_cues,
                    "unique_speakers": audio_stats.unique_speakers,
                    "file": audio_stats.sidecar_file.as_ref().map(|path| path.display().to_string()),
                },
                "register": {
                    "speakers_observed": register_stats.speakers_observed,
                    "speakers_formal": register_stats.speakers_formal,
                    "cues_labeled": register_stats.cues_labeled,
                },
                "relationship_graph_file": graph_path.as_ref().map(|path| path.display().to_string()),
                "graph": {
                    "nodes": graph_stats.node_count,
                    "edges": graph_stats.edge_count,
                    "utterances_labeled": graph_stats.utterances_labeled,
                }
            }));
            speaker_tags = build_speaker_tags_with_context(&cues, &merged);
            register_tags = register_infer_tags;
            speakers = merged;

            let count = speaker_tags.len().min(register_tags.len());
            for idx in 0..count {
                if register_tags[idx].is_empty() {
                    continue;
                }
                speaker_tags[idx].extend(register_tags[idx].iter().cloned());
            }
        }

        let translate_started = std::time::Instant::now();
        // CPT collapse-phase routing: translate a cheap NLLB prefix, certify
        // backend-document collapse early, and if certified reroute the WHOLE
        // document to the local LLM rung instead of finishing a doomed NLLB
        // pass. Falls back to a normal full NLLB translate when the LLM rung is
        // unavailable or the reroute fails — never worse than before.
        let mut translated = self.translate_with_collapse_routing(&cues, &speaker_tags, &events)?;
        // Q1: a non-Strict profile may have emitted best-effort output below the
        // neural MT quality floor. Capture the reason now so the sidecar verdict
        // can record verdict.pass=false while the SRT is still written.
        let best_effort_floor_reason = self.translator.take_best_effort_floor_reason();
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "translate",
            translate_started,
            serde_json::json!({
                "backend": format!("{:?}", self.translator.backend()),
                "source_cues": cues.len(),
                "translated_cues": translated.len(),
            }),
        );

        let mut glossary = crate::engine::character_glossary::CharacterGlossary::load_default();
        glossary.apply(&mut translated);
        glossary.learn(&translated);
        if let Err(error) = glossary.save() {
            eprintln!("warning: character glossary save failed: {error}");
        }

        let scene_rescue_started = std::time::Instant::now();
        self.rescue_low_quality_scene_translations(
            &cues,
            &mut translated,
            &events,
            runtime_trace.as_mut(),
        )?;
        if !register_tags.is_empty() {
            register::apply_register_post_edit(&mut translated, &register_tags);
        }
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "scene_translation_rescue",
            scene_rescue_started,
            serde_json::json!({
                "cue_count_after": translated.len(),
            }),
        );

        let discourse_started = std::time::Instant::now();
        self.enforce_discourse_consistency(&cues, &mut translated, &speakers);
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "discourse_consistency",
            discourse_started,
            serde_json::json!({
                "cue_count": translated.len(),
            }),
        );

        let voice_started = std::time::Instant::now();
        let mut voice_priors = crate::engine::voice_consistency::VoicePriors::load_default();
        let (voice_reports, voice_stats) = voice_priors.process_batch(&translated, &speakers);
        if let Err(error) = voice_priors.save() {
            eprintln!("warning: voice priors save failed: {error}");
        }
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "voice_consistency",
            voice_started,
            serde_json::json!({
                "cues_scored": voice_stats.cues_scored,
                "speakers_observed": voice_stats.speakers_observed,
                "mean_deviation": voice_stats.mean_deviation,
                "p95_deviation": voice_stats.p95_deviation,
                "max_deviation": voice_stats.max_deviation,
                "reports": voice_reports.len(),
            }),
        );

        let compact_started = std::time::Instant::now();
        let cues_before_compaction = translated.len();
        translated = self.compact_translated_cues(translated)?;
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "compact_translated_cues",
            compact_started,
            serde_json::json!({
                "cues_before": cues_before_compaction,
                "cues_after": translated.len(),
            }),
        );

        let translated_gate_started = std::time::Instant::now();
        self.enforce_translated_quality_gate(&translated)?;
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "translated_quality_gate",
            translated_gate_started,
            serde_json::json!({
                "cue_count": translated.len(),
            }),
        );

        // Final post-processing pass: catches whisper hallucinations (repetitive
        // non-speech patterns), normalizes sound effect labels, ensures quality.
        // Runs unconditionally — even for same-language transcription where the
        // translator might short-circuit without calling its internal postprocess.
        crate::engine::postprocess::postprocess(&mut translated);

        let write_output_started = std::time::Instant::now();
        write_srt_file(&output, &translated).map_err(|error| error.to_string())?;
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "write_output_srt",
            write_output_started,
            serde_json::json!({
                "output_file": output.display().to_string(),
                "cue_count": translated.len(),
            }),
        );

        let write_metadata_started = std::time::Instant::now();
        self.write_metadata_sidecar(
            input,
            &output,
            &translated,
            trace_path.as_deref(),
            speaker_info,
            Some(&voice_stats),
            best_effort_floor_reason.as_deref(),
        )?;
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "write_metadata_sidecar",
            write_metadata_started,
            serde_json::json!({
                "metadata_file": metadata_sidecar_path(input)?.display().to_string(),
            }),
        );

        let verify_started = std::time::Instant::now();
        let mut verify_ran = false;
        if self.config.verify {
            if let Some(audio_path) = audio_for_verify.as_ref() {
                verify_ran = true;
                let report = verify_srt_against_audio(
                    &output,
                    audio_path,
                    self.config.vad_threshold_db,
                    self.config.vad_min_silence,
                    self.config.vad_pad,
                    self.config.verify_min_speech_overlap,
                )?;
                let report_path = output.with_extension("verify.txt");
                std::fs::write(&report_path, report)
                    .map_err(|e| format!("{}: {e}", report_path.display()))?;
            } else {
                eprintln!(
                    "warning: --verify was requested but no audio path was available (verification skipped)."
                );
            }
        }
        record_runtime_stage(
            &events,
            runtime_trace.as_mut(),
            input,
            "verify_output",
            verify_started,
            serde_json::json!({
                "enabled": self.config.verify,
                "ran": verify_ran,
                "audio_available": audio_for_verify.is_some(),
            }),
        );

        if let (Some(trace), Some(trace_path)) = (runtime_trace.as_ref(), trace_path.as_ref()) {
            let payload = trace.as_json(input, &output, &self.config);
            write_runtime_trace_sidecar(trace_path, &payload)?;
        }

        Ok(output)
    }

    #[allow(clippy::too_many_arguments)] // sidecar inputs are independent optionals; a one-off struct adds no clarity
    fn write_metadata_sidecar(
        &self,
        input: &Path,
        output: &Path,
        translated: &[SubtitleCue],
        runtime_trace_path: Option<&Path>,
        speaker_info: Option<serde_json::Value>,
        voice_stats: Option<&crate::engine::voice_consistency::VoiceConsistencyStats>,
        best_effort_floor_reason: Option<&str>,
    ) -> Result<(), String> {
        let structural = assess_srt_health(translated)?;
        let semantic = assess_translation_semantics(translated, &self.config.target_lang);
        let scene_metrics = build_scene_metadata(translated);
        let checkpoint_summary = load_checkpoint_summary(input);
        let metadata_path = metadata_sidecar_path(input)?;
        let metadata_output_path = metadata_sidecar_path(output)?;
        let learned_gate = evaluate_learned_gate(&structural, &semantic, speaker_info.as_ref());

        // Pre-compute verdict incorporating chunk failure rate.
        // If > 30% of parallel chunks failed, the output is incomplete regardless
        // of per-cue quality metrics (which only evaluate what exists, not what's missing).
        let chunk_failure_ratio = checkpoint_summary
            .as_ref()
            .and_then(|cp| {
                let completed = cp.get("completed_chunks")?.as_u64()?;
                let failed = cp.get("failed_chunks")?.as_u64()?;
                let total = completed + failed;
                if total > 0 {
                    Some(failed as f64 / total as f64)
                } else {
                    None
                }
            })
            .unwrap_or(0.0);
        let semantic_fail = semantic.is_pathological(self.config.quality_profile);
        let coverage_degraded = chunk_failure_ratio > 0.3;
        // Q1: when the MT quality floor failed but the profile (Fast/Balanced)
        // emitted best-effort subtitles anyway, the SRT is written but the
        // verdict must record pass=false with the floor reason retained so the
        // honest quality signal reaches the user / TUI. See agent-memory
        // finding_ja_en_casual_mt_quality (JA→EN casual MT limit).
        let best_effort_fail = best_effort_floor_reason.is_some();
        let verdict_pass = !semantic_fail && !coverage_degraded && !best_effort_fail;
        let verdict_reason = if let Some(reason) = best_effort_floor_reason {
            reason.to_string()
        } else if semantic_fail {
            semantic.summary()
        } else if coverage_degraded {
            format!(
                "partial: {:.0}% of chunks failed during transcription",
                chunk_failure_ratio * 100.0
            )
        } else {
            "quality gate passed".to_string()
        };

        let payload = serde_json::json!({
            "version": "1.1",
            "algorithm": "IBVoid DOOM-QLOCK",
            "source_file": input.display().to_string(),
            "output_file": output.display().to_string(),
            "source_language": self.config.source_lang,
            "target_language": self.config.target_lang,
            "quality_profile": self.config.quality_profile.as_str(),
            "runtime_trace_file": runtime_trace_path.map(|path| path.display().to_string()),
            "generated_at_epoch_secs": now_epoch_secs(),
            "speaker": speaker_info,
            "voice_consistency": voice_stats.map(|s| serde_json::json!({
                "cues_scored": s.cues_scored,
                "speakers_observed": s.speakers_observed,
                "mean_deviation": s.mean_deviation,
                "p95_deviation": s.p95_deviation,
                "max_deviation": s.max_deviation,
            })),
            "learned_gate": learned_gate.as_ref().map(|out| serde_json::json!({
                "enabled": true,
                "model": out.model_path.display().to_string(),
                "schema_version": out.schema_version,
                "enforce": out.enforce,
                "threshold": out.threshold,
                "score": out.score,
                "pass": out.pass(),
                "decision": out.decision,
                "lower_bound": out.lower_bound,
                "upper_bound": out.upper_bound,
                "conformal_alpha": out.conformal_alpha,
                "conformal_q_hat": out.conformal_q_hat,
                "conformal_cal_size": out.conformal_cal_size,
                "isotonic_out_of_bounds": out.isotonic_out_of_bounds,
            })),
            "plan_used": {
                "parallel": self.config.parallel,
                "workers": self.config.max_workers,
                "chunk_duration_secs": self.config.chunk_duration_secs,
                "mt_batch_size": self.config.mt_batch_size,
                "mt_max_batch_tokens": self.config.mt_max_batch_tokens,
                "mt_oom_retries": self.config.mt_oom_retries,
                "mt_allow_cpu_fallback": self.config.mt_allow_cpu_fallback,
                "mt_daemon": self.config.mt_daemon,
                "mt_enforce_quality_floor": self.config.mt_enforce_quality_floor,
            },
            "quality": {
                "cue_count": translated.len(),
                "structural": {
                    "top_line_ratio": structural.top_line_ratio,
                    "overlap_ratio": structural.overlap_ratio,
                    "non_empty_ratio": structural.non_empty_ratio,
                },
                "semantic": {
                    "anomaly_ratio": semantic.anomaly_ratio,
                    "malformed_contraction_ratio": semantic.malformed_contraction_ratio,
                    "low_function_word_ratio": semantic.low_function_word_ratio,
                    "adjacent_repeat_ratio": semantic.adjacent_repeat_ratio,
                    "scene_low_quality_ratio": semantic.scene_low_quality_ratio,
                    "scene_count": semantic.scene_count,
                    "name_inconsistency_ratio": semantic.name_inconsistency_ratio,
                },
                "per_scene": scene_metrics,
            },
            "recovery_events": checkpoint_summary
                .as_ref()
                .map(|summary| summary.get("recovery_events").cloned().unwrap_or_else(|| serde_json::json!([])))
                .unwrap_or_else(|| serde_json::json!([])),
            "checkpoint": checkpoint_summary.as_ref().cloned().unwrap_or_else(|| serde_json::json!({
                "status": "none",
                "completed_chunks": 0,
                "failed_chunks": 0,
                "recovery_events": [],
            })),
            "warnings": build_metadata_warnings(&semantic),
            "verdict": {
                "pass": verdict_pass,
                "reason": verdict_reason,
                "chunk_failure_ratio": chunk_failure_ratio,
            }
        });
        let serialized = serde_json::to_string_pretty(&payload)
            .map_err(|e| format!("{} serialize metadata: {e}", metadata_path.display()))?;
        std::fs::write(&metadata_path, &serialized)
            .map_err(|e| format!("{}: {e}", metadata_path.display()))?;
        if metadata_output_path != metadata_path {
            std::fs::write(&metadata_output_path, &serialized)
                .map_err(|e| format!("{}: {e}", metadata_output_path.display()))?;
        }
        Ok(())
    }

    fn enforce_source_quality_gate(&self, cues: &[SubtitleCue]) -> Result<(), String> {
        let health = assess_srt_health(cues)?;
        if !health.is_pathological(self.config.quality_profile) {
            return Ok(());
        }

        match self.config.quality_profile {
            QualityProfile::Strict => Err(format!(
                "strict quality gate failed for source subtitles: {}",
                health.summary()
            )),
            _ => {
                eprintln!(
                    "warning: source subtitles triggered quality gate ({}); continuing because profile={}.",
                    health.summary(),
                    self.config.quality_profile.as_str()
                );
                Ok(())
            }
        }
    }

    fn enforce_translated_quality_gate(&self, cues: &[SubtitleCue]) -> Result<(), String> {
        let structural = assess_srt_health(cues)?;
        let health = assess_translation_semantics(cues, &self.config.target_lang);
        let learned = evaluate_learned_gate(&structural, &health, None);
        if !health.is_pathological(self.config.quality_profile) {
            if let Some(out) = learned {
                // Enforcement policy v2:
                //   - REJECT with enforce ⇒ hard fail.
                //   - REJECT without enforce ⇒ warn, continue.
                //   - ABSTAIN ⇒ warn (the gate honestly does not know),
                //     never enforce — abstain is not a fail.
                //   - PASS ⇒ silent.
                if out.decision.is_reject() && out.enforce {
                    return Err(format!(
                        "learned quality gate REJECT: score={:.3} band=[{:.3},{:.3}] threshold={:.3} model={}",
                        out.score,
                        out.lower_bound,
                        out.upper_bound,
                        out.threshold,
                        out.model_path.display()
                    ));
                }
                if !out.pass() {
                    eprintln!(
                        "warning: learned gate {} score={:.3} band=[{:.3},{:.3}] threshold={:.3} (model={})",
                        out.decision,
                        out.score,
                        out.lower_bound,
                        out.upper_bound,
                        out.threshold,
                        out.model_path.display()
                    );
                }
            }
            return Ok(());
        }

        match self.config.quality_profile {
            QualityProfile::Strict => Err(format!(
                "strict quality gate failed for translated subtitles: {}",
                health.summary()
            )),
            _ => {
                eprintln!(
                    "warning: translated subtitles triggered semantic quality gate ({}); continuing because profile={}.",
                    health.summary(),
                    self.config.quality_profile.as_str()
                );
                Ok(())
            }
        }
    }

    // Per-segment MT escalation ladder (guardrails #1–#6).
    //
    // Policy rationale + evidence: see agent-memory finding_ja_en_casual_mt_quality
    // (JA→EN casual MT limit). A failed scene is retried ONLY at a stronger rung,
    // bounded by per-profile reach and a session budget cap; one attempt per scene.
    fn rescue_low_quality_scene_translations(
        &self,
        source_cues: &[SubtitleCue],
        translated: &mut [SubtitleCue],
        events: &EventSink,
        mut runtime_trace: Option<&mut RuntimeTrace>,
    ) -> Result<(), String> {
        if !self.config.target_lang.eq_ignore_ascii_case("en")
            || source_cues.len() != translated.len()
        {
            return Ok(());
        }

        let base_beam = profile::default_mt_beam_for_profile(self.config.quality_profile);
        // WS-A: add the local LLM (Qwen3) rung when the llama-server binary and
        // GGUF model both resolve — the tier that rescues casual speech NLLB
        // degenerates on. Absent (or in phrase-table/test mode), the ladder is
        // the historical NLLB-only ladder, so nothing regresses without a model.
        let llm_rung = if self.config.force_phrase_table {
            None
        } else {
            crate::engine::llm_mt::resolve_llm_paths().map(|_| escalation::MtBackendStep {
                label: "qwen3-4b",
                model_name: "qwen3-4b".to_string(),
                beam_size: 1,
                kind: escalation::BackendKind::Llm,
            })
        };
        let policy = escalation::EscalationPolicy::for_profile(
            self.config.quality_profile,
            base_beam,
            llm_rung,
        );
        // Guardrail #1: Fast never escalates up — nothing to do here.
        if !policy.can_escalate() {
            return Ok(());
        }

        let total_segments = scenes::split_scene_ranges(translated).len();
        let mut low_scenes = collect_low_quality_scene_ranges(translated);
        if low_scenes.is_empty() {
            return Ok(());
        }

        // Process worst-deficit scenes first so the budget is spent where it matters.
        low_scenes.sort_by(|a, b| {
            let lhs = b.floor - b.score;
            let rhs = a.floor - a.score;
            lhs.total_cmp(&rhs)
        });

        // Map each failing scene to its scene index (the telemetry `segment`).
        let scene_ranges = scenes::split_scene_ranges(translated);
        let failing: Vec<escalation::FailingSegment> = low_scenes
            .iter()
            .map(|scene| escalation::FailingSegment {
                index: scene_ranges
                    .iter()
                    .position(|(start, end)| *start == scene.start && *end == scene.end)
                    .unwrap_or(0),
                start: scene.start,
                end: scene.end,
                base_score: scene.score,
                floor: scene.floor,
            })
            .collect();

        eprintln!(
            "ibvoid-doom-qlock: scene-escalation evaluating {} low-quality scenes (profile={})",
            failing.len(),
            self.config.quality_profile.as_str()
        );

        let backend = NeuralSegmentBackend::new(self);
        let scorer = RealSceneScorer;
        let report = escalation::run_escalation_ladder(
            &policy,
            &backend,
            &scorer,
            source_cues,
            translated,
            &failing,
            total_segments,
        );

        // Guardrail #5: emit one structured `mt_escalation` event per attempt to
        // the same trace/event stream the TUI reads, plus a human stderr line.
        for event in &report.events {
            let payload = event.as_event();
            events.emit(&payload);
            if let Some(trace) = runtime_trace.as_deref_mut() {
                trace.record_stage("mt_escalation", std::time::Instant::now(), payload.clone());
            }
            eprintln!(
                "ibvoid-doom-qlock: mt_escalation segment={} {}->{} outcome={} score {:.3}->{:.3}",
                event.segment,
                event.from,
                event.to,
                if event.recovered {
                    "recovered"
                } else {
                    "still_failed"
                },
                event.gate_score_before,
                event.gate_score_after,
            );
        }

        // Splice recovered scenes back in. Untouched scenes keep their best-effort
        // 600M text (guardrail #2 — passing scenes are never re-translated).
        for recovered in &report.recovered_texts {
            for (dst, text) in translated[recovered.start..recovered.end]
                .iter_mut()
                .zip(recovered.texts.iter())
            {
                dst.text = text.clone();
            }
        }

        // Guardrail #3: budget cap tripped → remaining scenes stay best-effort.
        // This is the NLLB-incompatible-content signal, not a transient failure.
        if report.budget_exceeded {
            eprintln!(
                "warning: ibvoid-doom-qlock mt_escalation_budget_exceeded ratio>{:.0}% \
                 remaining low-quality scenes emitted best-effort (content class \
                 appears incompatible with NLLB; more compute will not help).",
                escalation::ESCALATION_BUDGET_RATIO * 100.0
            );
        }

        let recovered = report
            .outcomes
            .iter()
            .filter(|(_, o)| matches!(o, escalation::SegmentOutcome::Recovered { .. }))
            .count();
        if recovered == 0 {
            eprintln!("ibvoid-doom-qlock: scene-escalation found no beneficial rewrites.");
        }

        // Guardrail #1 tail: Strict hard-fails if any escalated scene could not
        // be brought above its floor (best-effort-exhausted or budget-skipped).
        // Fast/Balanced keep the best-effort scenes (no error).
        if policy.hard_fail_on_exhaustion() {
            let unrecovered = report
                .outcomes
                .iter()
                .filter(|(_, o)| !matches!(o, escalation::SegmentOutcome::Recovered { .. }))
                .count();
            if unrecovered > 0 {
                return Err(format!(
                    "strict scene-escalation exhausted: {unrecovered} scene(s) remained below quality floor after escalation to the strongest backend"
                ));
            }
        }
        Ok(())
    }

    /// Document-level translation with CPT collapse-phase routing.
    ///
    /// Translates a cheap NLLB prefix, feeds it to the collapse-phase router, and
    /// if backend-document collapse is certified, reroutes the WHOLE document to
    /// the local LLM rung (avoiding the rest of a doomed NLLB pass). Degrades
    /// safely: when the LLM rung is unavailable (no model / phrase-table / test
    /// mode) or the reroute errors, it does a normal full NLLB translate, so it
    /// is never worse than the pre-CPT behavior.
    ///
    /// NOTE: the reroute path is exercised only with a real LLM model present and
    /// is pending a full live E2E run before release.
    fn translate_with_collapse_routing(
        &self,
        cues: &[SubtitleCue],
        speaker_tags: &[Vec<String>],
        events: &EventSink,
    ) -> Result<Vec<SubtitleCue>, String> {
        use crate::engine::collapse_phase_router::{
            BackendRouteDecision, CollapsePhaseConfig, CollapsePhaseRouter,
        };

        let full_translate = || {
            self.translator
                .translate_all_with_extra_tags(cues, speaker_tags)
                .map_err(|error| error.to_string())
        };

        let cfg = CollapsePhaseConfig::default();
        // Routing only makes sense when the LLM rung can actually be the
        // destination, the tags line up for safe slicing, and there is more
        // document than the prefix to save.
        let llm_available =
            !self.config.force_phrase_table && crate::engine::llm_mt::resolve_llm_paths().is_some();
        if !llm_available || speaker_tags.len() != cues.len() || cues.len() <= cfg.min_prefix_cues {
            return full_translate();
        }

        let m = cfg.min_prefix_cues;
        // Phase 1: cheap NLLB prefix.
        let prefix = self
            .translator
            .translate_all_with_extra_tags(&cues[..m], &speaker_tags[..m])
            .map_err(|error| error.to_string())?;

        // Certify collapse from the prefix, capturing the certificate details.
        let mut router = CollapsePhaseRouter::new();
        let mut certificate = None;
        for cue in &prefix {
            if let BackendRouteDecision::AbortAndRouteWholeDocument {
                dominant_phrase,
                density,
                lower_bound,
                cues_seen,
            } = router.observe(&cue.text, &cfg)
            {
                certificate = Some((dominant_phrase, density, lower_bound, cues_seen));
                break;
            }
        }

        if let Some((dominant_phrase, density, lower_bound, cues_seen)) = certificate {
            // Make the collapse certificate observable (TUI / trace consume this).
            let message = format!(
                "collapse certified at cue {cues_seen}: \"{dominant_phrase}\" {:.0}% of output — rerouting NLLB → local LLM",
                density * 100.0
            );
            events.emit(&serde_json::json!({
                "event": "collapse_route",
                "message": message,
                "from": "nllb",
                "to": "qwen3-4b",
                "dominant_phrase": dominant_phrase,
                "density": density,
                "lower_bound": lower_bound,
                "cue": cues_seen,
                "decision": "reroute_whole_document",
            }));
            // Reroute the whole document to the LLM rung; on any failure fall
            // back to the normal full NLLB translate (never worse than before).
            return match self.translate_whole_document_via_llm(cues) {
                Ok(llm_cues) => Ok(llm_cues),
                Err(error) => {
                    eprintln!(
                        "warning: collapse reroute to LLM failed ({error}); using NLLB output"
                    );
                    full_translate()
                }
            };
        }

        // Not collapsed: translate the remaining cues with NLLB and combine, so
        // the prefix work is reused rather than redone.
        let mut out = prefix;
        let rest = self
            .translator
            .translate_all_with_extra_tags(&cues[m..], &speaker_tags[m..])
            .map_err(|error| error.to_string())?;
        out.extend(rest);
        Ok(out)
    }

    /// Translate the whole document via the LLM rung, in cue chunks so each
    /// request stays within the model's context. One sidecar is spawned and
    /// reused across chunks. Any chunk error (or cue-count mismatch) is an error,
    /// so the caller can fall back to NLLB.
    fn translate_whole_document_via_llm(
        &self,
        cues: &[SubtitleCue],
    ) -> Result<Vec<SubtitleCue>, String> {
        const CHUNK: usize = 24;
        let backend = NeuralSegmentBackend::new(self);
        let mut out = Vec::with_capacity(cues.len());
        for chunk in cues.chunks(CHUNK) {
            let translated = backend.translate_llm(chunk)?;
            if translated.len() != chunk.len() {
                return Err(format!(
                    "llm chunk returned {} cues, expected {}",
                    translated.len(),
                    chunk.len()
                ));
            }
            out.extend(translated);
        }
        Ok(out)
    }

    /// Build a one-off Translator at a specific escalation rung (model + beam).
    /// Used by the per-segment ladder to retry a failing scene at a stronger
    /// backend without disturbing the document-level translator.
    fn build_escalation_translator(
        &self,
        step: &escalation::MtBackendStep,
    ) -> Result<Translator, String> {
        let profile = self.config.quality_profile;
        let base_batch = self
            .config
            .mt_batch_size
            .unwrap_or_else(|| default_mt_batch_for_profile(profile));
        let base_tokens = self
            .config
            .mt_max_batch_tokens
            .unwrap_or_else(|| default_mt_tokens_for_profile(profile));
        let base_oom = self
            .config
            .mt_oom_retries
            .unwrap_or_else(|| default_mt_oom_retries_for_profile(profile));

        Translator::new(TranslatorConfig {
            source_lang: self.config.source_lang.clone(),
            target_lang: self.config.target_lang.clone(),
            offline: self.config.offline,
            force_phrase_table: self.config.force_phrase_table,
            gpu: self.config.gpu,
            require_gpu: false,
            // Walk the ladder's model (guardrail #6 — model comes from the step).
            mt_model: Some(step.model_name.clone()),
            mt_batch_size: Some((base_batch / 2).max(4)),
            mt_max_batch_tokens: Some((base_tokens / 2).max(1024)),
            mt_oom_retries: Some((base_oom + 1).min(8)),
            mt_allow_cpu_fallback: true,
            mt_force_cpu: self.config.mt_force_cpu,
            // P0.2 amortization: escalated scenes route through the persistent
            // MT daemon whenever its script resolves — one Python process and
            // one model load per rung for the whole session, instead of a
            // fresh interpreter + multi-GB model load per failing scene
            // (mt_daemon.py caches loaded models by config signature). When
            // the daemon script is absent (or VOIDEX_MT_SCRIPT overrides the
            // transport), inherit the document-level setting so the Translator
            // never silently degrades to the phrase table.
            mt_daemon: self.config.mt_daemon || mt_daemon_script_available(),
            // The ladder owns the floor decision; the per-segment translator must
            // return its best effort so the scorer (not an early Err) judges it.
            mt_enforce_quality_floor: false,
            // Guardrail #1: apply the rung's beam bump (e.g. Strict 600M→1.3B+beam).
            mt_beam_size: Some(step.beam_size),
            quality_profile: self.config.quality_profile,
        })
        .map_err(|error| error.to_string())
    }

    fn compact_translated_cues(&self, cues: Vec<SubtitleCue>) -> Result<Vec<SubtitleCue>, String> {
        if self.config.quality_profile != QualityProfile::Strict
            || !self.config.target_lang.eq_ignore_ascii_case("en")
            || cues.len() < 2
        {
            return Ok(cues);
        }

        let baseline_semantic = assess_translation_semantics(&cues, &self.config.target_lang);
        let baseline_structural = assess_srt_health(&cues)?;
        let (compacted, stats) = compact_adjacent_cues(&cues, 0.08, 36, 2, 21.0, 6.5)?;
        if compacted.len() >= cues.len() {
            return Ok(cues);
        }

        let compacted_semantic = assess_translation_semantics(&compacted, &self.config.target_lang);
        let compacted_structural = assess_srt_health(&compacted)?;
        let degraded = compacted_semantic.malformed_contraction_ratio
            > baseline_semantic.malformed_contraction_ratio + f64::EPSILON
            || compacted_semantic.anomaly_ratio > baseline_semantic.anomaly_ratio + 0.001
            || compacted_semantic.adjacent_repeat_ratio
                > baseline_semantic.adjacent_repeat_ratio + 0.001
            || compacted_semantic.scene_low_quality_ratio
                > baseline_semantic.scene_low_quality_ratio + 0.002
            || compacted_structural.overlap_ratio > baseline_structural.overlap_ratio
            || compacted_structural.top_line_ratio > baseline_structural.top_line_ratio + 0.003;

        if degraded {
            eprintln!(
                "ibvoid-doom-qlock: cue-compaction rejected (semantic regression) cues={}→{}",
                cues.len(),
                compacted.len()
            );
            return Ok(cues);
        }

        if stats.merged_pairs > 0 || stats.dropped_duplicates > 0 {
            eprintln!(
                "ibvoid-doom-qlock: cue-compaction merged={} deduped={} cues={}→{}",
                stats.merged_pairs,
                stats.dropped_duplicates,
                cues.len(),
                compacted.len()
            );
        }
        Ok(compacted)
    }

    fn enforce_discourse_consistency(
        &self,
        source_cues: &[SubtitleCue],
        translated_cues: &mut [SubtitleCue],
        speakers: &[Option<String>],
    ) {
        if !self.config.target_lang.eq_ignore_ascii_case("en")
            || source_cues.len() != translated_cues.len()
            || translated_cues.len() < 4
        {
            return;
        }

        let stats = apply_source_phrase_consistency(source_cues, translated_cues);
        if stats.rewritten_cues > 0 {
            eprintln!(
                "ibvoid-doom-qlock: discourse-consistency clusters={} rewrites={}",
                stats.source_clusters, stats.rewritten_cues
            );
        }

        let speaker_stats =
            apply_source_phrase_consistency_by_speaker(source_cues, translated_cues, speakers);
        if speaker_stats.rewritten_cues > 0 {
            eprintln!(
                "ibvoid-doom-qlock: discourse-consistency-speaker speakers={} clusters={} rewrites={}",
                speaker_stats.speakers, speaker_stats.source_clusters, speaker_stats.rewritten_cues
            );
        }
    }

    fn rescue_low_quality_source_transcription(
        &self,
        cues: &mut Vec<SubtitleCue>,
        audio_input: Option<&Path>,
        source_confidence: Option<&[Option<CueAsrConfidence>]>,
    ) -> Result<(), String> {
        if self.config.quality_profile != QualityProfile::Strict || cues.is_empty() {
            return Ok(());
        }

        let Some(audio_input) = audio_input else {
            return Ok(());
        };
        if !audio_input.exists() {
            return Ok(());
        }

        let retry_transcriber = self.build_source_scene_retry_transcriber()?;
        let mut improved = 0usize;

        let mut low_confidence_spans = source_confidence
            .map(|scores| {
                collect_low_confidence_cue_spans(cues, scores, self.config.quality_profile)
            })
            .unwrap_or_default();
        if !low_confidence_spans.is_empty() {
            low_confidence_spans.sort_by(|a, b| {
                let lhs = b.floor - b.score;
                let rhs = a.floor - a.score;
                lhs.total_cmp(&rhs)
            });
            let conf_retry_limit = std::env::var("VOIDEX_LOW_CONF_RETRY_LIMIT")
                .ok()
                .and_then(|raw| raw.parse::<usize>().ok())
                .unwrap_or(8)
                .max(1);
            low_confidence_spans.truncate(low_confidence_spans.len().min(conf_retry_limit));
            low_confidence_spans.sort_by_key(|span| std::cmp::Reverse(span.start));

            eprintln!(
                "ibvoid-doom-qlock: strict confidence-rescue retrying {}/{} low-confidence spans",
                low_confidence_spans.len(),
                source_confidence
                    .map(|scores| collect_low_confidence_cue_spans(
                        cues,
                        scores,
                        self.config.quality_profile
                    )
                    .len())
                    .unwrap_or(0)
            );

            for span in low_confidence_spans {
                let old_slice = &cues[span.start..span.end];
                let Some((span_start, span_end)) = scene_time_span(old_slice) else {
                    continue;
                };
                if span_end <= span_start + 0.10 {
                    continue;
                }

                let start = (span_start - 0.30).max(0.0);
                let end = span_end + 0.40;
                let work_dir = create_temp_rescue_dir(span.start)?;
                let clip_wav = work_dir.join("source_conf_span.wav");
                extract_audio_segment_to_wav(audio_input, &clip_wav, start, end)?;
                let retried_srt = retry_transcriber
                    .transcribe_wav_to_srt(&clip_wav)
                    .map_err(|error| error.to_string())?;
                let retried_cues =
                    parse_srt_file(&retried_srt).map_err(|error| error.to_string())?;
                if retried_cues.is_empty() {
                    continue;
                }

                let retried_confidence =
                    load_cue_asr_confidence_from_whisper_json(&retried_srt, &retried_cues);
                let shifted = shift_cues_by_offset(&retried_cues, start)?;
                let old_struct_score = source_scene_quality_score(old_slice);
                let new_struct_score = source_scene_quality_score(&shifted);
                let new_health = assess_srt_health(&shifted)?;
                let new_conf_score = retried_confidence
                    .as_deref()
                    .and_then(mean_confidence_score)
                    .unwrap_or(new_struct_score);
                let conf_gain = new_conf_score - span.score;
                let struct_gain = new_struct_score - old_struct_score;

                if new_health.is_pathological(QualityProfile::Strict)
                    || conf_gain <= 0.06
                    || struct_gain <= 0.0
                {
                    continue;
                }

                cues.splice(span.start..span.end, shifted.into_iter());
                improved += 1;
                eprintln!(
                    "ibvoid-doom-qlock: confidence-rescue improved span {}-{} conf {:.3} -> {:.3}",
                    span.start, span.end, span.score, new_conf_score
                );
            }
        }

        let mut low_scenes = collect_low_quality_source_scene_ranges(cues);
        if low_scenes.is_empty() {
            if improved == 0 {
                eprintln!("ibvoid-doom-qlock: source-rescue found no beneficial rewrites.");
            }
            return Ok(());
        }

        low_scenes.sort_by(|a, b| {
            let lhs = b.floor - b.score;
            let rhs = a.floor - a.score;
            lhs.total_cmp(&rhs)
        });

        let retry_limit = std::env::var("VOIDEX_SOURCE_SCENE_RETRY_LIMIT")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        low_scenes.truncate(low_scenes.len().min(retry_limit));
        low_scenes.sort_by_key(|scene| std::cmp::Reverse(scene.start));

        eprintln!(
            "ibvoid-doom-qlock: strict source-rescue retrying {}/{} low-quality source scenes",
            low_scenes.len(),
            collect_low_quality_source_scene_ranges(cues).len()
        );

        for scene in low_scenes {
            let old_scene = &cues[scene.start..scene.end];
            let Some((scene_start, scene_end)) = scene_time_span(old_scene) else {
                continue;
            };
            if scene_end <= scene_start + 0.15 {
                continue;
            }

            let start = (scene_start - 0.25).max(0.0);
            let end = scene_end + 0.35;
            let work_dir = create_temp_rescue_dir(scene.start)?;
            let clip_wav = work_dir.join("source_scene.wav");
            extract_audio_segment_to_wav(audio_input, &clip_wav, start, end)?;
            let retried_srt = retry_transcriber
                .transcribe_wav_to_srt(&clip_wav)
                .map_err(|error| error.to_string())?;
            let retried_cues = parse_srt_file(&retried_srt).map_err(|error| error.to_string())?;
            if retried_cues.is_empty() {
                continue;
            }

            let shifted = shift_cues_by_offset(&retried_cues, start)?;
            let old_score = source_scene_quality_score(old_scene);
            let new_score = source_scene_quality_score(&shifted);
            let new_health = assess_srt_health(&shifted)?;
            if new_score <= old_score + 0.03 || new_health.is_pathological(QualityProfile::Strict) {
                continue;
            }

            cues.splice(scene.start..scene.end, shifted.into_iter());
            improved += 1;
            eprintln!(
                "ibvoid-doom-qlock: source-rescue improved scene {}-{} score {:.3} -> {:.3}",
                scene.start, scene.end, old_score, new_score
            );
        }

        if improved == 0 {
            eprintln!("ibvoid-doom-qlock: source-rescue found no beneficial rewrites.");
        }
        Ok(())
    }

    fn build_source_scene_retry_transcriber(&self) -> Result<Transcriber, String> {
        let mut config = Self::make_transcribe_config(&self.config, true);
        config.quality_profile = QualityProfile::Strict;
        config.vad = false;
        config.require_gpu = false;
        let transcriber = Transcriber::new(config)
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "failed to initialize source rescuer".to_string())?;
        Ok(transcriber)
    }

    fn rescue_transcribe_video(
        &self,
        input: &Path,
        events: &EventSink,
    ) -> Result<TranscriptionResult, String> {
        if self.config.parallel {
            let (srt_path, audio_wav_path) = self.parallel_transcribe(input, events)?;
            let Some(audio_wav_path) = audio_wav_path else {
                return Err(
                    "parallel rescue transcription did not return an audio path".to_string()
                );
            };
            return Ok(TranscriptionResult {
                srt_path,
                audio_wav_path,
            });
        }

        let transcriber = Transcriber::new(Self::make_transcribe_config(&self.config, true))
            .map_err(|error| error.to_string())?
            .ok_or_else(|| "failed to initialize rescue transcriber".to_string())?;
        transcriber
            .transcribe_video_to_srt(input)
            .map_err(|error| error.to_string())
    }

    fn resolve_subtitle_source(
        &self,
        input: &Path,
        events: &EventSink,
    ) -> Result<(PathBuf, Option<PathBuf>), String> {
        if !input.exists() {
            return Err(format!("input does not exist: {}", input.display()));
        }

        if is_srt_path(input) {
            if looks_like_simulated_placeholder_srt(input) {
                eprintln!(
                    "warning: input subtitle looks like placeholder output (\"Simulated...\"). \
provide the source video and re-run with --transcribe to generate real subtitles from audio."
                );
            }
            return Ok((input.to_path_buf(), None));
        }

        // Parallel transcription path
        if self.config.parallel && self.transcriber.is_some() {
            return self.parallel_transcribe(input, events);
        }

        // ── Serial transcription path (original) ──
        if let Some(transcriber) = &self.transcriber {
            let result = transcriber
                .transcribe_video_to_srt(input)
                .map_err(|error| error.to_string())?;
            return Ok((result.srt_path, Some(result.audio_wav_path)));
        }

        let sidecar = input.with_extension("srt");
        if sidecar.exists() {
            if self.config.quality_profile == QualityProfile::Strict {
                return Err(format!(
                    "strict profile requires audio-first transcription for video inputs; re-run without --no-transcribe for {}",
                    input.display()
                ));
            }
            if looks_like_simulated_placeholder_srt(&sidecar) {
                eprintln!(
                    "warning: sidecar subtitle looks like placeholder output (\"Simulated...\"). \
re-run with --transcribe + a local whisper.cpp model to generate real subtitles from audio."
                );
            }
            let audio_for_verify = if self.config.verify {
                Some(input.to_path_buf())
            } else {
                None
            };
            return Ok((sidecar, audio_for_verify));
        }

        // Verification-only fallback.
        if self.config.verify {
            let existing = output_path_for_target_lang(input, &self.config.target_lang)?;
            if existing.exists() {
                eprintln!(
                    "note: no sidecar .srt found for {}; using existing subtitle for verification: {}",
                    input.display(),
                    existing.display()
                );
                return Ok((existing, Some(input.to_path_buf())));
            }
        }

        Err(format!(
            "no subtitle source found for {} (expected .srt input, sidecar .srt, or --transcribe)",
            input.display()
        ))
    }

    /// Parallel transcription: chunk → parallel whisper → stitch.
    fn parallel_transcribe(
        &self,
        video: &Path,
        events: &EventSink,
    ) -> Result<(PathBuf, Option<PathBuf>), String> {
        let checkpoint_dir = checkpoint_dir_for(video)?;
        let temp_dir = checkpoint_dir.join("work");
        std::fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
        let checkpoint_path = checkpoint_dir.join("run_checkpoint.json");
        let wav_path = temp_dir.join("audio.wav");
        extract_audio_to_wav(video, &wav_path)?;

        let duration = crate::engine::transcribe::ffprobe_duration_seconds_pub(&wav_path)
            .map_err(|error| error.to_string())?;

        eprintln!(
            "parallel: audio duration {:.1}s, chunking into ~{:.0}s pieces...",
            duration, self.config.chunk_duration_secs
        );

        let chunker_config = ChunkerConfig {
            target_chunk_secs: self.config.chunk_duration_secs,
            min_silence_gap: 0.4,
            overlap_secs: 2.0,
            vad_threshold_db: self.config.vad_threshold_db,
            vad_min_silence: self.config.vad_min_silence,
            vad_pad: self.config.vad_pad,
        };

        let chunks = chunk_audio(&wav_path, &temp_dir, duration, &chunker_config)?;

        // VRAM-aware worker cap: Python whisper loads the full model per-process
        // (~1.5-3GB for medium/large). Multiple workers on a single GPU cause
        // VRAM thrashing that drops throughput 5-10x. Cap to 1 worker per GPU,
        // allow multi-worker only on CPU where RAM is plentiful.
        let effective_workers = if self.config.gpu {
            // On GPU: 1 worker gets full VRAM → 60-80fps.
            // Multiple workers split VRAM → 10-15fps each (net slower).
            self.config.max_workers.min(1)
        } else {
            self.config.max_workers
        };

        eprintln!(
            "parallel: {} chunks created, launching {} worker(s){}...",
            chunks.len(),
            effective_workers,
            if self.config.gpu && self.config.max_workers > 1 {
                " (capped to 1 for GPU VRAM efficiency)"
            } else {
                ""
            },
        );

        let transcribe_config = Self::make_transcribe_config(&self.config, true);

        let chunk_results = parallel_transcribe(
            &chunks,
            &transcribe_config,
            effective_workers,
            Some(checkpoint_path),
            events.clone(),
        )
        .map_err(|error| error.to_string())?;
        let merged_cues = stitch_chunks(&chunk_results).map_err(|error| error.to_string())?;

        // Coverage gate: detect catastrophic data loss from chunk timeouts.
        // If the output cues cover less than a threshold of the total audio
        // duration, the result is incomplete and shouldn't silently "pass."
        let failed_chunks = chunk_results.iter().filter(|c| c.cues.is_empty()).count();
        let total_chunks = chunk_results.len();
        let coverage_ratio = if total_chunks > 0 {
            (total_chunks - failed_chunks) as f64 / total_chunks as f64
        } else {
            0.0
        };
        if coverage_ratio < 0.5 {
            let msg = format!(
                "parallel transcription coverage too low: only {}/{} chunks succeeded ({:.0}% coverage). \
                 This usually means whisper timed out on CPU. Try: (1) install CUDA/Triton for GPU acceleration, \
                 (2) use whisper.cpp backend (--whisper-bin), or (3) increase chunk duration (--chunk-duration).",
                total_chunks - failed_chunks, total_chunks, coverage_ratio * 100.0
            );
            if self.config.quality_profile == QualityProfile::Strict {
                return Err(msg);
            }
            eprintln!("ERROR: {msg}");
        } else if failed_chunks > 0 {
            eprintln!(
                "warning: {failed_chunks}/{total_chunks} chunks failed ({:.0}% coverage). \
                 Output may be incomplete. Re-run with --gpu (with CUDA installed) or use whisper.cpp for faster processing.",
                coverage_ratio * 100.0
            );
        }

        eprintln!(
            "parallel: stitched {} total cues from {} chunks ({} failed)",
            merged_cues.len(),
            chunks.len(),
            failed_chunks,
        );

        // Write the raw (untranslated) SRT for the stitch result.
        let srt_path = output_path_for_target_lang(video, &self.config.source_lang)
            .map_err(|e| format!("resolve output path: {e}"))?;

        write_srt_file(&srt_path, &merged_cues).map_err(|error| error.to_string())?;
        if let Err(error) = write_parallel_confidence_sidecar(&chunks, &srt_path) {
            eprintln!(
                "warning: failed to stitch parallel confidence sidecar ({}): {error}",
                srt_path.display()
            );
        }

        Ok((srt_path, Some(wav_path)))
    }
}

/// Production wiring of the escalation backend seam: builds a real NLLB
/// Translator for the requested rung and translates the segment. Tests inject a
/// scripted fake instead (see `escalation::tests`), so the ladder orchestration
/// is exercised without NLLB / GPU / models.
///
/// P0.2 amortization: the Translator for each rung is built once and reused for
/// every failing scene routed to that rung. Combined with daemon routing in
/// `build_escalation_translator`, the rung's model loads once per session
/// instead of once per scene.
struct NeuralSegmentBackend<'a> {
    pipeline: &'a SubtitlePipeline,
    /// One cached NLLB Translator per rung label (e.g. "nllb-1.3B").
    translators: std::cell::RefCell<std::collections::HashMap<&'static str, Translator>>,
    /// The LLM rung's sidecar, built lazily on first use and kept alive (one
    /// llama-server load per session, not per scene).
    llm: std::cell::RefCell<Option<crate::engine::llm_mt::LlmTranslator>>,
}

impl<'a> NeuralSegmentBackend<'a> {
    fn new(pipeline: &'a SubtitlePipeline) -> Self {
        Self {
            pipeline,
            translators: std::cell::RefCell::new(std::collections::HashMap::new()),
            llm: std::cell::RefCell::new(None),
        }
    }

    /// NLLB rung: one cached Translator per rung label, reused across scenes
    /// (P0.2 amortization).
    fn translate_nllb(
        &self,
        step: &escalation::MtBackendStep,
        source: &[SubtitleCue],
    ) -> Result<Vec<SubtitleCue>, String> {
        use std::collections::hash_map::Entry;
        let mut cache = self.translators.borrow_mut();
        let translator = match cache.entry(step.label) {
            Entry::Occupied(entry) => entry.into_mut(),
            // A failed build inserts nothing, so a later scene retries cleanly
            // rather than hitting a poisoned cache slot.
            Entry::Vacant(slot) => slot.insert(self.pipeline.build_escalation_translator(step)?),
        };
        translator
            .translate_all(source)
            .map_err(|error| error.to_string())
    }

    /// LLM rung: lazily spawn the llama-server sidecar once, then reuse it for
    /// every escalated scene this session.
    fn translate_llm(&self, source: &[SubtitleCue]) -> Result<Vec<SubtitleCue>, String> {
        let mut slot = self.llm.borrow_mut();
        if slot.is_none() {
            let (binary, model) = crate::engine::llm_mt::resolve_llm_paths()
                .ok_or_else(|| "llm backend unavailable (binary/model not found)".to_string())?;
            // Feed the persisted character glossary into the prompt: rescued
            // translations stay consistent with names the document — and earlier
            // episodes — already settled on (cross-run self-improvement).
            let glossary = crate::engine::character_glossary::CharacterGlossary::load_default()
                .canonical_names();
            *slot = Some(crate::engine::llm_mt::LlmTranslator::new(
                binary,
                model,
                &self.pipeline.config.source_lang,
                &self.pipeline.config.target_lang,
                glossary,
            )?);
        }
        slot.as_ref()
            .expect("llm slot populated above")
            .translate_all(source)
    }
}

impl escalation::SegmentBackend for NeuralSegmentBackend<'_> {
    fn translate_segment(
        &self,
        step: &escalation::MtBackendStep,
        source: &[SubtitleCue],
    ) -> Result<Vec<SubtitleCue>, String> {
        match step.kind {
            escalation::BackendKind::Nllb => self.translate_nllb(step, source),
            escalation::BackendKind::Llm => self.translate_llm(source),
        }
    }
}

/// Production scene scorer: reuses the established scene-quality score and
/// semantic penalty so the ladder's acceptance test matches the rest of the
/// pipeline's quality model (no divergent heuristic).
struct RealSceneScorer;

impl escalation::SceneScorer for RealSceneScorer {
    fn score(&self, translated: &[SubtitleCue]) -> (f64, f64) {
        let (_difficulty, score) = scene_quality_for_slice(translated);
        let health = assess_translation_semantics(translated, "en");
        (score, scene_semantic_penalty(&health))
    }
}

#[cfg(test)]
mod tests;
