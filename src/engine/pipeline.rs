// Sub-Zero GOD-TIER Pipeline (Theorem 4: Dual-Pipeline Convergence).
//
// Composes the Speed Pipeline (parallel chunked transcription) with the
// Accuracy Pipeline (context-aware neural translation + post-processing).

use crate::engine::chunker::{chunk_audio, AudioChunk, ChunkerConfig};
use crate::engine::parallel::parallel_transcribe;
use crate::engine::srt::{parse_srt_file, write_srt_file, SubtitleCue};
use crate::engine::stitcher::stitch_chunks;
use crate::engine::transcribe::{
    QualityProfile, TranscribeConfig, Transcriber, TranscriptionResult,
};
use crate::engine::translate::{Translator, TranslatorConfig};
use std::collections::HashMap;
use std::ffi::OsString;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub source_lang: String,
    pub target_lang: String,
    pub offline: bool,
    pub transcribe: bool,
    pub whisper_bin: Option<PathBuf>,
    pub whisper_model: Option<PathBuf>,
    pub whisper_args: Vec<String>,
    pub skip_existing: bool,
    pub vad: bool,
    pub vad_threshold_db: f64,
    pub vad_min_silence: f64,
    pub vad_pad: f64,
    pub verify: bool,
    pub verify_min_speech_overlap: f64,
    pub gpu: bool,
    pub require_gpu: bool,
    // ── GOD-TIER additions ──
    /// Enable parallel chunked transcription (speed pipeline).
    pub parallel: bool,
    /// Max parallel whisper workers.
    pub max_workers: usize,
    /// Target chunk duration in seconds (default 300 = 5 min).
    pub chunk_duration_secs: f64,
    /// Force phrase-table translator instead of neural MT.
    pub force_phrase_table: bool,
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
    /// If sidecar subtitles look degenerate, auto-transcribe from video audio.
    pub auto_repair_sidecar: bool,
    /// Quality/latency operating mode.
    pub quality_profile: QualityProfile,
}

pub struct SubtitlePipeline {
    config: PipelineConfig,
    translator: Translator,
    transcriber: Option<Transcriber>,
}

impl SubtitlePipeline {
    pub fn new(config: PipelineConfig) -> Result<Self, String> {
        let transcriber =
            Transcriber::new(Self::make_transcribe_config(&config, config.transcribe))?;
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
            quality_profile: config.quality_profile,
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
            vad: config.vad,
            vad_threshold_db: config.vad_threshold_db,
            vad_min_silence: config.vad_min_silence,
            vad_pad: config.vad_pad,
            gpu: config.gpu,
            require_gpu: config.require_gpu,
            quality_profile: config.quality_profile,
        }
    }

    pub fn process_input(&self, input: &Path) -> Result<PathBuf, String> {
        let output = output_path_for_target_lang(input, &self.config.target_lang)?;
        if self.config.skip_existing && output.exists() {
            return Ok(output);
        }

        let (mut source_srt, mut audio_for_verify) = self.resolve_subtitle_source(input)?;
        let mut cues =
            parse_srt_file(&source_srt).map_err(|e| format!("{}: {e}", source_srt.display()))?;
        let mut source_confidence = load_cue_asr_confidence_from_whisper_json(&source_srt, &cues);

        if is_video_sidecar_source(input, &source_srt) {
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
                let rescue = self.rescue_transcribe_video(input)?;
                source_srt = rescue.srt_path;
                audio_for_verify = Some(rescue.audio_wav_path);
                cues = parse_srt_file(&source_srt)
                    .map_err(|e| format!("{}: {e}", source_srt.display()))?;
                source_confidence = load_cue_asr_confidence_from_whisper_json(&source_srt, &cues);
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

        self.rescue_low_quality_source_transcription(
            &mut cues,
            audio_for_verify.as_deref(),
            source_confidence.as_deref(),
        )?;
        self.enforce_source_quality_gate(&cues)?;

        // Translate — batched (neural) or per-cue (phrase-table).
        let mut translated = self.translator.translate_all(&cues)?;
        self.rescue_low_quality_scene_translations(&cues, &mut translated)?;
        self.enforce_discourse_consistency(&cues, &mut translated);
        translated = self.compact_translated_cues(translated)?;
        self.enforce_translated_quality_gate(&translated)?;

        write_srt_file(&output, &translated).map_err(|e| format!("{}: {e}", output.display()))?;
        self.write_metadata_sidecar(input, &output, &translated)?;

        if self.config.verify {
            if let Some(audio_path) = audio_for_verify {
                let report = verify_srt_against_audio(
                    &output,
                    &audio_path,
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

        Ok(output)
    }

    fn write_metadata_sidecar(
        &self,
        input: &Path,
        output: &Path,
        translated: &[SubtitleCue],
    ) -> Result<(), String> {
        let structural = assess_srt_health(translated)?;
        let semantic = assess_translation_semantics(translated, &self.config.target_lang);
        let scene_metrics = build_scene_metadata(translated);
        let checkpoint_summary = load_checkpoint_summary(input);
        let metadata_path = metadata_sidecar_path(input)?;
        let payload = serde_json::json!({
            "version": "1.0",
            "algorithm": "IBVoid DOOM-QLOCK",
            "source_file": input.display().to_string(),
            "output_file": output.display().to_string(),
            "source_language": self.config.source_lang,
            "target_language": self.config.target_lang,
            "quality_profile": self.config.quality_profile.as_str(),
            "generated_at_epoch_secs": now_epoch_secs(),
            "plan_used": {
                "parallel": self.config.parallel,
                "workers": self.config.max_workers,
                "chunk_duration_secs": self.config.chunk_duration_secs,
                "mt_batch_size": self.config.mt_batch_size,
                "mt_max_batch_tokens": self.config.mt_max_batch_tokens,
                "mt_oom_retries": self.config.mt_oom_retries,
                "mt_allow_cpu_fallback": self.config.mt_allow_cpu_fallback,
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
                .map(|summary| summary.get("recovery_events").cloned().unwrap_or(serde_json::json!([])))
                .unwrap_or(serde_json::json!([])),
            "checkpoint": checkpoint_summary.unwrap_or(serde_json::json!({
                "status": "none",
                "completed_chunks": 0,
                "failed_chunks": 0,
                "recovery_events": [],
            })),
            "warnings": build_metadata_warnings(&semantic),
            "verdict": {
                "pass": !semantic.is_pathological(self.config.quality_profile),
                "reason": if semantic.is_pathological(self.config.quality_profile) {
                    semantic.summary()
                } else {
                    "quality gate passed".to_string()
                }
            }
        });
        let serialized = serde_json::to_string_pretty(&payload)
            .map_err(|e| format!("{} serialize metadata: {e}", metadata_path.display()))?;
        std::fs::write(&metadata_path, serialized)
            .map_err(|e| format!("{}: {e}", metadata_path.display()))
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
        let health = assess_translation_semantics(cues, &self.config.target_lang);
        if !health.is_pathological(self.config.quality_profile) {
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

    fn rescue_low_quality_scene_translations(
        &self,
        source_cues: &[SubtitleCue],
        translated: &mut [SubtitleCue],
    ) -> Result<(), String> {
        if self.config.quality_profile != QualityProfile::Strict
            || !self.config.target_lang.eq_ignore_ascii_case("en")
            || source_cues.len() != translated.len()
        {
            return Ok(());
        }

        let mut low_scenes = collect_low_quality_scene_ranges(translated);
        if low_scenes.is_empty() {
            return Ok(());
        }

        low_scenes.sort_by(|a, b| {
            let lhs = b.floor - b.score;
            let rhs = a.floor - a.score;
            lhs.total_cmp(&rhs)
        });

        let retry_limit = std::env::var("SUB_ZERO_SCENE_RETRY_LIMIT")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(6)
            .max(1);
        let retry_count = low_scenes.len().min(retry_limit);
        if retry_count == 0 {
            return Ok(());
        }

        eprintln!(
            "ibvoid-doom-qlock: strict scene-rescue retrying {retry_count}/{} low-quality scenes",
            low_scenes.len()
        );

        let retry_translator = self.build_scene_retry_translator()?;
        let mut improved = 0usize;
        for scene in low_scenes.into_iter().take(retry_count) {
            let source_scene = &source_cues[scene.start..scene.end];
            let retried = retry_translator.translate_all(source_scene)?;
            if retried.len() != source_scene.len() {
                eprintln!(
                    "warning: scene-rescue skipped scene {}-{} due to cue count mismatch (expected {}, got {})",
                    scene.start,
                    scene.end,
                    source_scene.len(),
                    retried.len()
                );
                continue;
            }

            let (_, original_score) = scene_quality_for_slice(&translated[scene.start..scene.end]);
            let (_, retry_score) = scene_quality_for_slice(&retried);
            let original_health =
                assess_translation_semantics(&translated[scene.start..scene.end], "en");
            let retry_health = assess_translation_semantics(&retried, "en");
            let original_penalty = scene_semantic_penalty(&original_health);
            let retry_penalty = scene_semantic_penalty(&retry_health);

            if retry_score <= original_score + 0.01
                || retry_penalty > original_penalty - 0.01
                || retry_health.malformed_contraction_ratio
                    > original_health.malformed_contraction_ratio + f64::EPSILON
            {
                continue;
            }

            for (dst, src) in translated[scene.start..scene.end]
                .iter_mut()
                .zip(retried.into_iter())
            {
                dst.text = src.text;
            }
            improved += 1;
            eprintln!(
                "ibvoid-doom-qlock: scene-rescue improved scene {}-{} score {:.3} -> {:.3}",
                scene.start, scene.end, original_score, retry_score
            );
        }

        if improved == 0 {
            eprintln!("ibvoid-doom-qlock: scene-rescue found no beneficial rewrites.");
        }
        Ok(())
    }

    fn build_scene_retry_translator(&self) -> Result<Translator, String> {
        let base_batch = self
            .config
            .mt_batch_size
            .unwrap_or(default_mt_batch_for_profile(self.config.quality_profile));
        let base_tokens = self
            .config
            .mt_max_batch_tokens
            .unwrap_or(default_mt_tokens_for_profile(self.config.quality_profile));
        let base_oom = self
            .config
            .mt_oom_retries
            .unwrap_or(default_mt_oom_retries_for_profile(
                self.config.quality_profile,
            ));

        Translator::new(TranslatorConfig {
            source_lang: self.config.source_lang.clone(),
            target_lang: self.config.target_lang.clone(),
            offline: self.config.offline,
            force_phrase_table: self.config.force_phrase_table,
            gpu: self.config.gpu,
            require_gpu: false,
            mt_model: self.config.mt_model.clone(),
            mt_batch_size: Some((base_batch / 2).max(4)),
            mt_max_batch_tokens: Some((base_tokens / 2).max(1024)),
            mt_oom_retries: Some((base_oom + 1).min(8)),
            mt_allow_cpu_fallback: true,
            quality_profile: QualityProfile::Strict,
        })
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
            let conf_retry_limit = std::env::var("SUB_ZERO_LOW_CONF_RETRY_LIMIT")
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
                let retried_srt = retry_transcriber.transcribe_wav_to_srt(&clip_wav)?;
                let retried_cues = parse_srt_file(&retried_srt)
                    .map_err(|e| format!("{}: {e}", retried_srt.display()))?;
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

        let retry_limit = std::env::var("SUB_ZERO_SOURCE_SCENE_RETRY_LIMIT")
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
            let retried_srt = retry_transcriber.transcribe_wav_to_srt(&clip_wav)?;
            let retried_cues = parse_srt_file(&retried_srt)
                .map_err(|e| format!("{}: {e}", retried_srt.display()))?;
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
        let transcriber = Transcriber::new(config)?
            .ok_or_else(|| "failed to initialize source rescuer".to_string())?;
        Ok(transcriber)
    }

    fn rescue_transcribe_video(&self, input: &Path) -> Result<TranscriptionResult, String> {
        if self.config.parallel {
            let (srt_path, audio_wav_path) = self.parallel_transcribe(input)?;
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

        let transcriber = Transcriber::new(Self::make_transcribe_config(&self.config, true))?
            .ok_or_else(|| "failed to initialize rescue transcriber".to_string())?;
        transcriber.transcribe_video_to_srt(input)
    }

    fn resolve_subtitle_source(&self, input: &Path) -> Result<(PathBuf, Option<PathBuf>), String> {
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

        // ── Parallel transcription path (GOD-TIER speed pipeline) ──
        if self.config.parallel && self.transcriber.is_some() {
            return self.parallel_transcribe(input);
        }

        // ── Serial transcription path (original) ──
        if let Some(transcriber) = &self.transcriber {
            let result = transcriber.transcribe_video_to_srt(input)?;
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

    /// GOD-TIER parallel transcription: chunk → parallel whisper → stitch.
    fn parallel_transcribe(&self, video: &Path) -> Result<(PathBuf, Option<PathBuf>), String> {
        let checkpoint_dir = checkpoint_dir_for(video)?;
        let temp_dir = checkpoint_dir.join("work");
        std::fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
        let checkpoint_path = checkpoint_dir.join("run_checkpoint.json");
        let wav_path = temp_dir.join("audio.wav");
        extract_audio_to_wav(video, &wav_path)?;

        let duration = crate::engine::transcribe::ffprobe_duration_seconds_pub(&wav_path)?;

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
        eprintln!(
            "parallel: {} chunks created, launching {} workers...",
            chunks.len(),
            self.config.max_workers
        );

        let transcribe_config = Self::make_transcribe_config(&self.config, true);

        let chunk_results = parallel_transcribe(
            &chunks,
            &transcribe_config,
            self.config.max_workers,
            Some(checkpoint_path),
        )?;
        let merged_cues = stitch_chunks(&chunk_results)?;

        eprintln!(
            "parallel: stitched {} total cues from {} chunks",
            merged_cues.len(),
            chunks.len()
        );

        // Write the raw (untranslated) SRT for the stitch result.
        let srt_path = output_path_for_target_lang(video, &self.config.source_lang)
            .map_err(|e| format!("resolve output path: {e}"))?;

        write_srt_file(&srt_path, &merged_cues).map_err(|e| format!("write stitched SRT: {e}"))?;
        if let Err(error) = write_parallel_confidence_sidecar(&chunks, &srt_path) {
            eprintln!(
                "warning: failed to stitch parallel confidence sidecar ({}): {error}",
                srt_path.display()
            );
        }

        Ok((srt_path, Some(wav_path)))
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn output_path_for_target_lang(input: &Path, target_lang: &str) -> Result<PathBuf, String> {
    let stem = input
        .file_stem()
        .ok_or_else(|| format!("invalid input filename: {}", input.display()))?;

    let mut file_name = OsString::from(stem);
    file_name.push(format!(".{target_lang}.srt"));
    Ok(input.with_file_name(file_name))
}

fn metadata_sidecar_path(input: &Path) -> Result<PathBuf, String> {
    let stem = input
        .file_stem()
        .ok_or_else(|| format!("invalid input filename: {}", input.display()))?;
    let mut file_name = OsString::from(stem);
    file_name.push(".sub-zero.json");
    Ok(input.with_file_name(file_name))
}

fn write_parallel_confidence_sidecar(
    chunks: &[AudioChunk],
    output_srt: &Path,
) -> Result<(), String> {
    let mut merged_segments = Vec::<serde_json::Value>::new();
    for chunk in chunks {
        let json_path = chunk.wav_path.with_extension("json");
        if !json_path.is_file() {
            continue;
        }
        let raw = std::fs::read_to_string(&json_path)
            .map_err(|e| format!("{}: {e}", json_path.display()))?;
        let parsed: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("{} parse error: {e}", json_path.display()))?;
        let Some(segments) = parsed.get("segments").and_then(serde_json::Value::as_array) else {
            continue;
        };

        for segment in segments {
            let Some(start) = segment.get("start").and_then(serde_json::Value::as_f64) else {
                continue;
            };
            let Some(end) = segment.get("end").and_then(serde_json::Value::as_f64) else {
                continue;
            };
            if end <= start {
                continue;
            }
            let mut payload = serde_json::Map::<String, serde_json::Value>::new();
            payload.insert(
                "start".to_string(),
                serde_json::json!(start + chunk.start_sec),
            );
            payload.insert("end".to_string(), serde_json::json!(end + chunk.start_sec));
            if let Some(value) = segment.get("avg_logprob") {
                payload.insert("avg_logprob".to_string(), value.clone());
            }
            if let Some(value) = segment.get("no_speech_prob") {
                payload.insert("no_speech_prob".to_string(), value.clone());
            }
            if let Some(value) = segment.get("compression_ratio") {
                payload.insert("compression_ratio".to_string(), value.clone());
            }
            merged_segments.push(serde_json::Value::Object(payload));
        }
    }

    if merged_segments.is_empty() {
        return Ok(());
    }
    merged_segments.sort_by(|a, b| {
        let lhs = a
            .get("start")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let rhs = b
            .get("start")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0);
        lhs.partial_cmp(&rhs).unwrap_or(std::cmp::Ordering::Equal)
    });

    let sidecar_path = output_srt.with_extension("json");
    let payload = serde_json::json!({ "segments": merged_segments });
    let serialized = serde_json::to_string_pretty(&payload)
        .map_err(|e| format!("{} serialize error: {e}", sidecar_path.display()))?;
    std::fs::write(&sidecar_path, serialized)
        .map_err(|e| format!("{}: {e}", sidecar_path.display()))
}

fn build_scene_metadata(cues: &[SubtitleCue]) -> Vec<serde_json::Value> {
    let scenes = split_scenes(cues);
    scenes
        .iter()
        .enumerate()
        .map(|(index, scene)| {
            let (difficulty, score) = scene_quality(scene);
            let floor = scene_floor_for_difficulty(difficulty);
            serde_json::json!({
                "scene": index + 1,
                "cue_count": scene.len(),
                "difficulty": difficulty,
                "score": score,
                "floor": floor,
                "pass": score >= floor,
            })
        })
        .collect()
}

fn load_checkpoint_summary(input: &Path) -> Option<serde_json::Value> {
    let dir = checkpoint_dir_for(input).ok()?;
    let path = dir.join("run_checkpoint.json");
    if !path.is_file() {
        return None;
    }
    let raw = std::fs::read_to_string(&path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let completed = value
        .get("completed")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let failed = value
        .get("failed")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    let recovery_events = value
        .get("failed")
        .and_then(|v| v.as_array())
        .map(|entries| {
            entries
                .iter()
                .map(|entry| {
                    serde_json::json!({
                        "chunk": entry.get("chunk_index").cloned().unwrap_or(serde_json::json!(null)),
                        "event": "chunk_failure",
                        "reason": entry.get("reason").cloned().unwrap_or(serde_json::json!("unknown")),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Some(serde_json::json!({
        "status": "available",
        "path": path.display().to_string(),
        "completed_chunks": completed,
        "failed_chunks": failed,
        "recovery_events": recovery_events,
    }))
}

fn build_metadata_warnings(semantic: &TranslationSemanticHealth) -> Vec<String> {
    let mut warnings = Vec::<String>::new();
    if semantic.scene_low_quality_ratio > 0.0 {
        warnings.push(format!(
            "scene_low_quality_ratio={:.2}%",
            semantic.scene_low_quality_ratio * 100.0
        ));
    }
    if semantic.name_inconsistency_ratio > 0.0 {
        warnings.push(format!(
            "name_inconsistency_ratio={:.2}%",
            semantic.name_inconsistency_ratio * 100.0
        ));
    }
    if semantic.malformed_contraction_ratio > 0.0 {
        warnings.push(format!(
            "malformed_contraction_ratio={:.2}%",
            semantic.malformed_contraction_ratio * 100.0
        ));
    }
    warnings
}

fn is_srt_path(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(|ext| ext.eq_ignore_ascii_case("srt"))
        .unwrap_or(false)
}

fn is_video_sidecar_source(input: &Path, source_srt: &Path) -> bool {
    !is_srt_path(input) && source_srt == input.with_extension("srt")
}

#[derive(Debug, Clone, Copy)]
struct SrtHealth {
    cue_count: usize,
    top_line_ratio: f64,
    overlap_ratio: f64,
    non_empty_ratio: f64,
}

impl SrtHealth {
    fn is_pathological(&self, profile: QualityProfile) -> bool {
        let thresholds = HealthThresholds::for_profile(profile);
        if self.cue_count < thresholds.min_cues {
            return false;
        }
        self.top_line_ratio >= thresholds.max_top_line_ratio
            || self.overlap_ratio >= thresholds.max_overlap_ratio
            || self.non_empty_ratio < thresholds.min_non_empty_ratio
    }

    fn summary(&self) -> String {
        format!(
            "cues={} top_line_ratio={:.2}% overlap_ratio={:.2}% non_empty_ratio={:.2}%",
            self.cue_count,
            self.top_line_ratio * 100.0,
            self.overlap_ratio * 100.0,
            self.non_empty_ratio * 100.0
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct TranslationSemanticHealth {
    cue_count: usize,
    anomaly_ratio: f64,
    malformed_contraction_ratio: f64,
    low_function_word_ratio: f64,
    adjacent_repeat_ratio: f64,
    scene_low_quality_ratio: f64,
    scene_count: usize,
    name_inconsistency_ratio: f64,
}

impl TranslationSemanticHealth {
    fn is_pathological(&self, profile: QualityProfile) -> bool {
        if self.cue_count == 0 {
            return false;
        }
        let thresholds = SemanticThresholds::for_profile(profile);
        self.anomaly_ratio >= thresholds.max_anomaly_ratio
            || self.malformed_contraction_ratio >= thresholds.max_malformed_contraction_ratio
            || self.low_function_word_ratio >= thresholds.max_low_function_word_ratio
            || self.adjacent_repeat_ratio >= thresholds.max_adjacent_repeat_ratio
            || self.scene_low_quality_ratio >= thresholds.max_scene_low_quality_ratio
            || self.name_inconsistency_ratio >= thresholds.max_name_inconsistency_ratio
    }

    fn summary(&self) -> String {
        format!(
            "cues={} anomaly_ratio={:.2}% malformed_contraction_ratio={:.2}% low_function_word_ratio={:.2}% adjacent_repeat_ratio={:.2}% scene_low_quality_ratio={:.2}% scene_count={} name_inconsistency_ratio={:.2}%",
            self.cue_count,
            self.anomaly_ratio * 100.0,
            self.malformed_contraction_ratio * 100.0,
            self.low_function_word_ratio * 100.0,
            self.adjacent_repeat_ratio * 100.0,
            self.scene_low_quality_ratio * 100.0,
            self.scene_count,
            self.name_inconsistency_ratio * 100.0,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct HealthThresholds {
    min_cues: usize,
    max_top_line_ratio: f64,
    max_overlap_ratio: f64,
    min_non_empty_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
struct SemanticThresholds {
    max_anomaly_ratio: f64,
    max_malformed_contraction_ratio: f64,
    max_low_function_word_ratio: f64,
    max_adjacent_repeat_ratio: f64,
    max_scene_low_quality_ratio: f64,
    max_name_inconsistency_ratio: f64,
}

impl SemanticThresholds {
    fn for_profile(profile: QualityProfile) -> Self {
        match profile {
            QualityProfile::Fast => Self {
                max_anomaly_ratio: 0.35,
                max_malformed_contraction_ratio: 0.25,
                max_low_function_word_ratio: 0.35,
                max_adjacent_repeat_ratio: 0.30,
                max_scene_low_quality_ratio: 0.45,
                max_name_inconsistency_ratio: 0.20,
            },
            QualityProfile::Balanced => Self {
                max_anomaly_ratio: 0.25,
                max_malformed_contraction_ratio: 0.15,
                max_low_function_word_ratio: 0.22,
                max_adjacent_repeat_ratio: 0.18,
                max_scene_low_quality_ratio: 0.30,
                max_name_inconsistency_ratio: 0.12,
            },
            QualityProfile::Strict => Self {
                max_anomaly_ratio: 0.03,
                max_malformed_contraction_ratio: 0.01,
                max_low_function_word_ratio: 0.08,
                max_adjacent_repeat_ratio: 0.02,
                max_scene_low_quality_ratio: 0.04,
                max_name_inconsistency_ratio: 0.03,
            },
        }
    }
}

impl HealthThresholds {
    fn for_profile(profile: QualityProfile) -> Self {
        match profile {
            QualityProfile::Fast => Self {
                min_cues: 240,
                max_top_line_ratio: 0.35,
                max_overlap_ratio: 0.40,
                min_non_empty_ratio: 0.65,
            },
            QualityProfile::Balanced => Self {
                min_cues: 200,
                max_top_line_ratio: 0.30,
                max_overlap_ratio: 0.35,
                min_non_empty_ratio: 0.70,
            },
            QualityProfile::Strict => Self {
                min_cues: 80,
                max_top_line_ratio: 0.20,
                max_overlap_ratio: 0.25,
                min_non_empty_ratio: 0.90,
            },
        }
    }
}

fn assess_srt_health(cues: &[SubtitleCue]) -> Result<SrtHealth, String> {
    if cues.is_empty() {
        return Ok(SrtHealth {
            cue_count: 0,
            top_line_ratio: 0.0,
            overlap_ratio: 0.0,
            non_empty_ratio: 0.0,
        });
    }

    let mut freq = HashMap::<String, usize>::new();
    let mut non_empty = 0usize;
    for cue in cues {
        let normalized = normalize_health_text(&cue.text);
        if normalized.is_empty() {
            continue;
        }
        non_empty += 1;
        *freq.entry(normalized).or_insert(0) += 1;
    }
    let top_count = freq.values().copied().max().unwrap_or(0);
    let top_line_ratio = (top_count as f64) / (cues.len() as f64);
    let non_empty_ratio = (non_empty as f64) / (cues.len() as f64);

    let mut overlaps = 0usize;
    let mut parsed = 0usize;
    let mut prev_end = 0.0f64;
    for cue in cues {
        let (start, end) = parse_srt_timing_line(&cue.timing)?;
        if parsed > 0 && start < prev_end {
            overlaps += 1;
        }
        parsed += 1;
        prev_end = prev_end.max(end);
    }
    let overlap_ratio = if parsed > 1 {
        (overlaps as f64) / ((parsed - 1) as f64)
    } else {
        0.0
    };

    Ok(SrtHealth {
        cue_count: cues.len(),
        top_line_ratio,
        overlap_ratio,
        non_empty_ratio,
    })
}

fn normalize_health_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn assess_translation_semantics(
    cues: &[SubtitleCue],
    target_lang: &str,
) -> TranslationSemanticHealth {
    // This semantic gate currently targets English fluency. For other targets,
    // we skip fluency scoring to avoid language-specific false positives.
    if !target_lang.eq_ignore_ascii_case("en") {
        return TranslationSemanticHealth {
            cue_count: cues.len(),
            anomaly_ratio: 0.0,
            malformed_contraction_ratio: 0.0,
            low_function_word_ratio: 0.0,
            adjacent_repeat_ratio: 0.0,
            scene_low_quality_ratio: 0.0,
            scene_count: 0,
            name_inconsistency_ratio: 0.0,
        };
    }

    let mut anomaly_count = 0usize;
    let mut malformed_contractions = 0usize;
    let mut low_function_word = 0usize;
    let mut adjacent_repeat = 0usize;

    for cue in cues {
        let tokens = tokenize_ascii_words(&cue.text);
        let mut anomalous = false;

        let has_malformed_contraction =
            cue_has_malformed_contraction(&cue.text) || token_has_double_apostrophe(&tokens);
        if has_malformed_contraction {
            malformed_contractions += 1;
            anomalous = true;
        }

        if cue_has_low_function_word_coverage(&tokens) {
            low_function_word += 1;
            anomalous = true;
        }

        if cue_has_adjacent_repeat(&tokens) {
            adjacent_repeat += 1;
            anomalous = true;
        }

        if anomalous {
            anomaly_count += 1;
        }
    }

    let total = cues.len().max(1) as f64;
    let scene_report = assess_scene_quality(cues);
    let name_inconsistency_ratio = assess_name_inconsistency(cues);
    TranslationSemanticHealth {
        cue_count: cues.len(),
        anomaly_ratio: (anomaly_count as f64) / total,
        malformed_contraction_ratio: (malformed_contractions as f64) / total,
        low_function_word_ratio: (low_function_word as f64) / total,
        adjacent_repeat_ratio: (adjacent_repeat as f64) / total,
        scene_low_quality_ratio: scene_report.low_quality_ratio,
        scene_count: scene_report.scene_count,
        name_inconsistency_ratio,
    }
}

#[derive(Debug, Clone, Copy)]
struct SceneQualityReport {
    scene_count: usize,
    low_quality_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
struct LowQualitySceneRange {
    start: usize,
    end: usize,
    score: f64,
    floor: f64,
}

#[derive(Debug, Clone, Copy)]
struct CueAsrConfidence {
    score: f64,
    avg_logprob: f64,
    no_speech_prob: f64,
    compression_ratio: f64,
    word_prob_mean: f64,
    low_word_prob_ratio: f64,
    suspicious: bool,
}

#[derive(Debug, Clone, Copy)]
struct LowConfidenceCueSpan {
    start: usize,
    end: usize,
    score: f64,
    floor: f64,
}

#[derive(Debug, Default, Clone, Copy)]
struct CueCompactionStats {
    merged_pairs: usize,
    dropped_duplicates: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct DiscourseConsistencyStats {
    source_clusters: usize,
    rewritten_cues: usize,
}

fn assess_scene_quality(cues: &[SubtitleCue]) -> SceneQualityReport {
    let scenes = split_scenes(cues);
    if scenes.is_empty() {
        return SceneQualityReport {
            scene_count: 0,
            low_quality_ratio: 0.0,
        };
    }

    let mut low_quality = 0usize;
    for scene in &scenes {
        let (difficulty, score) = scene_quality(scene);
        let floor = scene_floor_for_difficulty(difficulty);
        if score < floor {
            low_quality += 1;
        }
    }

    SceneQualityReport {
        scene_count: scenes.len(),
        low_quality_ratio: low_quality as f64 / scenes.len() as f64,
    }
}

fn split_scenes(cues: &[SubtitleCue]) -> Vec<Vec<&SubtitleCue>> {
    split_scene_ranges(cues)
        .into_iter()
        .map(|(start, end)| cues[start..end].iter().collect())
        .collect()
}

fn split_scene_ranges(cues: &[SubtitleCue]) -> Vec<(usize, usize)> {
    if cues.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::<(usize, usize)>::new();
    let mut scene_start: Option<usize> = None;
    let mut scene_end_exclusive = 0usize;
    let mut prev_end = 0.0f64;
    for (idx, cue) in cues.iter().enumerate() {
        let (start, end) = match parse_srt_timing_line(&cue.timing) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let gap = (start - prev_end).max(0.0);
        if scene_start.is_none() {
            scene_start = Some(idx);
        } else if gap >= 1.5 || (end - prev_end) >= 420.0 {
            if let Some(start_idx) = scene_start {
                ranges.push((start_idx, scene_end_exclusive));
            }
            scene_start = Some(idx);
        }
        scene_end_exclusive = idx + 1;
        prev_end = end.max(prev_end);
    }
    if let Some(start_idx) = scene_start {
        ranges.push((start_idx, scene_end_exclusive));
    }
    ranges
}

fn scene_quality(scene: &[&SubtitleCue]) -> (f64, f64) {
    if scene.is_empty() {
        return (0.5, 1.0);
    }
    let mut anomaly = 0usize;
    let mut malformed = 0usize;
    let mut total_tokens = 0usize;
    let mut durations = 0.0f64;
    let mut prev_end = 0.0f64;
    let mut fast_turns = 0usize;

    for cue in scene {
        let tokens = tokenize_ascii_words(&cue.text);
        total_tokens += tokens.len();
        if cue_has_malformed_contraction(&cue.text) || token_has_double_apostrophe(&tokens) {
            malformed += 1;
            anomaly += 1;
        }
        if cue_has_adjacent_repeat(&tokens) || cue_has_low_function_word_coverage(&tokens) {
            anomaly += 1;
        }
        if let Ok((start, end)) = parse_srt_timing_line(&cue.timing) {
            durations += (end - start).max(0.0);
            let gap = (start - prev_end).max(0.0);
            if gap < 0.15 {
                fast_turns += 1;
            }
            prev_end = end;
        }
    }

    let cues = scene.len().max(1) as f64;
    let anomaly_ratio = anomaly as f64 / cues;
    let malformed_ratio = malformed as f64 / cues;
    let avg_tokens = total_tokens as f64 / cues;
    let avg_duration = durations / cues.max(1.0);
    let fast_turn_ratio = fast_turns as f64 / cues;

    let difficulty = (0.20_f64
        + if avg_duration <= 1.8 {
            0.28_f64
        } else {
            0.10_f64
        }
        + if avg_tokens >= 8.0 {
            0.22_f64
        } else {
            0.08_f64
        }
        + if fast_turn_ratio >= 0.35 {
            0.22_f64
        } else {
            0.05_f64
        })
    .clamp(0.10_f64, 0.95_f64);

    let score = (1.0 - (anomaly_ratio * 1.35 + malformed_ratio * 1.55)).clamp(0.0, 1.0);
    (difficulty, score)
}

fn scene_quality_for_slice(scene: &[SubtitleCue]) -> (f64, f64) {
    let refs = scene.iter().collect::<Vec<_>>();
    scene_quality(&refs)
}

fn scene_floor_for_difficulty(difficulty: f64) -> f64 {
    if difficulty >= 0.70 {
        0.70
    } else if difficulty <= 0.30 {
        0.90
    } else {
        0.80
    }
}

fn collect_low_quality_scene_ranges(cues: &[SubtitleCue]) -> Vec<LowQualitySceneRange> {
    let mut scenes = Vec::<LowQualitySceneRange>::new();
    for (start, end) in split_scene_ranges(cues) {
        if end <= start {
            continue;
        }
        let (difficulty, score) = scene_quality_for_slice(&cues[start..end]);
        let floor = scene_floor_for_difficulty(difficulty);
        if score < floor {
            scenes.push(LowQualitySceneRange {
                start,
                end,
                score,
                floor,
            });
        }
    }
    scenes
}

fn collect_low_quality_source_scene_ranges(cues: &[SubtitleCue]) -> Vec<LowQualitySceneRange> {
    let mut scenes = Vec::<LowQualitySceneRange>::new();
    for (start, end) in split_scene_ranges(cues) {
        if end <= start {
            continue;
        }
        let scene = &cues[start..end];
        let score = source_scene_quality_score(scene);
        let floor = source_scene_quality_floor(scene.len());
        if score < floor {
            scenes.push(LowQualitySceneRange {
                start,
                end,
                score,
                floor,
            });
        }
    }
    scenes
}

#[derive(Debug, Clone, Copy)]
struct WhisperConfidenceSegment {
    start: f64,
    end: f64,
    avg_logprob: f64,
    no_speech_prob: f64,
    compression_ratio: f64,
    word_prob_mean: f64,
    low_word_prob_ratio: f64,
}

fn load_cue_asr_confidence_from_whisper_json(
    srt_path: &Path,
    cues: &[SubtitleCue],
) -> Option<Vec<Option<CueAsrConfidence>>> {
    if cues.is_empty() {
        return Some(Vec::new());
    }

    let json_path = srt_path.with_extension("json");
    let raw = std::fs::read_to_string(&json_path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let segments = parse_whisper_confidence_segments(&parsed)?;
    if segments.is_empty() {
        return None;
    }

    let mut mapped = Vec::<Option<CueAsrConfidence>>::with_capacity(cues.len());
    for cue in cues {
        let Ok((cue_start, cue_end)) = parse_srt_timing_line(&cue.timing) else {
            mapped.push(None);
            continue;
        };
        let cue_interval = Interval {
            start: cue_start,
            end: cue_end,
        };
        let mut weighted_score = 0.0f64;
        let mut weighted_logprob = 0.0f64;
        let mut weighted_no_speech = 0.0f64;
        let mut weighted_compression = 0.0f64;
        let mut weighted_word_prob = 0.0f64;
        let mut weighted_low_word_prob = 0.0f64;
        let mut total_overlap = 0.0f64;
        let mut suspicious = false;

        for segment in &segments {
            let overlap = interval_overlap_seconds(
                cue_interval,
                Interval {
                    start: segment.start,
                    end: segment.end,
                },
            );
            if overlap <= 0.0 {
                continue;
            }

            let score = segment_confidence_score(segment);
            weighted_score += score * overlap;
            weighted_logprob += segment.avg_logprob * overlap;
            weighted_no_speech += segment.no_speech_prob * overlap;
            weighted_compression += segment.compression_ratio * overlap;
            weighted_word_prob += segment.word_prob_mean * overlap;
            weighted_low_word_prob += segment.low_word_prob_ratio * overlap;
            total_overlap += overlap;
            suspicious |= segment_is_suspicious(segment);
        }

        if total_overlap <= 0.0 {
            mapped.push(None);
            continue;
        }

        mapped.push(Some(CueAsrConfidence {
            score: (weighted_score / total_overlap).clamp(0.0, 1.0),
            avg_logprob: weighted_logprob / total_overlap,
            no_speech_prob: weighted_no_speech / total_overlap,
            compression_ratio: weighted_compression / total_overlap,
            word_prob_mean: weighted_word_prob / total_overlap,
            low_word_prob_ratio: weighted_low_word_prob / total_overlap,
            suspicious,
        }));
    }

    let covered = mapped.iter().filter(|entry| entry.is_some()).count();
    if covered == 0 {
        return None;
    }

    eprintln!(
        "ibvoid-doom-qlock: loaded ASR confidence from {} (coverage={}/{})",
        json_path.display(),
        covered,
        cues.len()
    );
    Some(mapped)
}

fn parse_whisper_confidence_segments(
    root: &serde_json::Value,
) -> Option<Vec<WhisperConfidenceSegment>> {
    let segments = root.get("segments")?.as_array()?;
    let mut out = Vec::<WhisperConfidenceSegment>::with_capacity(segments.len());
    for segment in segments {
        let Some(start) = segment.get("start").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let Some(end) = segment.get("end").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        if end <= start {
            continue;
        }
        let avg_logprob = segment
            .get("avg_logprob")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(-1.5);
        let no_speech_prob = segment
            .get("no_speech_prob")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let compression_ratio = segment
            .get("compression_ratio")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0)
            .max(0.0);
        let (word_prob_mean, low_word_prob_ratio) = parse_word_probability_stats(segment);
        out.push(WhisperConfidenceSegment {
            start,
            end,
            avg_logprob,
            no_speech_prob,
            compression_ratio,
            word_prob_mean,
            low_word_prob_ratio,
        });
    }
    Some(out)
}

fn parse_word_probability_stats(segment: &serde_json::Value) -> (f64, f64) {
    let Some(words) = segment.get("words").and_then(serde_json::Value::as_array) else {
        return (0.0, 0.0);
    };
    let mut sum = 0.0f64;
    let mut total = 0usize;
    let mut low = 0usize;
    for word in words {
        let Some(prob) = word.get("probability").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let clamped = prob.clamp(0.0, 1.0);
        sum += clamped;
        total += 1;
        if clamped < 0.45 {
            low += 1;
        }
    }
    if total == 0 {
        (0.0, 0.0)
    } else {
        (sum / total as f64, low as f64 / total as f64)
    }
}

fn segment_confidence_score(segment: &WhisperConfidenceSegment) -> f64 {
    let logprob_confidence = segment.avg_logprob.exp().clamp(0.0, 1.0) * 0.55;
    let word_confidence = segment.word_prob_mean.clamp(0.0, 1.0) * 0.45;
    let no_speech_penalty = segment.no_speech_prob * 0.45;
    let compression_penalty = ((segment.compression_ratio - 2.0).max(0.0) / 2.0).min(0.35);
    let low_word_penalty = (segment.low_word_prob_ratio * 0.35).min(0.25);
    (logprob_confidence + word_confidence
        - no_speech_penalty
        - compression_penalty
        - low_word_penalty)
        .clamp(0.0, 1.0)
}

fn segment_is_suspicious(segment: &WhisperConfidenceSegment) -> bool {
    let weak_logprob = segment.avg_logprob <= -1.25;
    let weak_words = segment.word_prob_mean > 0.0
        && (segment.word_prob_mean < 0.45 || segment.low_word_prob_ratio > 0.55);
    segment.compression_ratio > 2.4
        || (segment.no_speech_prob > 0.70 && weak_logprob)
        || segment.avg_logprob <= -1.60
        || weak_words
}

fn collect_low_confidence_cue_spans(
    cues: &[SubtitleCue],
    confidence: &[Option<CueAsrConfidence>],
    profile: QualityProfile,
) -> Vec<LowConfidenceCueSpan> {
    if cues.is_empty() || cues.len() != confidence.len() {
        return Vec::new();
    }

    let floor = match profile {
        QualityProfile::Fast => 0.42,
        QualityProfile::Balanced => 0.50,
        QualityProfile::Strict => 0.56,
    };

    let mut spans = Vec::<LowConfidenceCueSpan>::new();
    let mut idx = 0usize;
    while idx < cues.len() {
        if !is_low_confidence_entry(confidence[idx], floor) {
            idx += 1;
            continue;
        }

        let start = idx;
        let mut end = idx + 1;
        let mut non_low_budget = 1usize;
        while end < cues.len() {
            if is_low_confidence_entry(confidence[end], floor) {
                non_low_budget = 1;
                end += 1;
                continue;
            }
            if non_low_budget == 0 {
                break;
            }
            non_low_budget -= 1;
            end += 1;
        }
        while end > start && !is_low_confidence_entry(confidence[end - 1], floor) {
            end -= 1;
        }
        if end <= start {
            idx += 1;
            continue;
        }

        let score = mean_confidence_score(&confidence[start..end]).unwrap_or(0.0);
        spans.push(LowConfidenceCueSpan {
            start,
            end,
            score,
            floor,
        });
        idx = end;
    }
    spans
}

fn is_low_confidence_entry(entry: Option<CueAsrConfidence>, floor: f64) -> bool {
    let Some(entry) = entry else {
        return false;
    };
    entry.score < floor
        || entry.suspicious
        || entry.avg_logprob <= -1.25
        || (entry.word_prob_mean > 0.0 && entry.word_prob_mean < 0.42)
        || entry.low_word_prob_ratio > 0.60
        || (entry.no_speech_prob > 0.65 && entry.score < floor + 0.08)
}

fn mean_confidence_score(confidence: &[Option<CueAsrConfidence>]) -> Option<f64> {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for entry in confidence {
        let Some(entry) = entry else {
            continue;
        };
        // Compression is treated as a secondary penalty to avoid over-trusting
        // high-logprob segments that look like repetition loops.
        let compression_penalty = ((entry.compression_ratio - 2.2).max(0.0) / 2.2).min(0.20);
        let word_penalty = (entry.low_word_prob_ratio * 0.20).min(0.15);
        sum += (entry.score - compression_penalty - word_penalty).clamp(0.0, 1.0);
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

fn apply_source_phrase_consistency(
    source_cues: &[SubtitleCue],
    translated_cues: &mut [SubtitleCue],
) -> DiscourseConsistencyStats {
    let mut source_to_target = HashMap::<String, HashMap<String, usize>>::new();
    let mut source_target_display = HashMap::<String, HashMap<String, String>>::new();
    for (source, target) in source_cues.iter().zip(translated_cues.iter()) {
        let source_key = normalize_source_phrase_for_consistency(&source.text);
        if source_key.len() < 2 || source_key.len() > 64 {
            continue;
        }
        let target_display = normalize_display_phrase(&target.text);
        let target_key = normalize_target_phrase_for_consistency(&target.text);
        if target_key.is_empty() || target_key.split_whitespace().count() > 10 {
            continue;
        }
        *source_to_target
            .entry(source_key.clone())
            .or_default()
            .entry(target_key.clone())
            .or_insert(0) += 1;
        source_target_display
            .entry(source_key)
            .or_default()
            .entry(target_key)
            .or_insert(target_display);
    }

    let mut canonical = HashMap::<String, String>::new();
    let mut stats = DiscourseConsistencyStats::default();
    for (source_key, variants) in source_to_target {
        let total = variants.values().sum::<usize>();
        if total < 3 || variants.len() < 2 {
            continue;
        }
        let Some((best_key, best_count)) = variants.iter().max_by_key(|(_, count)| **count) else {
            continue;
        };
        if (*best_count as f64) / (total as f64) < 0.50 {
            continue;
        }
        let display = source_target_display
            .get(&source_key)
            .and_then(|variant_map| variant_map.get(best_key))
            .cloned()
            .unwrap_or_else(|| best_key.clone());
        canonical.insert(source_key, display);
        stats.source_clusters += 1;
    }

    if canonical.is_empty() {
        return stats;
    }

    for (index, source) in source_cues.iter().enumerate() {
        let source_key = normalize_source_phrase_for_consistency(&source.text);
        let Some(canonical_display) = canonical.get(&source_key) else {
            continue;
        };
        let canonical_key = normalize_target_phrase_for_consistency(canonical_display);
        let current_key = normalize_target_phrase_for_consistency(&translated_cues[index].text);
        if current_key.is_empty() || current_key == canonical_key {
            continue;
        }
        if !should_replace_consistency_variant(&current_key, &canonical_key) {
            continue;
        }
        let rewritten =
            rewrite_with_canonical_phrase(&translated_cues[index].text, canonical_display);
        if normalize_target_phrase_for_consistency(&rewritten) == current_key {
            continue;
        }
        translated_cues[index].text = rewritten;
        stats.rewritten_cues += 1;
    }

    stats
}

fn normalize_display_phrase(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_source_phrase_for_consistency(text: &str) -> String {
    text.chars()
        .filter(|ch| {
            ch.is_ascii_alphanumeric()
                || ('\u{3040}'..='\u{30FF}').contains(ch)
                || ('\u{4E00}'..='\u{9FFF}').contains(ch)
        })
        .collect::<String>()
}

fn normalize_target_phrase_for_consistency(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() || ch == '\'' {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn should_replace_consistency_variant(current: &str, canonical: &str) -> bool {
    if current.is_empty() || canonical.is_empty() {
        return false;
    }
    let similarity = strsim::normalized_levenshtein(current, canonical);
    if similarity >= 0.60 {
        return true;
    }

    let current_words = current.split_whitespace().count();
    let canonical_words = canonical.split_whitespace().count();
    current_words <= 4
        && canonical_words <= 4
        && (current.starts_with(canonical) || canonical.starts_with(current))
}

fn rewrite_with_canonical_phrase(original: &str, canonical: &str) -> String {
    let mut rewritten = canonical.to_string();
    if original
        .trim_start()
        .chars()
        .next()
        .map(|ch| ch.is_ascii_uppercase())
        .unwrap_or(false)
    {
        rewritten = capitalize_ascii_first(&rewritten);
    }
    let punctuation_suffix = original
        .trim_end()
        .chars()
        .rev()
        .take_while(|ch| matches!(ch, '.' | '!' | '?'))
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    if !punctuation_suffix.is_empty() && !rewritten.ends_with(&punctuation_suffix) {
        rewritten.push_str(&punctuation_suffix);
    }
    rewritten
}

fn capitalize_ascii_first(text: &str) -> String {
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    if !first.is_ascii_lowercase() {
        return text.to_string();
    }
    format!("{}{}", first.to_ascii_uppercase(), chars.as_str())
}

fn compact_adjacent_cues(
    cues: &[SubtitleCue],
    max_gap_s: f64,
    max_chars_per_line: usize,
    max_lines: usize,
    max_cps: f64,
    max_duration_s: f64,
) -> Result<(Vec<SubtitleCue>, CueCompactionStats), String> {
    if cues.is_empty() {
        return Ok((Vec::new(), CueCompactionStats::default()));
    }

    let mut out = Vec::<SubtitleCue>::new();
    let mut stats = CueCompactionStats::default();
    let max_merge_pairs = (cues.len() / 16).max(20);
    let mut idx = 0usize;

    while idx < cues.len() {
        let mut current = cues[idx].clone();
        current.text = normalize_compaction_text(&current.text);
        let (start, mut end) = parse_srt_timing_line(&current.timing)?;
        let mut current_text = current.text.clone();
        let mut lookahead = idx + 1;

        while lookahead < cues.len() {
            let next = &cues[lookahead];
            let (next_start, next_end) = parse_srt_timing_line(&next.timing)?;
            let next_text = normalize_compaction_text(&next.text);
            if next_text.is_empty() {
                lookahead += 1;
                continue;
            }

            let gap = (next_start - end).max(0.0);
            if gap > max_gap_s {
                break;
            }

            if normalized_text_key(&current_text) == normalized_text_key(&next_text)
                && gap <= 0.12
                && current_text.chars().count() <= 48
            {
                end = next_end.max(end);
                stats.dropped_duplicates += 1;
                lookahead += 1;
                continue;
            }

            if !can_merge_cue_texts(&current_text, &next_text) {
                break;
            }
            if stats.merged_pairs >= max_merge_pairs {
                break;
            }

            let merged_text = merge_cue_texts(&current_text, &next_text);
            let merged_duration = (next_end - start).max(0.001);
            let merged_chars = merged_text.chars().count() as f64;
            let merged_cps = merged_chars / merged_duration;
            let merged_line_budget = max_chars_per_line * max_lines;
            let merged_tokens = tokenize_ascii_words(&merged_text);
            if merged_duration > max_duration_s
                || merged_cps > max_cps
                || merged_text.chars().count() > merged_line_budget
                || cue_has_adjacent_repeat(&merged_tokens)
                || has_repeated_ngram(&merged_tokens, 3)
                || cue_has_low_function_word_coverage(&merged_tokens)
            {
                break;
            }

            current_text = merged_text;
            end = next_end.max(end);
            stats.merged_pairs += 1;
            lookahead += 1;
        }

        out.push(SubtitleCue {
            index: current.index,
            timing: format_srt_timing_line(start, end),
            text: current_text,
        });
        idx = lookahead.max(idx + 1);
    }

    Ok((out, stats))
}

fn normalize_compaction_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalized_text_key(text: &str) -> String {
    normalize_compaction_text(text).to_ascii_lowercase()
}

fn can_merge_cue_texts(left: &str, right: &str) -> bool {
    if left.is_empty() || right.is_empty() {
        return false;
    }

    if left.contains('[') || left.contains(']') || right.contains('[') || right.contains(']') {
        return false;
    }

    let left_trim = left.trim_end();
    let right_trim = right.trim_start();
    let left_terminal = left_trim.chars().last().unwrap_or(' ');
    let right_initial = right_trim.chars().next().unwrap_or(' ');
    let left_words = left_trim.split_whitespace().count();
    let right_words = right_trim.split_whitespace().count();

    if left_words > 8 || right_words > 8 {
        return false;
    }

    if right_initial.is_ascii_uppercase() {
        return false;
    }
    if matches!(left_terminal, '.' | '?' | '!' | ':') {
        return false;
    }
    if left_trim.ends_with("...") && right_initial.is_ascii_alphabetic() {
        return false;
    }
    true
}

fn merge_cue_texts(left: &str, right: &str) -> String {
    let left = left.trim_end();
    let right = right.trim_start();
    if left.is_empty() {
        return right.to_string();
    }
    if right.is_empty() {
        return left.to_string();
    }
    format!("{left} {right}")
}

fn has_repeated_ngram(tokens: &[String], n: usize) -> bool {
    if n < 2 || tokens.len() < n * 2 {
        return false;
    }
    for i in 0..=tokens.len() - n {
        for j in (i + n)..=tokens.len() - n {
            if tokens[i..i + n] == tokens[j..j + n] {
                return true;
            }
        }
    }
    false
}

fn source_scene_quality_floor(scene_len: usize) -> f64 {
    if scene_len >= 8 {
        0.82
    } else if scene_len >= 4 {
        0.76
    } else {
        0.70
    }
}

fn source_scene_quality_score(scene: &[SubtitleCue]) -> f64 {
    if scene.is_empty() {
        return 1.0;
    }

    let health = match assess_srt_health(scene) {
        Ok(v) => v,
        Err(_) => return 0.0,
    };

    let mut short = 0usize;
    let mut very_short = 0usize;
    let mut adjacent_duplicates = 0usize;
    let mut prev = String::new();
    for cue in scene {
        if let Ok((start, end)) = parse_srt_timing_line(&cue.timing) {
            let duration = (end - start).max(0.0);
            if duration < 0.45 {
                short += 1;
            }
            if duration < 0.20 {
                very_short += 1;
            }
        }
        let normalized = normalize_health_text(&cue.text);
        if !normalized.is_empty() && normalized == prev {
            adjacent_duplicates += 1;
        }
        prev = normalized;
    }

    let total = scene.len().max(1) as f64;
    let short_ratio = short as f64 / total;
    let very_short_ratio = very_short as f64 / total;
    let duplicate_ratio = adjacent_duplicates as f64 / total;

    (1.0 - (health.top_line_ratio * 1.45
        + health.overlap_ratio * 1.35
        + (1.0 - health.non_empty_ratio) * 1.6
        + short_ratio * 0.55
        + very_short_ratio * 0.8
        + duplicate_ratio * 0.95))
        .clamp(0.0, 1.0)
}

fn scene_time_span(scene: &[SubtitleCue]) -> Option<(f64, f64)> {
    let mut min_start = f64::INFINITY;
    let mut max_end = 0.0f64;
    for cue in scene {
        let (start, end) = parse_srt_timing_line(&cue.timing).ok()?;
        min_start = min_start.min(start);
        max_end = max_end.max(end);
    }
    if min_start.is_finite() && max_end > min_start {
        Some((min_start, max_end))
    } else {
        None
    }
}

fn shift_cues_by_offset(cues: &[SubtitleCue], offset: f64) -> Result<Vec<SubtitleCue>, String> {
    let mut shifted = Vec::<SubtitleCue>::with_capacity(cues.len());
    for cue in cues {
        let (start, end) = parse_srt_timing_line(&cue.timing)?;
        shifted.push(SubtitleCue {
            index: cue.index,
            timing: format_srt_timing_line(start + offset, end + offset),
            text: cue.text.clone(),
        });
    }
    Ok(shifted)
}

fn format_srt_timing_line(start: f64, end: f64) -> String {
    format!(
        "{} --> {}",
        format_srt_timestamp(start),
        format_srt_timestamp(end)
    )
}

fn format_srt_timestamp(seconds: f64) -> String {
    let clamped = seconds.max(0.0);
    let total_ms = (clamped * 1000.0).round() as u64;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let secs = (total_ms % 60_000) / 1000;
    let millis = total_ms % 1000;
    format!("{hours:02}:{minutes:02}:{secs:02},{millis:03}")
}

fn create_temp_rescue_dir(scene_start: usize) -> Result<PathBuf, String> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "sub_zero_source_scene_rescue_{scene_start}_{stamp}"
    ));
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(dir)
}

fn extract_audio_segment_to_wav(
    input: &Path,
    wav_out: &Path,
    start: f64,
    end: f64,
) -> Result<(), String> {
    if end <= start + 0.05 {
        return Err(format!(
            "invalid rescue segment: start={start:.3} end={end:.3}"
        ));
    }
    let ffmpeg = find_in_path(&["ffmpeg", "ffmpeg.exe"])
        .ok_or_else(|| "ffmpeg not found in PATH (required for source rescue)".to_string())?;
    let status = Command::new(ffmpeg)
        .arg("-hide_banner")
        .arg("-nostdin")
        .arg("-v")
        .arg("error")
        .arg("-y")
        .arg("-ss")
        .arg(format!("{start:.3}"))
        .arg("-to")
        .arg(format!("{end:.3}"))
        .arg("-i")
        .arg(input)
        .arg("-vn")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-f")
        .arg("wav")
        .arg(wav_out)
        .status()
        .map_err(|e| format!("failed to spawn ffmpeg for source rescue: {e}"))?;
    if !status.success() {
        return Err(format!("ffmpeg source rescue failed with status: {status}"));
    }
    Ok(())
}

fn default_mt_batch_for_profile(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 32,
        QualityProfile::Balanced => 24,
        QualityProfile::Strict => 16,
    }
}

fn default_mt_tokens_for_profile(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 8_192,
        QualityProfile::Balanced => 6_144,
        QualityProfile::Strict => 4_096,
    }
}

fn default_mt_oom_retries_for_profile(profile: QualityProfile) -> usize {
    match profile {
        QualityProfile::Fast => 1,
        QualityProfile::Balanced => 2,
        QualityProfile::Strict => 3,
    }
}

fn scene_semantic_penalty(health: &TranslationSemanticHealth) -> f64 {
    health.anomaly_ratio * 1.3
        + health.malformed_contraction_ratio * 1.5
        + health.low_function_word_ratio
        + health.adjacent_repeat_ratio * 1.2
}

fn assess_name_inconsistency(cues: &[SubtitleCue]) -> f64 {
    let mut names = Vec::<String>::new();
    for cue in cues {
        names.extend(extract_titlecase_tokens(&cue.text));
    }
    if names.len() < 4 {
        return 0.0;
    }

    let mut freq = HashMap::<String, usize>::new();
    for name in &names {
        *freq.entry(name.to_string()).or_insert(0) += 1;
    }

    let mut inconsistent_mentions = 0usize;
    let total_mentions = names.len();
    let keys: Vec<String> = freq.keys().cloned().collect();
    for i in 0..keys.len() {
        for j in (i + 1)..keys.len() {
            let a = &keys[i];
            let b = &keys[j];
            if a.chars().next() != b.chars().next() {
                continue;
            }
            if (a.len() as isize - b.len() as isize).abs() > 2 {
                continue;
            }
            let similarity =
                strsim::normalized_levenshtein(&a.to_ascii_lowercase(), &b.to_ascii_lowercase());
            if (0.78..1.0).contains(&similarity) {
                let count_a = freq.get(a).copied().unwrap_or(0);
                let count_b = freq.get(b).copied().unwrap_or(0);
                inconsistent_mentions += count_a.min(count_b);
            }
        }
    }
    (inconsistent_mentions as f64 / total_mentions as f64).clamp(0.0, 1.0)
}

fn extract_titlecase_tokens(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_ascii_alphabetic())
        .filter(|token| token.len() >= 3)
        .filter(|token| {
            token
                .chars()
                .next()
                .map(|ch| ch.is_ascii_uppercase())
                .unwrap_or(false)
                && token.chars().skip(1).all(|ch| ch.is_ascii_lowercase())
        })
        .map(|token| token.to_string())
        .collect()
}

fn tokenize_ascii_words(text: &str) -> Vec<String> {
    let mut tokens = Vec::<String>::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || ch == '\'' {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn token_has_double_apostrophe(tokens: &[String]) -> bool {
    tokens.iter().any(|token| token.matches('\'').count() >= 2)
}

fn cue_has_malformed_contraction(text: &str) -> bool {
    let lowered = text.to_ascii_lowercase();
    [
        "i'm's",
        "you're's",
        "we're's",
        "they're's",
        "he's's",
        "she's's",
        "it's's",
        "let's's",
        "i'm be",
        "i'm let",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

fn cue_has_low_function_word_coverage(tokens: &[String]) -> bool {
    if tokens.len() < 6 {
        return false;
    }

    const FUNCTION_WORDS: &[&str] = &[
        "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "for", "with", "is",
        "are", "was", "were", "be", "been", "being", "i", "you", "he", "she", "it", "we", "they",
        "me", "my", "your", "our", "their", "this", "that", "these", "those", "not", "do", "did",
        "does", "have", "has", "had", "can", "could", "will", "would", "should", "as", "at",
        "from",
    ];

    let function_words = tokens
        .iter()
        .map(|token| token.to_ascii_lowercase())
        .filter(|token| FUNCTION_WORDS.contains(&token.as_str()))
        .count();
    let ratio = (function_words as f64) / (tokens.len() as f64);
    ratio < 0.10
}

fn cue_has_adjacent_repeat(tokens: &[String]) -> bool {
    if tokens.len() < 4 {
        return false;
    }
    let lowered: Vec<String> = tokens
        .iter()
        .map(|token| token.to_ascii_lowercase())
        .collect();

    // Repeated token: "no no no"
    for pair in lowered.windows(2) {
        if pair[0] == pair[1] {
            return true;
        }
    }

    // Repeated bigram: "wait a wait a"
    for i in 0..(lowered.len().saturating_sub(3)) {
        if lowered[i..i + 2] == lowered[i + 2..i + 4] {
            return true;
        }
    }
    false
}

fn looks_like_simulated_placeholder_srt(path: &Path) -> bool {
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = vec![0u8; 4096];
    let Ok(bytes_read) = file.read(&mut buf) else {
        return false;
    };
    buf.truncate(bytes_read);
    let Ok(prefix) = std::str::from_utf8(&buf) else {
        return false;
    };
    let lower = prefix.to_ascii_lowercase();
    lower.contains("(simulated) subtitle #") || lower.contains("ai processing...")
}

fn checkpoint_dir_for(video: &Path) -> Result<PathBuf, String> {
    let video_key = video.canonicalize().unwrap_or_else(|_| video.to_path_buf());
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    video_key.hash(&mut hasher);
    let run_hash = format!("{:016x}", hasher.finish());

    let base = if let Some(home) = std::env::var_os("SUB_ZERO_HOME") {
        PathBuf::from(home)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".sub-zero")
    } else if let Some(home) = std::env::var_os("USERPROFILE") {
        PathBuf::from(home).join(".sub-zero")
    } else {
        std::env::temp_dir().join(".sub-zero")
    };

    let dir = base.join("checkpoints").join(run_hash);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(dir)
}

fn extract_audio_to_wav(video: &Path, wav_out: &Path) -> Result<(), String> {
    let ffmpeg = find_in_path(&["ffmpeg", "ffmpeg.exe"])
        .ok_or_else(|| "ffmpeg not found in PATH (required for --transcribe)".to_string())?;

    let status = std::process::Command::new(ffmpeg)
        .arg("-hide_banner")
        .arg("-nostdin")
        .arg("-v")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(video)
        .arg("-vn")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-f")
        .arg("wav")
        .arg(wav_out)
        .status()
        .map_err(|e| format!("failed to spawn ffmpeg: {e}"))?;

    if !status.success() {
        return Err(format!("ffmpeg failed with status: {status}"));
    }
    Ok(())
}

fn find_in_path(candidates: &[&str]) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        for name in candidates {
            let path = dir.join(name);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn now_epoch_secs() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn parse_srt_timestamp_to_seconds(ts: &str) -> Result<f64, String> {
    let ts = ts.trim();
    let (hms, ms) = ts
        .split_once(',')
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?;
    let mut parts = hms.split(':');
    let h = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let m = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let s = parts
        .next()
        .ok_or_else(|| format!("invalid timestamp: {ts}"))?
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    let ms = ms
        .parse::<u64>()
        .map_err(|_| format!("invalid timestamp: {ts}"))?;
    Ok((h as f64) * 3600.0 + (m as f64) * 60.0 + (s as f64) + (ms as f64) / 1000.0)
}

fn parse_srt_timing_line(line: &str) -> Result<(f64, f64), String> {
    let (start, end) = line
        .split_once("-->")
        .ok_or_else(|| format!("invalid timing line: {line}"))?;
    let start = parse_srt_timestamp_to_seconds(start)?;
    let end = parse_srt_timestamp_to_seconds(end)?;
    if end < start {
        return Err(format!("invalid timing (end < start): {line}"));
    }
    Ok((start, end))
}

#[derive(Debug, Clone, Copy)]
struct Interval {
    start: f64,
    end: f64,
}

fn interval_overlap_seconds(a: Interval, b: Interval) -> f64 {
    let start = a.start.max(b.start);
    let end = a.end.min(b.end);
    if end <= start {
        0.0
    } else {
        end - start
    }
}

fn verify_srt_against_audio(
    output_srt: &Path,
    audio_wav: &Path,
    threshold_db: f64,
    min_silence: f64,
    pad: f64,
    min_speech_overlap: f64,
) -> Result<String, String> {
    let cues = parse_srt_file(output_srt).map_err(|e| format!("{}: {e}", output_srt.display()))?;
    let mut max_end = 0.0f64;
    for cue in &cues {
        let (_, end) = parse_srt_timing_line(&cue.timing)?;
        if end > max_end {
            max_end = end;
        }
    }
    let analysis_end = (max_end + 1.0).max(0.0);
    let speech = crate::engine::transcribe::detect_speech_intervals_from_wav(
        audio_wav,
        threshold_db,
        min_silence,
        pad,
        Some(analysis_end),
    )?;

    let mut issues = Vec::<String>::new();
    let mut last_end = 0.0f64;

    for cue in &cues {
        let (start, end) = parse_srt_timing_line(&cue.timing)?;
        if start < last_end {
            issues.push(format!(
                "non-monotonic cue {}: {} (prev_end={:.3})",
                cue.index, cue.timing, last_end
            ));
        }
        last_end = end;

        let dur = (end - start).max(0.000_001);
        let cue_interval = Interval { start, end };
        let mut overlap = 0.0f64;
        for s in &speech {
            overlap += interval_overlap_seconds(
                cue_interval,
                Interval {
                    start: s.start,
                    end: s.end,
                },
            );
        }
        let ratio = overlap / dur;
        if ratio < min_speech_overlap {
            issues.push(format!(
                "low speech overlap cue {}: {:.2}% ({}): {}",
                cue.index,
                ratio * 100.0,
                cue.timing,
                cue.text.replace('\n', " / ")
            ));
        }
    }

    let mut report = String::new();
    report.push_str("sub-zero verify report\n");
    report.push_str(&format!("srt: {}\n", output_srt.display()));
    report.push_str(&format!("audio: {}\n", audio_wav.display()));
    report.push_str(&format!(
        "vad: threshold_db={threshold_db} min_silence={min_silence} pad={pad}\n"
    ));
    report.push_str(&format!("min_speech_overlap: {min_speech_overlap}\n\n"));

    if issues.is_empty() {
        report.push_str("status: ok\n");
    } else {
        report.push_str(&format!("status: issues={}\n", issues.len()));
        for issue in issues {
            report.push_str("- ");
            report.push_str(&issue);
            report.push('\n');
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::{
        apply_source_phrase_consistency, assess_translation_semantics,
        collect_low_confidence_cue_spans, collect_low_quality_scene_ranges,
        collect_low_quality_source_scene_ranges, compact_adjacent_cues,
        load_cue_asr_confidence_from_whisper_json, output_path_for_target_lang,
        shift_cues_by_offset, source_scene_quality_score, split_scene_ranges,
        write_parallel_confidence_sidecar, CueAsrConfidence, PipelineConfig, QualityProfile,
        SubtitlePipeline,
    };
    use crate::engine::chunker::AudioChunk;
    use crate::engine::srt::{parse_srt_file, SubtitleCue};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_case_dir(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time should be monotonic")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("sub_zero_{name}_{stamp}"));
        fs::create_dir_all(&path).expect("temp dir should be creatable");
        path
    }

    #[test]
    fn resolve_sidecar_for_video_path() {
        let dir = temp_case_dir("resolve_sidecar");
        let video = dir.join("sample.mkv");
        let srt = dir.join("sample.srt");

        fs::write(&video, "video").expect("video file should be writable");
        fs::write(&srt, "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n")
            .expect("srt should be writable");

        let pipeline = SubtitlePipeline::new(PipelineConfig {
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            offline: true,
            transcribe: false,
            whisper_bin: None,
            whisper_model: None,
            whisper_args: Vec::new(),
            skip_existing: false,
            vad: false,
            vad_threshold_db: -35.0,
            vad_min_silence: 0.35,
            vad_pad: 0.20,
            verify: false,
            verify_min_speech_overlap: 0.35,
            gpu: false,
            require_gpu: false,
            parallel: false,
            max_workers: 4,
            chunk_duration_secs: 300.0,
            force_phrase_table: true,
            mt_model: None,
            mt_batch_size: None,
            mt_max_batch_tokens: None,
            mt_oom_retries: None,
            mt_allow_cpu_fallback: true,
            auto_repair_sidecar: true,
            quality_profile: QualityProfile::Balanced,
        })
        .expect("pipeline should build");

        let (resolved, audio_for_verify) = pipeline
            .resolve_subtitle_source(&video)
            .expect("sidecar should resolve");
        assert_eq!(resolved, srt);
        assert!(audio_for_verify.is_none());
    }

    #[test]
    fn strict_profile_rejects_video_sidecar_without_transcribe() {
        let dir = temp_case_dir("strict_rejects_sidecar");
        let video = dir.join("sample.mkv");
        let srt = dir.join("sample.srt");

        fs::write(&video, "video").expect("video file should be writable");
        fs::write(&srt, "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n")
            .expect("srt should be writable");

        let pipeline = SubtitlePipeline::new(PipelineConfig {
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            offline: true,
            transcribe: false,
            whisper_bin: None,
            whisper_model: None,
            whisper_args: Vec::new(),
            skip_existing: false,
            vad: false,
            vad_threshold_db: -35.0,
            vad_min_silence: 0.35,
            vad_pad: 0.20,
            verify: false,
            verify_min_speech_overlap: 0.35,
            gpu: false,
            require_gpu: false,
            parallel: false,
            max_workers: 4,
            chunk_duration_secs: 300.0,
            force_phrase_table: true,
            mt_model: None,
            mt_batch_size: None,
            mt_max_batch_tokens: None,
            mt_oom_retries: None,
            mt_allow_cpu_fallback: true,
            auto_repair_sidecar: true,
            quality_profile: QualityProfile::Strict,
        })
        .expect("pipeline should build");

        let error = pipeline
            .resolve_subtitle_source(&video)
            .expect_err("strict should reject sidecar-only video path");
        assert!(error.contains("strict profile requires audio-first transcription"));
    }

    #[test]
    fn output_path_appends_target_lang() {
        let source = PathBuf::from("movie.srt");
        let out = output_path_for_target_lang(&source, "en").expect("path should build");
        assert_eq!(out, PathBuf::from("movie.en.srt"));
    }

    #[test]
    fn output_path_for_video_input() {
        let source = PathBuf::from("movie.mkv");
        let out = output_path_for_target_lang(&source, "en").expect("path should build");
        assert_eq!(out, PathBuf::from("movie.en.srt"));
    }

    #[test]
    fn process_file_translates_and_writes() {
        let dir = temp_case_dir("pipeline_translate");
        let source = dir.join("sample.srt");
        fs::write(
            &source,
            "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n\n2\n00:00:01,000 --> 00:00:02,000\nありがとう\n",
        )
        .expect("source srt should be writable");

        let pipeline = SubtitlePipeline::new(PipelineConfig {
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            offline: true,
            transcribe: false,
            whisper_bin: None,
            whisper_model: None,
            whisper_args: Vec::new(),
            skip_existing: false,
            vad: false,
            vad_threshold_db: -35.0,
            vad_min_silence: 0.35,
            vad_pad: 0.20,
            verify: false,
            verify_min_speech_overlap: 0.35,
            gpu: false,
            require_gpu: false,
            parallel: false,
            max_workers: 4,
            chunk_duration_secs: 300.0,
            force_phrase_table: true,
            mt_model: None,
            mt_batch_size: None,
            mt_max_batch_tokens: None,
            mt_oom_retries: None,
            mt_allow_cpu_fallback: true,
            auto_repair_sidecar: true,
            quality_profile: QualityProfile::Balanced,
        })
        .expect("pipeline should build");

        let output = pipeline
            .process_input(&source)
            .expect("process should succeed");
        let cues = parse_srt_file(&output).expect("translated output should parse");
        let metadata = dir.join("sample.sub-zero.json");
        let metadata_text = fs::read_to_string(&metadata).expect("metadata sidecar should exist");

        assert_eq!(cues[0].text, "hello");
        assert_eq!(cues[1].text, "thank you");
        assert!(metadata_text.contains("\"algorithm\": \"IBVoid DOOM-QLOCK\""));
    }

    #[test]
    fn assess_srt_health_detects_pathological_repetition() {
        let mut cues = Vec::<crate::engine::srt::SubtitleCue>::new();
        for i in 0..240usize {
            cues.push(crate::engine::srt::SubtitleCue {
                index: i + 1,
                timing: format!("00:00:{:02},000 --> 00:00:{:02},500", i % 60, (i % 60) + 1),
                text: "わかります".to_string(),
            });
        }

        let health = super::assess_srt_health(&cues).expect("health analysis should succeed");
        assert!(health.is_pathological(QualityProfile::Balanced));
        assert!(health.top_line_ratio > 0.7);
    }

    #[test]
    fn pathological_sidecar_fails_without_auto_repair() {
        let dir = temp_case_dir("pathological_sidecar");
        let video = dir.join("sample.mkv");
        let sidecar = dir.join("sample.srt");
        fs::write(&video, "video").expect("video file should be writable");

        let mut body = String::new();
        for i in 0..240usize {
            let start = i;
            let end = i + 1;
            body.push_str(&format!(
                "{}\n00:{:02}:{:02},000 --> 00:{:02}:{:02},000\nわかります\n\n",
                i + 1,
                (start / 60) % 60,
                start % 60,
                (end / 60) % 60,
                end % 60
            ));
        }
        fs::write(&sidecar, body).expect("pathological sidecar should be writable");

        let pipeline = SubtitlePipeline::new(PipelineConfig {
            source_lang: "ja".to_string(),
            target_lang: "en".to_string(),
            offline: true,
            transcribe: false,
            whisper_bin: None,
            whisper_model: None,
            whisper_args: Vec::new(),
            skip_existing: false,
            vad: false,
            vad_threshold_db: -35.0,
            vad_min_silence: 0.35,
            vad_pad: 0.20,
            verify: false,
            verify_min_speech_overlap: 0.35,
            gpu: false,
            require_gpu: false,
            parallel: false,
            max_workers: 4,
            chunk_duration_secs: 300.0,
            force_phrase_table: true,
            mt_model: None,
            mt_batch_size: None,
            mt_max_batch_tokens: None,
            mt_oom_retries: None,
            mt_allow_cpu_fallback: true,
            auto_repair_sidecar: false,
            quality_profile: QualityProfile::Balanced,
        })
        .expect("pipeline should build");

        let error = pipeline
            .process_input(&video)
            .expect_err("pathological sidecar should be rejected");
        assert!(error.contains("sidecar subtitles look degraded"));
    }

    #[test]
    fn semantic_quality_detects_malformed_english() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:02,000".to_string(),
                text: "I'm's going there now.".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:02,000 --> 00:00:04,000".to_string(),
                text: "I'm let you know right now.".to_string(),
            },
            SubtitleCue {
                index: 3,
                timing: "00:00:04,000 --> 00:00:06,000".to_string(),
                text: "Wait wait wait wait.".to_string(),
            },
        ];

        let health = assess_translation_semantics(&cues, "en");
        assert!(health.is_pathological(QualityProfile::Strict));
        assert!(health.malformed_contraction_ratio > 0.0);
    }

    #[test]
    fn semantic_quality_is_neutral_for_non_english_targets() {
        let cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:02,000".to_string(),
            text: "Bonjour tout le monde".to_string(),
        }];
        let health = assess_translation_semantics(&cues, "fr");
        assert_eq!(health.anomaly_ratio, 0.0);
        assert!(!health.is_pathological(QualityProfile::Strict));
    }

    #[test]
    fn semantic_quality_detects_name_inconsistency() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:02,000".to_string(),
                text: "Sakura, wait for me.".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:02,500 --> 00:00:04,000".to_string(),
                text: "Sakra is over there.".to_string(),
            },
            SubtitleCue {
                index: 3,
                timing: "00:00:04,500 --> 00:00:06,000".to_string(),
                text: "Sakura, this way!".to_string(),
            },
            SubtitleCue {
                index: 4,
                timing: "00:00:06,500 --> 00:00:08,000".to_string(),
                text: "Sakra, hurry up.".to_string(),
            },
        ];
        let health = assess_translation_semantics(&cues, "en");
        assert!(health.name_inconsistency_ratio > 0.0);
    }

    #[test]
    fn semantic_quality_tracks_scene_low_quality_ratio() {
        let mut cues = Vec::<SubtitleCue>::new();
        for i in 0..12usize {
            cues.push(SubtitleCue {
                index: i + 1,
                timing: format!("00:00:{:02},000 --> 00:00:{:02},900", i, i),
                text: "I'm's let go go go now now.".to_string(),
            });
        }
        // Scene boundary by gap.
        cues.push(SubtitleCue {
            index: 13,
            timing: "00:00:30,000 --> 00:00:32,000".to_string(),
            text: "Everything is fine now.".to_string(),
        });
        let health = assess_translation_semantics(&cues, "en");
        assert!(health.scene_count >= 2);
        assert!(health.scene_low_quality_ratio > 0.0);
    }

    #[test]
    fn split_scene_ranges_honors_gap_boundaries() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "One.".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,050 --> 00:00:02,000".to_string(),
                text: "Two.".to_string(),
            },
            SubtitleCue {
                index: 3,
                timing: "00:00:04,000 --> 00:00:05,000".to_string(),
                text: "Three.".to_string(),
            },
        ];
        let ranges = split_scene_ranges(&cues);
        assert_eq!(ranges, vec![(0, 2), (2, 3)]);
    }

    #[test]
    fn collect_low_quality_scene_ranges_flags_noisy_scene() {
        let mut cues = Vec::<SubtitleCue>::new();
        for i in 0..8usize {
            cues.push(SubtitleCue {
                index: i + 1,
                timing: format!("00:00:{:02},000 --> 00:00:{:02},900", i, i),
                text: "I'm's let go go go now now.".to_string(),
            });
        }
        cues.push(SubtitleCue {
            index: 9,
            timing: "00:00:20,000 --> 00:00:22,000".to_string(),
            text: "Everything is fine now.".to_string(),
        });
        let low = collect_low_quality_scene_ranges(&cues);
        assert!(!low.is_empty());
        assert_eq!(low[0].start, 0);
        assert!(low[0].end >= 8);
    }

    #[test]
    fn collect_low_quality_source_scene_ranges_flags_repetitive_source() {
        let mut cues = Vec::<SubtitleCue>::new();
        for i in 0..10usize {
            cues.push(SubtitleCue {
                index: i + 1,
                timing: format!("00:00:{:02},000 --> 00:00:{:02},180", i, i),
                text: "えええ".to_string(),
            });
        }
        cues.push(SubtitleCue {
            index: 11,
            timing: "00:00:30,000 --> 00:00:32,200".to_string(),
            text: "大丈夫だよ".to_string(),
        });

        let low = collect_low_quality_source_scene_ranges(&cues);
        assert!(!low.is_empty());
        assert_eq!(low[0].start, 0);
        assert!(low[0].score < low[0].floor);
    }

    #[test]
    fn shift_cues_by_offset_moves_timing_forward() {
        let cues = vec![SubtitleCue {
            index: 1,
            timing: "00:00:01,250 --> 00:00:02,500".to_string(),
            text: "hello".to_string(),
        }];
        let shifted = shift_cues_by_offset(&cues, 10.0).expect("shift should succeed");
        assert_eq!(shifted[0].timing, "00:00:11,250 --> 00:00:12,500");
    }

    #[test]
    fn source_scene_quality_score_prefers_clean_scene() {
        let noisy = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:00,120".to_string(),
                text: "え".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:00,120 --> 00:00:00,240".to_string(),
                text: "え".to_string(),
            },
        ];
        let clean = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,200".to_string(),
                text: "こんにちは".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,400 --> 00:00:02,600".to_string(),
                text: "ありがとうございます".to_string(),
            },
        ];
        assert!(source_scene_quality_score(&clean) > source_scene_quality_score(&noisy));
    }

    #[test]
    fn compact_adjacent_cues_merges_short_continuations() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "I think".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,050 --> 00:00:02,000".to_string(),
                text: "we should go.".to_string(),
            },
        ];
        let (compacted, stats) =
            compact_adjacent_cues(&cues, 0.2, 42, 2, 21.0, 7.0).expect("compaction should succeed");
        assert_eq!(compacted.len(), 1);
        assert_eq!(stats.merged_pairs, 1);
        assert_eq!(compacted[0].text, "I think we should go.");
    }

    #[test]
    fn compact_adjacent_cues_dedupes_stutter_duplicates() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "Run!".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,050 --> 00:00:01,500".to_string(),
                text: "Run!".to_string(),
            },
        ];
        let (compacted, stats) =
            compact_adjacent_cues(&cues, 0.2, 42, 2, 21.0, 7.0).expect("compaction should succeed");
        assert_eq!(compacted.len(), 1);
        assert_eq!(stats.dropped_duplicates, 1);
    }

    #[test]
    fn compact_adjacent_cues_keeps_large_gap_separate() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "Hello".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,600 --> 00:00:02,400".to_string(),
                text: "world".to_string(),
            },
        ];
        let (compacted, _) =
            compact_adjacent_cues(&cues, 0.2, 42, 2, 21.0, 7.0).expect("compaction should succeed");
        assert_eq!(compacted.len(), 2);
    }

    #[test]
    fn collect_low_confidence_cue_spans_groups_adjacent_low_confidence() {
        let cues = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "a".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,050 --> 00:00:02,000".to_string(),
                text: "b".to_string(),
            },
            SubtitleCue {
                index: 3,
                timing: "00:00:02,050 --> 00:00:03,000".to_string(),
                text: "c".to_string(),
            },
            SubtitleCue {
                index: 4,
                timing: "00:00:03,100 --> 00:00:04,000".to_string(),
                text: "d".to_string(),
            },
        ];
        let confidence = vec![
            Some(CueAsrConfidence {
                score: 0.72,
                avg_logprob: -0.35,
                no_speech_prob: 0.02,
                compression_ratio: 1.1,
                word_prob_mean: 0.86,
                low_word_prob_ratio: 0.0,
                suspicious: false,
            }),
            Some(CueAsrConfidence {
                score: 0.40,
                avg_logprob: -1.45,
                no_speech_prob: 0.09,
                compression_ratio: 1.2,
                word_prob_mean: 0.38,
                low_word_prob_ratio: 0.66,
                suspicious: false,
            }),
            Some(CueAsrConfidence {
                score: 0.44,
                avg_logprob: -1.20,
                no_speech_prob: 0.10,
                compression_ratio: 1.4,
                word_prob_mean: 0.41,
                low_word_prob_ratio: 0.50,
                suspicious: true,
            }),
            Some(CueAsrConfidence {
                score: 0.78,
                avg_logprob: -0.25,
                no_speech_prob: 0.02,
                compression_ratio: 1.1,
                word_prob_mean: 0.91,
                low_word_prob_ratio: 0.0,
                suspicious: false,
            }),
        ];
        let spans = collect_low_confidence_cue_spans(&cues, &confidence, QualityProfile::Strict);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start, 1);
        assert_eq!(spans[0].end, 3);
    }

    #[test]
    fn source_phrase_consistency_rewrites_minor_variants() {
        let source = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "さくらはどこ？".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,200 --> 00:00:02,000".to_string(),
                text: "さくらはどこ？".to_string(),
            },
            SubtitleCue {
                index: 3,
                timing: "00:00:02,200 --> 00:00:03,000".to_string(),
                text: "さくらはどこ？".to_string(),
            },
            SubtitleCue {
                index: 4,
                timing: "00:00:03,200 --> 00:00:04,000".to_string(),
                text: "さくらはどこ？".to_string(),
            },
        ];
        let mut translated = vec![
            SubtitleCue {
                index: 1,
                timing: "00:00:00,000 --> 00:00:01,000".to_string(),
                text: "Where is Sakura?".to_string(),
            },
            SubtitleCue {
                index: 2,
                timing: "00:00:01,200 --> 00:00:02,000".to_string(),
                text: "Where is Sakura?".to_string(),
            },
            SubtitleCue {
                index: 3,
                timing: "00:00:02,200 --> 00:00:03,000".to_string(),
                text: "Where is Sakra?".to_string(),
            },
            SubtitleCue {
                index: 4,
                timing: "00:00:03,200 --> 00:00:04,000".to_string(),
                text: "Where is Sakura?".to_string(),
            },
        ];
        let stats = apply_source_phrase_consistency(&source, &mut translated);
        assert!(stats.source_clusters >= 1);
        assert!(stats.rewritten_cues >= 1);
        assert_eq!(translated[2].text, "Where is Sakura?");
    }

    #[test]
    fn load_cue_asr_confidence_from_whisper_json_maps_scores() {
        let dir = temp_case_dir("asr_confidence_json");
        let srt_path = dir.join("clip.srt");
        let json_path = dir.join("clip.json");
        fs::write(
            &srt_path,
            "1\n00:00:00,000 --> 00:00:01,000\na\n\n2\n00:00:01,000 --> 00:00:02,000\nb\n",
        )
        .expect("srt should be writable");
        fs::write(
            &json_path,
            r#"{
  "segments": [
    {"start": 0.0, "end": 1.0, "avg_logprob": -0.20, "no_speech_prob": 0.01, "compression_ratio": 1.1},
    {"start": 1.0, "end": 2.0, "avg_logprob": -1.50, "no_speech_prob": 0.15, "compression_ratio": 1.3}
  ]
}"#,
        )
        .expect("json should be writable");

        let cues = parse_srt_file(&srt_path).expect("cues should parse");
        let confidence = load_cue_asr_confidence_from_whisper_json(&srt_path, &cues)
            .expect("confidence should parse");
        assert_eq!(confidence.len(), 2);
        assert!(
            confidence[0]
                .expect("first cue confidence must exist")
                .score
                > confidence[1]
                    .expect("second cue confidence must exist")
                    .score
        );
    }

    #[test]
    fn write_parallel_confidence_sidecar_merges_chunk_offsets() {
        let dir = temp_case_dir("parallel_confidence_merge");
        let chunk0_wav = dir.join("chunk_000.wav");
        let chunk1_wav = dir.join("chunk_001.wav");
        fs::write(&chunk0_wav, "").expect("chunk wav should be writable");
        fs::write(&chunk1_wav, "").expect("chunk wav should be writable");
        fs::write(
            chunk0_wav.with_extension("json"),
            r#"{"segments":[{"start":0.0,"end":1.0,"avg_logprob":-0.2}]}"#,
        )
        .expect("chunk json should be writable");
        fs::write(
            chunk1_wav.with_extension("json"),
            r#"{"segments":[{"start":2.0,"end":3.0,"avg_logprob":-0.8}]}"#,
        )
        .expect("chunk json should be writable");

        let chunks = vec![
            AudioChunk {
                index: 0,
                start_sec: 0.0,
                end_sec: 5.0,
                wav_path: chunk0_wav,
                overlap_before: 0.0,
                overlap_after: 0.0,
            },
            AudioChunk {
                index: 1,
                start_sec: 100.0,
                end_sec: 110.0,
                wav_path: chunk1_wav,
                overlap_before: 0.0,
                overlap_after: 0.0,
            },
        ];
        let stitched = dir.join("stitched.ja.srt");
        write_parallel_confidence_sidecar(&chunks, &stitched).expect("merge should succeed");

        let merged = fs::read_to_string(stitched.with_extension("json"))
            .expect("merged confidence sidecar should exist");
        let parsed: serde_json::Value =
            serde_json::from_str(&merged).expect("merged sidecar should parse");
        let segments = parsed
            .get("segments")
            .and_then(serde_json::Value::as_array)
            .expect("segments should exist");
        assert_eq!(segments.len(), 2);
        assert_eq!(
            segments[0]
                .get("start")
                .and_then(serde_json::Value::as_f64)
                .expect("start should exist"),
            0.0
        );
        assert_eq!(
            segments[1]
                .get("start")
                .and_then(serde_json::Value::as_f64)
                .expect("start should exist"),
            102.0
        );
    }
}
