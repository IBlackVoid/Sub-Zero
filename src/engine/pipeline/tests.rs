use super::scenes::split_scene_ranges;
use super::{
    apply_source_phrase_consistency, apply_source_phrase_consistency_by_speaker,
    assess_translation_semantics, collect_low_confidence_cue_spans,
    collect_low_quality_scene_ranges, collect_low_quality_source_scene_ranges,
    compact_adjacent_cues, load_cue_asr_confidence_from_whisper_json, output_path_for_target_lang,
    shift_cues_by_offset, source_scene_quality_score, write_parallel_confidence_sidecar,
    CueAsrConfidence, PipelineConfig, QualityProfile, SubtitlePipeline,
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
    let path = std::env::temp_dir().join(format!("voidex_{name}_{stamp}"));
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
        audio_lang: None,
        audio_stream_index: None,
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
        stream: false,
        stream_async: false,
        max_workers: 4,
        chunk_duration_secs: 300.0,
        force_phrase_table: true,
        speaker_aware: false,
        speaker_diarize: false,
        speaker_max_speakers: 4,
        mt_model: None,
        mt_batch_size: None,
        mt_max_batch_tokens: None,
        mt_oom_retries: None,
        mt_allow_cpu_fallback: true,
        mt_force_cpu: false,
        mt_daemon: false,
        mt_enforce_quality_floor: true,
        auto_repair_sidecar: true,
        trace_runtime: false,
        events_json: false,
        events_file: None,
        http_events: None,
        ws_events: None,
        quality_profile: QualityProfile::Balanced,
    })
    .expect("pipeline should build");

    let events =
        crate::engine::events::EventSink::new(false, None, None, None).expect("event sink");
    let (resolved, audio_for_verify) = pipeline
        .resolve_subtitle_source(&video, &events)
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
        audio_lang: None,
        audio_stream_index: None,
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
        stream: false,
        stream_async: false,
        max_workers: 4,
        chunk_duration_secs: 300.0,
        force_phrase_table: true,
        speaker_aware: false,
        speaker_diarize: false,
        speaker_max_speakers: 4,
        mt_model: None,
        mt_batch_size: None,
        mt_max_batch_tokens: None,
        mt_oom_retries: None,
        mt_allow_cpu_fallback: true,
        mt_force_cpu: false,
        mt_daemon: false,
        mt_enforce_quality_floor: true,
        auto_repair_sidecar: true,
        trace_runtime: false,
        events_json: false,
        events_file: None,
        http_events: None,
        ws_events: None,
        quality_profile: QualityProfile::Strict,
    })
    .expect("pipeline should build");

    let events =
        crate::engine::events::EventSink::new(false, None, None, None).expect("event sink");
    let error = pipeline
        .resolve_subtitle_source(&video, &events)
        .expect_err("strict should reject sidecar-only video path");
    assert!(error
        .to_string()
        .contains("strict profile requires audio-first transcription"));
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
        audio_lang: None,
        audio_stream_index: None,
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
        stream: false,
        stream_async: false,
        max_workers: 4,
        chunk_duration_secs: 300.0,
        force_phrase_table: true,
        speaker_aware: false,
        speaker_diarize: false,
        speaker_max_speakers: 4,
        mt_model: None,
        mt_batch_size: None,
        mt_max_batch_tokens: None,
        mt_oom_retries: None,
        mt_allow_cpu_fallback: true,
        mt_force_cpu: false,
        mt_daemon: false,
        mt_enforce_quality_floor: true,
        auto_repair_sidecar: true,
        trace_runtime: false,
        events_json: false,
        events_file: None,
        http_events: None,
        ws_events: None,
        quality_profile: QualityProfile::Balanced,
    })
    .expect("pipeline should build");

    let output = pipeline
        .process_input(&source)
        .expect("process should succeed");
    let cues = parse_srt_file(&output).expect("translated output should parse");
    let metadata = dir.join("sample.voidex.json");
    let metadata_text = fs::read_to_string(&metadata).expect("metadata sidecar should exist");

    // postprocess() runs fix_capitalization, which intentionally capitalizes
    // sentence starts (see fix_capitalization_works / postprocess_full_pipeline).
    assert_eq!(cues[0].text, "Hello");
    assert_eq!(cues[1].text, "Thank you");
    assert!(metadata_text.contains("\"algorithm\": \"IBVoid DOOM-QLOCK\""));
}

#[test]
fn process_file_emits_runtime_trace_when_enabled() {
    let dir = temp_case_dir("pipeline_trace");
    let source = dir.join("sample.srt");
    fs::write(
        &source,
        "1\n00:00:00,000 --> 00:00:01,000\nã“ã‚“ã«ã¡ã¯\n\n2\n00:00:01,000 --> 00:00:02,000\nã‚ã‚ŠãŒã¨ã†\n",
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
        audio_lang: None,
        audio_stream_index: None,
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
        stream: false,
        stream_async: false,
        max_workers: 4,
        chunk_duration_secs: 300.0,
        force_phrase_table: true,
        speaker_aware: false,
        speaker_diarize: false,
        speaker_max_speakers: 4,
        mt_model: None,
        mt_batch_size: None,
        mt_max_batch_tokens: None,
        mt_oom_retries: None,
        mt_allow_cpu_fallback: true,
        mt_force_cpu: false,
        mt_daemon: false,
        mt_enforce_quality_floor: true,
        auto_repair_sidecar: true,
        trace_runtime: true,
        events_json: false,
        events_file: None,
        http_events: None,
        ws_events: None,
        quality_profile: QualityProfile::Balanced,
    })
    .expect("pipeline should build");

    let output = pipeline
        .process_input(&source)
        .expect("process should succeed");
    let metadata = dir.join("sample.voidex.json");
    let trace_path = dir.join("sample.voidex.trace.json");

    let metadata_text = fs::read_to_string(&metadata).expect("metadata sidecar should exist");
    let trace_text = fs::read_to_string(&trace_path).expect("trace sidecar should exist");
    let trace_json: serde_json::Value =
        serde_json::from_str(&trace_text).expect("trace sidecar should parse");
    let stages = trace_json
        .get("stages")
        .and_then(serde_json::Value::as_array)
        .expect("trace stages should exist");

    assert_eq!(output, dir.join("sample.en.srt"));
    assert!(metadata_text.contains("\"runtime_trace_file\""));
    assert!(metadata_text.contains("sample.voidex.trace.json"));
    assert_eq!(
        trace_json
            .get("trace_kind")
            .and_then(serde_json::Value::as_str),
        Some("runtime-performance")
    );
    assert!(
        stages.iter().any(|stage| {
            stage.get("name").and_then(serde_json::Value::as_str) == Some("translate")
        }),
        "trace should include translate stage"
    );
    assert!(
        stages.iter().any(|stage| {
            stage.get("name").and_then(serde_json::Value::as_str) == Some("write_output_srt")
        }),
        "trace should include write_output_srt stage"
    );
}

/// End-to-end check that F.2 W2-b (voice consistency) is wired into the
/// pipeline and surfaces in the user-facing sidecar JSON. The phrase-table
/// backend keeps this test offline and deterministic; speaker_aware is on,
/// so each speaker accumulates a prior and the sidecar reports
/// `voice_consistency` with non-zero `speakers_observed`.
#[test]
fn process_file_emits_voice_consistency_sidecar() {
    let dir = temp_case_dir("pipeline_voice");
    let source = dir.join("sample.srt");
    // Two cues per speaker so the singleton-filter in infer_speakers()
    // promotes them. Phrase-table maps こんにちは→hello, ありがとう→thank you.
    // `Alice: ...` / `Bob: ...` are the colon-prefix form recognised by
    // `infer_speaker_from_text`. Two cues per speaker so the singleton
    // filter promotes them.
    fs::write(
        &source,
        "1\n00:00:00,000 --> 00:00:01,000\nAlice: こんにちは\n\n\
         2\n00:00:01,000 --> 00:00:02,000\nAlice: ありがとう\n\n\
         3\n00:00:02,000 --> 00:00:03,000\nBob: こんにちは\n\n\
         4\n00:00:03,000 --> 00:00:04,000\nBob: ありがとう\n",
    )
    .expect("source srt should be writable");

    // Isolate voice priors store per test to avoid cross-test contamination.
    std::env::set_var("VOIDEX_HOME", &dir);

    let pipeline = SubtitlePipeline::new(PipelineConfig {
        source_lang: "ja".to_string(),
        target_lang: "en".to_string(),
        offline: true,
        transcribe: false,
        whisper_bin: None,
        whisper_model: None,
        whisper_args: Vec::new(),
        audio_lang: None,
        audio_stream_index: None,
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
        stream: false,
        stream_async: false,
        max_workers: 4,
        chunk_duration_secs: 300.0,
        force_phrase_table: true,
        speaker_aware: true,
        speaker_diarize: false,
        speaker_max_speakers: 4,
        mt_model: None,
        mt_batch_size: None,
        mt_max_batch_tokens: None,
        mt_oom_retries: None,
        mt_allow_cpu_fallback: true,
        mt_force_cpu: false,
        mt_daemon: false,
        mt_enforce_quality_floor: false,
        auto_repair_sidecar: true,
        trace_runtime: false,
        events_json: false,
        events_file: None,
        http_events: None,
        ws_events: None,
        quality_profile: QualityProfile::Balanced,
    })
    .expect("pipeline should build");

    pipeline
        .process_input(&source)
        .expect("process should succeed");
    let metadata = dir.join("sample.voidex.json");
    let metadata_text = fs::read_to_string(&metadata).expect("metadata sidecar should exist");
    let metadata_json: serde_json::Value =
        serde_json::from_str(&metadata_text).expect("metadata should parse");
    let voice = metadata_json
        .get("voice_consistency")
        .expect("sidecar should expose voice_consistency block");
    let speakers_observed = voice
        .get("speakers_observed")
        .and_then(serde_json::Value::as_u64)
        .expect("speakers_observed should be present");
    assert_eq!(
        speakers_observed, 2,
        "Alice and Bob should both be observed: {voice}"
    );
    // Voice priors should have persisted to the isolated VOIDEX_HOME.
    let priors_path = dir.join("voice_priors.json");
    assert!(
        priors_path.is_file(),
        "voice_priors.json should be persisted to VOIDEX_HOME"
    );

    std::env::remove_var("VOIDEX_HOME");
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

    // A degraded sidecar whose dominant line saturates the transcript
    // (high top_line_ratio) but is *interleaved* with distinct filler so
    // no run of identical cues is long enough to trip the upstream
    // decode-loop collapse. This keeps the test exercising the health
    // gate / rescue path rather than the loop-collapse path: every other
    // cue is the same hallucinated phrase (top_line_ratio ~= 0.5), but
    // the longest consecutive identical run is 1.
    let mut body = String::new();
    for i in 0..300usize {
        let start = i;
        let end = i + 1;
        let text = if i % 2 == 0 {
            "わかります".to_string()
        } else {
            format!("フィラー{i}")
        };
        body.push_str(&format!(
            "{}\n00:{:02}:{:02},000 --> 00:{:02}:{:02},000\n{}\n\n",
            i + 1,
            (start / 60) % 60,
            start % 60,
            (end / 60) % 60,
            end % 60,
            text
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
        audio_lang: None,
        audio_stream_index: None,
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
        stream: false,
        stream_async: false,
        max_workers: 4,
        chunk_duration_secs: 300.0,
        force_phrase_table: true,
        speaker_aware: false,
        speaker_diarize: false,
        speaker_max_speakers: 4,
        mt_model: None,
        mt_batch_size: None,
        mt_max_batch_tokens: None,
        mt_oom_retries: None,
        mt_allow_cpu_fallback: true,
        mt_force_cpu: false,
        mt_daemon: false,
        mt_enforce_quality_floor: true,
        auto_repair_sidecar: false,
        trace_runtime: false,
        events_json: false,
        events_file: None,
        http_events: None,
        ws_events: None,
        quality_profile: QualityProfile::Balanced,
    })
    .expect("pipeline should build");

    let error = pipeline
        .process_input(&video)
        .expect_err("pathological sidecar should be rejected");
    assert!(error
        .to_string()
        .contains("sidecar subtitles look degraded"));
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
fn source_phrase_consistency_by_speaker_isolated_clusters() {
    let source = vec![
        SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: "xx".to_string(),
        },
        SubtitleCue {
            index: 2,
            timing: "00:00:01,000 --> 00:00:02,000".to_string(),
            text: "xx".to_string(),
        },
        SubtitleCue {
            index: 3,
            timing: "00:00:02,000 --> 00:00:03,000".to_string(),
            text: "xx".to_string(),
        },
        SubtitleCue {
            index: 4,
            timing: "00:00:03,000 --> 00:00:04,000".to_string(),
            text: "xx".to_string(),
        },
        SubtitleCue {
            index: 5,
            timing: "00:00:04,000 --> 00:00:05,000".to_string(),
            text: "xx".to_string(),
        },
        SubtitleCue {
            index: 6,
            timing: "00:00:05,000 --> 00:00:06,000".to_string(),
            text: "xx".to_string(),
        },
    ];
    let mut translated = vec![
        SubtitleCue {
            index: 1,
            timing: "00:00:00,000 --> 00:00:01,000".to_string(),
            text: "Hello there.".to_string(),
        },
        SubtitleCue {
            index: 2,
            timing: "00:00:01,000 --> 00:00:02,000".to_string(),
            text: "Hello there.".to_string(),
        },
        SubtitleCue {
            index: 3,
            timing: "00:00:02,000 --> 00:00:03,000".to_string(),
            text: "Helo there.".to_string(),
        },
        SubtitleCue {
            index: 4,
            timing: "00:00:03,000 --> 00:00:04,000".to_string(),
            text: "Good day.".to_string(),
        },
        SubtitleCue {
            index: 5,
            timing: "00:00:04,000 --> 00:00:05,000".to_string(),
            text: "Good day.".to_string(),
        },
        SubtitleCue {
            index: 6,
            timing: "00:00:05,000 --> 00:00:06,000".to_string(),
            text: "Gd day.".to_string(),
        },
    ];
    let speakers = vec![
        Some("alice".to_string()),
        Some("alice".to_string()),
        Some("alice".to_string()),
        Some("bob".to_string()),
        Some("bob".to_string()),
        Some("bob".to_string()),
    ];

    let stats = apply_source_phrase_consistency_by_speaker(&source, &mut translated, &speakers);
    assert_eq!(stats.speakers, 2);
    assert!(stats.source_clusters >= 2);
    assert!(stats.rewritten_cues >= 2);
    assert_eq!(translated[2].text, "Hello there.");
    assert_eq!(translated[5].text, "Good day.");
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

#[test]
fn escalation_backend_caches_one_translator_per_rung() {
    // P0.2 regression guard: before the fix, every failing scene built a fresh
    // Translator (fresh interpreter + multi-GB model load each). The backend
    // must reuse one Translator per rung label across all scenes routed to it.
    use super::escalation::{BackendKind, MtBackendStep, SegmentBackend};
    use super::NeuralSegmentBackend;

    let pipeline = SubtitlePipeline::new(PipelineConfig {
        source_lang: "ja".to_string(),
        target_lang: "en".to_string(),
        offline: true,
        transcribe: false,
        whisper_bin: None,
        whisper_model: None,
        whisper_args: Vec::new(),
        audio_lang: None,
        audio_stream_index: None,
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
        stream: false,
        stream_async: false,
        max_workers: 4,
        chunk_duration_secs: 300.0,
        // Phrase-table keeps the test hermetic: no Python, no models, no GPU.
        force_phrase_table: true,
        speaker_aware: false,
        speaker_diarize: false,
        speaker_max_speakers: 4,
        mt_model: None,
        mt_batch_size: None,
        mt_max_batch_tokens: None,
        mt_oom_retries: None,
        mt_allow_cpu_fallback: true,
        mt_force_cpu: false,
        mt_daemon: false,
        mt_enforce_quality_floor: true,
        auto_repair_sidecar: true,
        trace_runtime: false,
        events_json: false,
        events_file: None,
        http_events: None,
        ws_events: None,
        quality_profile: QualityProfile::Balanced,
    })
    .expect("pipeline should build");

    let backend = NeuralSegmentBackend::new(&pipeline);
    let rung = MtBackendStep {
        label: "nllb-1.3B",
        model_name: "nllb-200-distilled-1.3B".to_string(),
        beam_size: 6,
        kind: BackendKind::Nllb,
    };
    let scene = vec![SubtitleCue {
        index: 1,
        timing: "00:00:00,000 --> 00:00:01,000".to_string(),
        text: "こんにちは".to_string(),
    }];

    // Two scenes routed to the same rung → exactly one Translator built.
    backend
        .translate_segment(&rung, &scene)
        .expect("first scene should translate");
    backend
        .translate_segment(&rung, &scene)
        .expect("second scene should translate");
    assert_eq!(
        backend.translators.borrow().len(),
        1,
        "same rung must reuse one cached Translator"
    );

    // A different rung gets its own Translator — and only one.
    let other_rung = MtBackendStep {
        label: "nllb-600M",
        model_name: "nllb-200-distilled-600M".to_string(),
        beam_size: 4,
        kind: BackendKind::Nllb,
    };
    backend
        .translate_segment(&other_rung, &scene)
        .expect("other rung should translate");
    assert_eq!(
        backend.translators.borrow().len(),
        2,
        "each rung owns exactly one cached Translator"
    );
}
