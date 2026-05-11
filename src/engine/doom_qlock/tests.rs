use super::knowledge::build_knowledge_snapshot;
use super::plan::{
    choose_mt_tuning, default_mt_batch_for_profile, default_mt_max_tokens_for_profile,
    duration_to_chunk_secs,
};
use super::probe::{parse_ffprobe_output, GpuProbe, HardwareProbe};
use super::{
    assess_output_health, ExecutionPlan, HistoryStore, PlanLookupQuery, RunRecord, WorkloadEstimate,
};
use crate::engine::pipeline::PipelineConfig;
use crate::engine::transcribe::QualityProfile;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn sample_config() -> PipelineConfig {
    PipelineConfig {
        source_lang: "ja".to_string(),
        target_lang: "en".to_string(),
        offline: true,
        transcribe: true,
        whisper_bin: None,
        whisper_model: None,
        whisper_args: Vec::new(),
        audio_lang: None,
        audio_stream_index: None,
        skip_existing: false,
        vad: true,
        vad_threshold_db: -35.0,
        vad_min_silence: 0.35,
        vad_pad: 0.20,
        verify: false,
        verify_min_speech_overlap: 0.35,
        gpu: true,
        require_gpu: false,
        parallel: false,
        stream: false,
        stream_async: false,
        max_workers: 12,
        chunk_duration_secs: 300.0,
        force_phrase_table: false,
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
    }
}

fn temp_case_dir(name: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after UNIX_EPOCH")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("sub_zero_doom_qlock_{name}_{stamp}"));
    fs::create_dir_all(&path).expect("temp directory should be creatable");
    path
}

#[test]
fn heuristic_plan_scales_mt_for_low_vram_gpu() {
    let base = sample_config();
    let hardware = HardwareProbe {
        cpu_cores: 16,
        total_ram_mb: Some(32_768),
        disk_write_mbps: Some(450.0),
        gpu: Some(GpuProbe {
            backend: "cuda".to_string(),
            name: "RTX 3050".to_string(),
            vram_mb: Some(4_096),
            compute_capability: Some("8.6".to_string()),
        }),
    };
    let workload = WorkloadEstimate {
        input_kind: "matroska,webm".to_string(),
        is_video: true,
        duration_secs: Some(6_000.0),
        speech_duration_secs: Some(5_400.0),
        estimated_cues: 2_000,
        avg_difficulty: Some(0.45),
        speaker_complexity: Some(0.35),
        energy_variance: Some(0.25),
        scene_count: Some(32),
        detected_source_lang: "ja".to_string(),
        detected_confidence: 0.7,
        detection_mode: "hint".to_string(),
    };

    let plan = ExecutionPlan::heuristic(&base, &hardware, &workload);
    assert!(plan.parallel);
    assert!(plan.workers <= 12);
    assert_eq!(plan.chunk_duration_secs, 300.0);
    assert!(plan.mt_batch_size.expect("batch should be set") <= 16);
    assert!(plan.mt_max_batch_tokens.expect("max tokens should be set") <= 4096);
}

#[test]
fn mt_tuning_is_backend_aware() {
    let cuda_hw = HardwareProbe {
        cpu_cores: 8,
        total_ram_mb: Some(16_384),
        disk_write_mbps: Some(800.0),
        gpu: Some(GpuProbe {
            backend: "cuda".to_string(),
            name: "RTX".to_string(),
            vram_mb: Some(8_192),
            compute_capability: Some("8.6".to_string()),
        }),
    };
    let metal_hw = HardwareProbe {
        cpu_cores: 8,
        total_ram_mb: Some(16_384),
        disk_write_mbps: Some(800.0),
        gpu: Some(GpuProbe {
            backend: "metal".to_string(),
            name: "Apple GPU".to_string(),
            vram_mb: Some(8_192),
            compute_capability: None,
        }),
    };

    let cuda = choose_mt_tuning(&cuda_hw, QualityProfile::Strict, true);
    let metal = choose_mt_tuning(&metal_hw, QualityProfile::Strict, true);
    assert!(metal.batch_size < cuda.batch_size);
    assert!(metal.max_batch_tokens < cuda.max_batch_tokens);
    assert!(metal.oom_retries >= cuda.oom_retries);
}

#[test]
fn history_prefers_fastest_successful_plan() {
    let plan_fast = ExecutionPlan {
        parallel: true,
        workers: 8,
        chunk_duration_secs: 240.0,
        mt_batch_size: Some(16),
        mt_max_batch_tokens: Some(4096),
        mt_oom_retries: Some(2),
        mt_allow_cpu_fallback: true,
    };
    let plan_slow = ExecutionPlan {
        parallel: true,
        workers: 4,
        chunk_duration_secs: 300.0,
        mt_batch_size: Some(8),
        mt_max_batch_tokens: Some(2048),
        mt_oom_retries: Some(4),
        mt_allow_cpu_fallback: true,
    };

    let mut history = HistoryStore::default();
    history.push_record(RunRecord {
        timestamp_epoch_secs: 1,
        device_fingerprint: "dev-1".to_string(),
        content_profile_hash: "content-a".to_string(),
        gpu_backend: Some("cuda".to_string()),
        gpu_vram_mb: Some(8_192),
        cpu_cores: Some(8),
        total_ram_mb: Some(16_384),
        disk_write_mbps: Some(400.0),
        source_lang: "ja".to_string(),
        target_lang: "en".to_string(),
        quality_profile: "strict".to_string(),
        input_kind: "srt".to_string(),
        audio_duration_secs: Some(120.0),
        speech_duration_secs: Some(115.0),
        estimated_cues: 40,
        avg_difficulty: Some(0.3),
        speaker_complexity: Some(0.1),
        energy_variance: Some(0.1),
        scene_count: Some(1),
        plan: plan_slow.clone(),
        success: true,
        elapsed_secs: 20.0,
        output_cues: Some(40),
        output_non_empty_ratio: Some(1.0),
        output_top_line_ratio: Some(0.10),
        error: None,
    });
    history.push_record(RunRecord {
        timestamp_epoch_secs: 2,
        device_fingerprint: "dev-1".to_string(),
        content_profile_hash: "content-a".to_string(),
        gpu_backend: Some("cuda".to_string()),
        gpu_vram_mb: Some(8_192),
        cpu_cores: Some(8),
        total_ram_mb: Some(16_384),
        disk_write_mbps: Some(400.0),
        source_lang: "ja".to_string(),
        target_lang: "en".to_string(),
        quality_profile: "strict".to_string(),
        input_kind: "srt".to_string(),
        audio_duration_secs: Some(120.0),
        speech_duration_secs: Some(115.0),
        estimated_cues: 40,
        avg_difficulty: Some(0.3),
        speaker_complexity: Some(0.1),
        energy_variance: Some(0.1),
        scene_count: Some(1),
        plan: plan_fast.clone(),
        success: true,
        elapsed_secs: 8.0,
        output_cues: Some(40),
        output_non_empty_ratio: Some(0.99),
        output_top_line_ratio: Some(0.08),
        error: None,
    });

    let selected = history.best_plan_exact(
        "dev-1",
        PlanLookupQuery {
            source_lang: "ja",
            target_lang: "en",
            profile: QualityProfile::Strict,
            input_kind: "srt",
            content_profile_hash: "content-a",
        },
    );
    assert_eq!(selected.expect("plan should be found"), plan_fast);
}

#[test]
fn history_similar_plan_fallback_uses_compatible_hardware() {
    let plan_gpu = ExecutionPlan {
        parallel: true,
        workers: 6,
        chunk_duration_secs: 240.0,
        mt_batch_size: Some(12),
        mt_max_batch_tokens: Some(3072),
        mt_oom_retries: Some(3),
        mt_allow_cpu_fallback: true,
    };
    let plan_cpu = ExecutionPlan {
        parallel: false,
        workers: 4,
        chunk_duration_secs: 300.0,
        mt_batch_size: Some(8),
        mt_max_batch_tokens: Some(2048),
        mt_oom_retries: Some(4),
        mt_allow_cpu_fallback: true,
    };
    let mut history = HistoryStore::default();
    history.push_record(RunRecord {
        timestamp_epoch_secs: 3,
        device_fingerprint: "cpu8-ram32768-gpucuda-rtx-3060-vram8192-disk420".to_string(),
        content_profile_hash: "content-old".to_string(),
        gpu_backend: Some("cuda".to_string()),
        gpu_vram_mb: Some(8_192),
        cpu_cores: Some(8),
        total_ram_mb: Some(32_768),
        disk_write_mbps: Some(420.0),
        source_lang: "ja".to_string(),
        target_lang: "en".to_string(),
        quality_profile: "strict".to_string(),
        input_kind: "srt".to_string(),
        audio_duration_secs: Some(600.0),
        speech_duration_secs: Some(540.0),
        estimated_cues: 180,
        avg_difficulty: Some(0.45),
        speaker_complexity: Some(0.25),
        energy_variance: Some(0.20),
        scene_count: Some(12),
        plan: plan_gpu.clone(),
        success: true,
        elapsed_secs: 42.0,
        output_cues: Some(180),
        output_non_empty_ratio: Some(0.98),
        output_top_line_ratio: Some(0.05),
        error: None,
    });
    history.push_record(RunRecord {
        timestamp_epoch_secs: 4,
        device_fingerprint: "cpu8-ram32768-gpunone-vram0-disk390".to_string(),
        content_profile_hash: "content-old".to_string(),
        gpu_backend: Some("none".to_string()),
        gpu_vram_mb: Some(0),
        cpu_cores: Some(8),
        total_ram_mb: Some(32_768),
        disk_write_mbps: Some(390.0),
        source_lang: "ja".to_string(),
        target_lang: "en".to_string(),
        quality_profile: "strict".to_string(),
        input_kind: "srt".to_string(),
        audio_duration_secs: Some(600.0),
        speech_duration_secs: Some(540.0),
        estimated_cues: 180,
        avg_difficulty: Some(0.45),
        speaker_complexity: Some(0.25),
        energy_variance: Some(0.20),
        scene_count: Some(12),
        plan: plan_cpu.clone(),
        success: true,
        elapsed_secs: 55.0,
        output_cues: Some(180),
        output_non_empty_ratio: Some(0.98),
        output_top_line_ratio: Some(0.05),
        error: None,
    });

    let hardware = HardwareProbe {
        cpu_cores: 8,
        total_ram_mb: Some(32_768),
        disk_write_mbps: Some(410.0),
        gpu: Some(GpuProbe {
            backend: "cuda".to_string(),
            name: "RTX 3070".to_string(),
            vram_mb: Some(8_192),
            compute_capability: Some("8.6".to_string()),
        }),
    };
    let workload = WorkloadEstimate {
        input_kind: "srt".to_string(),
        is_video: false,
        duration_secs: Some(650.0),
        speech_duration_secs: Some(590.0),
        estimated_cues: 190,
        avg_difficulty: Some(0.48),
        speaker_complexity: Some(0.30),
        energy_variance: Some(0.22),
        scene_count: Some(14),
        detected_source_lang: "ja".to_string(),
        detected_confidence: 0.9,
        detection_mode: "declared".to_string(),
    };

    let selected = history.best_plan_similar(
        &hardware,
        &workload,
        PlanLookupQuery {
            source_lang: "ja",
            target_lang: "en",
            profile: QualityProfile::Strict,
            input_kind: "srt",
            content_profile_hash: "content-new",
        },
    );
    assert_eq!(
        selected.expect("similar fallback should select plan"),
        plan_gpu
    );
}

#[test]
fn parse_ffprobe_output_reads_duration_and_format() {
    let parsed = parse_ffprobe_output("format_name=matroska,webm\nduration=6262.123\n")
        .expect("ffprobe output should parse");
    assert_eq!(parsed.format_name.as_deref(), Some("matroska,webm"));
    let duration = parsed.duration_secs.expect("duration should exist");
    assert!((duration - 6262.123).abs() < 0.001);
}

#[test]
fn duration_to_chunk_secs_scales_with_length() {
    assert_eq!(duration_to_chunk_secs(Some(900.0), None), 180.0);
    assert_eq!(duration_to_chunk_secs(Some(2_500.0), None), 240.0);
    assert_eq!(duration_to_chunk_secs(Some(4_000.0), None), 300.0);
    assert_eq!(duration_to_chunk_secs(Some(9_000.0), None), 360.0);
    assert_eq!(duration_to_chunk_secs(None, None), 300.0);
    assert_eq!(duration_to_chunk_secs(Some(4_000.0), Some(0.8)), 225.0);
    assert_eq!(duration_to_chunk_secs(Some(4_000.0), Some(0.2)), 345.0);
}

#[test]
fn output_health_flags_repetition() {
    let dir = temp_case_dir("output_health");
    let srt = dir.join("sample.en.srt");
    let mut body = String::new();
    for idx in 1..=10usize {
        body.push_str(&format!(
            "{idx}\n00:00:{:02},000 --> 00:00:{:02},500\nrepeat line\n\n",
            idx,
            idx + 1
        ));
    }
    fs::write(&srt, body).expect("test SRT should be writable");
    let health = assess_output_health(&srt).expect("health should parse");
    assert_eq!(health.cue_count, 10);
    assert!(health.non_empty_ratio >= 0.99);
    assert!(health.top_line_ratio >= 0.90);
}

#[test]
fn strict_defaults_are_stable() {
    assert_eq!(default_mt_batch_for_profile(QualityProfile::Strict), 16);
    assert_eq!(
        default_mt_max_tokens_for_profile(QualityProfile::Strict),
        4096
    );
}

#[test]
fn knowledge_snapshot_aggregates_device_and_language_stats() {
    let plan = ExecutionPlan {
        parallel: true,
        workers: 6,
        chunk_duration_secs: 240.0,
        mt_batch_size: Some(12),
        mt_max_batch_tokens: Some(3072),
        mt_oom_retries: Some(3),
        mt_allow_cpu_fallback: true,
    };
    let mut history = HistoryStore::default();
    history.push_record(RunRecord {
        timestamp_epoch_secs: 10,
        device_fingerprint: "gpu-a".to_string(),
        content_profile_hash: "hash-1".to_string(),
        gpu_backend: Some("cuda".to_string()),
        gpu_vram_mb: Some(12_288),
        cpu_cores: Some(8),
        total_ram_mb: Some(32_768),
        disk_write_mbps: Some(450.0),
        source_lang: "ja".to_string(),
        target_lang: "en".to_string(),
        quality_profile: "strict".to_string(),
        input_kind: "matroska,webm".to_string(),
        audio_duration_secs: Some(600.0),
        speech_duration_secs: Some(500.0),
        estimated_cues: 100,
        avg_difficulty: Some(0.4),
        speaker_complexity: Some(0.2),
        energy_variance: Some(0.2),
        scene_count: Some(6),
        plan: plan.clone(),
        success: true,
        elapsed_secs: 40.0,
        output_cues: Some(100),
        output_non_empty_ratio: Some(0.98),
        output_top_line_ratio: Some(0.05),
        error: None,
    });
    history.push_record(RunRecord {
        timestamp_epoch_secs: 11,
        device_fingerprint: "gpu-a".to_string(),
        content_profile_hash: "hash-1".to_string(),
        gpu_backend: Some("cuda".to_string()),
        gpu_vram_mb: Some(12_288),
        cpu_cores: Some(8),
        total_ram_mb: Some(32_768),
        disk_write_mbps: Some(450.0),
        source_lang: "ja".to_string(),
        target_lang: "en".to_string(),
        quality_profile: "strict".to_string(),
        input_kind: "matroska,webm".to_string(),
        audio_duration_secs: Some(1200.0),
        speech_duration_secs: Some(1000.0),
        estimated_cues: 180,
        avg_difficulty: Some(0.5),
        speaker_complexity: Some(0.3),
        energy_variance: Some(0.3),
        scene_count: Some(10),
        plan: plan.clone(),
        success: false,
        elapsed_secs: 85.0,
        output_cues: None,
        output_non_empty_ratio: None,
        output_top_line_ratio: None,
        error: Some("oom".to_string()),
    });

    let snapshot = build_knowledge_snapshot(&history);
    let device = snapshot
        .devices
        .get("cpu8-ram32768-gpucuda-vram12288")
        .expect("device knowledge should exist");
    assert_eq!(device.runs, 2);
    assert!(device.success_rate > 0.4 && device.success_rate < 0.6);

    let pair = snapshot
        .language_pairs
        .get("ja->en")
        .expect("pair knowledge should exist");
    assert_eq!(pair.runs, 2);
    assert!(pair.avg_elapsed_per_audio_hour_secs.is_some());
}
