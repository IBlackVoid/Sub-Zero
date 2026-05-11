mod engine;

use engine::doom_qlock::DoomQlock;
use engine::http_sidecar::{start_http_sidecar, HttpSidecarConfig};
use engine::pipeline::{PipelineConfig, SubtitlePipeline};
use engine::transcribe::QualityProfile;
use engine::ws_sidecar::{start_ws_sidecar, WsSidecarConfig};
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct Args {
    inputs: Vec<PathBuf>,
    source_lang: String,
    target_lang: String,
    offline: bool,
    transcribe: bool,
    whisper_bin: Option<PathBuf>,
    whisper_model: Option<PathBuf>,
    whisper_args: Vec<String>,
    audio_lang: Option<String>,
    audio_stream_index: Option<usize>,
    skip_existing: bool,
    vad: bool,
    vad_threshold_db: f64,
    vad_min_silence: f64,
    vad_pad: f64,
    verify: bool,
    verify_min_speech_overlap: f64,
    gpu: bool,
    require_gpu: bool,

    parallel: bool,
    stream: bool,
    stream_async: bool,
    max_workers: usize,
    chunk_duration_secs: f64,
    force_phrase_table: bool,
    speaker_aware: bool,
    speaker_diarize: bool,
    speaker_max_speakers: usize,
    mt_model: Option<String>,
    mt_batch_size: Option<usize>,
    mt_max_batch_tokens: Option<usize>,
    mt_oom_retries: Option<usize>,
    mt_allow_cpu_fallback: bool,
    mt_force_cpu: bool,
    mt_daemon: bool,
    mt_enforce_quality_floor: bool,
    auto_repair_sidecar: bool,
    doom_qlock: bool,
    trace_runtime: bool,
    events_json: bool,
    events_file: Option<PathBuf>,
    http_events_addr: Option<String>,
    ws_events_addr: Option<String>,
    quality_profile: QualityProfile,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if is_help_requested(&raw_args) {
        println!("{}", help_text());
        return Ok(());
    }

    let args = parse_args(raw_args)?;

    let http_events = if let Some(addr) = args.http_events_addr.as_deref() {
        let handle = start_http_sidecar(HttpSidecarConfig {
            bind_addr: addr.to_string(),
        })?;
        Some(handle.events_tx)
    } else {
        None
    };
    let ws_events = if let Some(addr) = args.ws_events_addr.as_deref() {
        let handle = start_ws_sidecar(WsSidecarConfig {
            bind_addr: addr.to_string(),
        })?;
        Some(handle.events_tx)
    } else {
        None
    };

    let base_config = PipelineConfig {
        source_lang: args.source_lang,
        target_lang: args.target_lang,
        offline: args.offline,
        transcribe: args.transcribe,
        whisper_bin: args.whisper_bin,
        whisper_model: args.whisper_model,
        whisper_args: args.whisper_args,
        audio_lang: args.audio_lang,
        audio_stream_index: args.audio_stream_index,
        skip_existing: args.skip_existing,
        vad: args.vad,
        vad_threshold_db: args.vad_threshold_db,
        vad_min_silence: args.vad_min_silence,
        vad_pad: args.vad_pad,
        verify: args.verify,
        verify_min_speech_overlap: args.verify_min_speech_overlap,
        gpu: args.gpu,
        require_gpu: args.require_gpu,
        parallel: args.parallel,
        stream: args.stream,
        stream_async: args.stream_async,
        max_workers: args.max_workers,
        chunk_duration_secs: args.chunk_duration_secs,
        force_phrase_table: args.force_phrase_table,
        speaker_aware: args.speaker_aware,
        speaker_diarize: args.speaker_diarize,
        speaker_max_speakers: args.speaker_max_speakers,
        mt_model: args.mt_model,
        mt_batch_size: args.mt_batch_size,
        mt_max_batch_tokens: args.mt_max_batch_tokens,
        mt_oom_retries: args.mt_oom_retries,
        mt_allow_cpu_fallback: args.mt_allow_cpu_fallback,
        mt_force_cpu: args.mt_force_cpu,
        mt_daemon: args.mt_daemon,
        mt_enforce_quality_floor: args.mt_enforce_quality_floor,
        auto_repair_sidecar: args.auto_repair_sidecar,
        trace_runtime: args.trace_runtime,
        events_json: args.events_json,
        events_file: args.events_file,
        http_events,
        ws_events,
        quality_profile: args.quality_profile,
    };

    let mut doom_qlock = if args.doom_qlock {
        Some(DoomQlock::load_default())
    } else {
        None
    };

    for input in &args.inputs {
        let prepared = if let Some(qlock) = doom_qlock.as_ref() {
            Some(
                qlock
                    .prepare_run(input, &base_config)
                    .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };

        let effective_config = prepared
            .as_ref()
            .map(|run| run.effective_config.clone())
            .unwrap_or_else(|| base_config.clone());

        let emit_events_json = effective_config.events_json;
        if emit_events_json {
            let payload = serde_json::json!({
                "event": "input_start",
                "input": input.display().to_string(),
                "source_lang": &effective_config.source_lang,
                "target_lang": &effective_config.target_lang,
                "quality_profile": effective_config.quality_profile.as_str(),
            });
            if let Ok(line) = serde_json::to_string(&payload) {
                println!("{line}");
            }
        }

        let pipeline =
            SubtitlePipeline::new(effective_config).map_err(|error| error.to_string())?;
        let start = std::time::Instant::now();
        let output = match pipeline.process_input(input) {
            Ok(output) => output,
            Err(error) => {
                if let (Some(qlock), Some(prepared)) = (doom_qlock.as_mut(), prepared.as_ref()) {
                    let elapsed_secs = start.elapsed().as_secs_f64();
                    qlock.record_failure(prepared, elapsed_secs, &error.to_string());
                }
                return Err(error.to_string());
            }
        };

        let elapsed = start.elapsed().as_secs_f64();
        if let (Some(qlock), Some(prepared)) = (doom_qlock.as_mut(), prepared.as_ref()) {
            qlock.record_success(prepared, &output, elapsed);
        }
        if emit_events_json {
            let payload = serde_json::json!({
                "event": "input_complete",
                "input": input.display().to_string(),
                "output": output.display().to_string(),
                "elapsed_secs": elapsed,
            });
            if let Ok(line) = serde_json::to_string(&payload) {
                println!("{line}");
            }
        } else {
            println!(
                "translated: {} -> {} ({:.1}s)",
                input.display(),
                output.display(),
                elapsed
            );
        }
    }
    Ok(())
}

fn is_help_requested(raw: &[String]) -> bool {
    raw.iter().any(|arg| arg == "-h" || arg == "--help")
}

fn parse_args(raw: Vec<String>) -> Result<Args, String> {
    if raw.is_empty() {
        return Err(help_text());
    }

    let mut inputs = Vec::<PathBuf>::new();
    let mut source_lang = String::from("ja");
    let mut target_lang = String::from("en");
    let mut offline = false;

    let mut transcribe = true;
    let mut whisper_bin = Option::<PathBuf>::None;
    let mut whisper_model = Option::<PathBuf>::None;
    let mut whisper_args = Vec::<String>::new();
    let mut audio_lang = Option::<String>::None;
    let mut audio_stream_index = Option::<usize>::None;
    let mut skip_existing = false;
    let mut vad = true;
    let mut vad_threshold_db = -35.0f64;
    let mut vad_min_silence = 0.35f64;
    let mut vad_pad = 0.20f64;
    let mut verify = false;
    let mut verify_min_speech_overlap = 0.35f64;
    let mut gpu = false;
    let mut require_gpu = false;

    let mut parallel = false;
    let mut stream = false;
    let mut stream_async = false;
    let mut max_workers = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4);
    let mut chunk_duration_secs = 300.0f64;
    let mut force_phrase_table = false;
    let mut speaker_aware = false;
    let mut speaker_diarize = false;
    let mut speaker_max_speakers = 4usize;
    let mut mt_model = Option::<String>::None;
    let mut mt_batch_size = Option::<usize>::None;
    let mut mt_max_batch_tokens = Option::<usize>::None;
    let mut mt_oom_retries = Option::<usize>::None;
    let mut mt_allow_cpu_fallback = true;
    let mut mt_force_cpu = false;
    let mut mt_daemon = false;
    let mut mt_enforce_quality_floor = true;
    let mut auto_repair_sidecar = true;
    let mut doom_qlock = true;
    let mut trace_runtime = false;
    let mut events_json = false;
    let mut events_file = Option::<PathBuf>::None;
    let mut http_events_addr = Option::<String>::None;
    let mut ws_events_addr = Option::<String>::None;
    let mut quality_profile = QualityProfile::Balanced;

    let mut index = 0usize;
    while index < raw.len() {
        match raw[index].as_str() {
            "--offline" => {
                offline = true;
                index += 1;
            }
            "--transcribe" => {
                transcribe = true;
                index += 1;
            }
            "--no-transcribe" => {
                transcribe = false;
                index += 1;
            }
            "--skip-existing" => {
                skip_existing = true;
                index += 1;
            }
            "--vad" => {
                vad = true;
                index += 1;
            }
            "--no-vad" => {
                vad = false;
                index += 1;
            }
            "--verify" => {
                verify = true;
                index += 1;
            }
            "--parallel" => {
                parallel = true;
                index += 1;
            }
            "--stream" => {
                stream = true;
                index += 1;
            }
            "--stream-async" => {
                stream = true;
                stream_async = true;
                index += 1;
            }
            "--phrase-table" => {
                force_phrase_table = true;
                index += 1;
            }
            "--speaker-aware" => {
                speaker_aware = true;
                index += 1;
            }
            "--no-speaker-aware" => {
                speaker_aware = false;
                index += 1;
            }
            "--speaker-diarize" => {
                speaker_aware = true;
                speaker_diarize = true;
                index += 1;
            }
            "--no-speaker-diarize" => {
                speaker_diarize = false;
                index += 1;
            }
            "--auto-repair-sidecar" => {
                auto_repair_sidecar = true;
                index += 1;
            }
            "--no-auto-repair-sidecar" => {
                auto_repair_sidecar = false;
                index += 1;
            }
            "--doom-qlock" => {
                doom_qlock = true;
                index += 1;
            }
            "--no-doom-qlock" => {
                doom_qlock = false;
                index += 1;
            }
            "--trace-runtime" => {
                trace_runtime = true;
                index += 1;
            }
            "--events-json" => {
                events_json = true;
                index += 1;
            }
            "--http-events" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --http-events".to_string());
                };
                if value.trim().is_empty() {
                    return Err("--http-events cannot be empty".to_string());
                }
                http_events_addr = Some(value.clone());
                // Sidecar needs the JSON event stream.
                events_json = true;
                index += 1;
            }
            "--ws-events" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --ws-events".to_string());
                };
                if value.trim().is_empty() {
                    return Err("--ws-events cannot be empty".to_string());
                }
                ws_events_addr = Some(value.clone());
                events_json = true;
                index += 1;
            }
            "--events-file" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --events-file".to_string());
                };
                if value.trim().is_empty() {
                    return Err("--events-file cannot be empty".to_string());
                }
                events_file = Some(PathBuf::from(value));
                index += 1;
            }
            "--profile" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --profile".to_string());
                };
                quality_profile =
                    QualityProfile::parse(value).map_err(|error| error.to_string())?;
                index += 1;
            }
            "--speaker-max-speakers" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --speaker-max-speakers".to_string());
                };
                speaker_max_speakers = value
                    .parse::<usize>()
                    .map_err(|_| "invalid integer for --speaker-max-speakers".to_string())?;
                index += 1;
            }
            "--vad-threshold-db" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --vad-threshold-db".to_string());
                };
                vad_threshold_db = value
                    .parse::<f64>()
                    .map_err(|_| "invalid float for --vad-threshold-db".to_string())?;
                index += 1;
            }
            "--vad-min-silence" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --vad-min-silence".to_string());
                };
                vad_min_silence = value
                    .parse::<f64>()
                    .map_err(|_| "invalid float for --vad-min-silence".to_string())?;
                index += 1;
            }
            "--vad-pad" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --vad-pad".to_string());
                };
                vad_pad = value
                    .parse::<f64>()
                    .map_err(|_| "invalid float for --vad-pad".to_string())?;
                index += 1;
            }
            "--verify-min-speech-overlap" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --verify-min-speech-overlap".to_string());
                };
                verify_min_speech_overlap = value
                    .parse::<f64>()
                    .map_err(|_| "invalid float for --verify-min-speech-overlap".to_string())?;
                index += 1;
            }
            "--gpu" => {
                gpu = true;
                index += 1;
            }
            "--require-gpu" => {
                require_gpu = true;
                gpu = true;
                index += 1;
            }
            "--whisper-bin" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --whisper-bin".to_string());
                };
                whisper_bin = Some(PathBuf::from(value));
                index += 1;
            }
            "--whisper-model" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --whisper-model".to_string());
                };
                whisper_model = Some(PathBuf::from(value));
                index += 1;
            }
            "--whisper-arg" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --whisper-arg".to_string());
                };
                whisper_args.push(value.clone());
                index += 1;
            }
            "--audio-lang" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --audio-lang".to_string());
                };
                if value.trim().is_empty() {
                    return Err("--audio-lang cannot be empty".to_string());
                }
                audio_lang = Some(value.clone());
                index += 1;
            }
            "--audio-stream" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --audio-stream".to_string());
                };
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| "invalid integer for --audio-stream".to_string())?;
                audio_stream_index = Some(parsed);
                index += 1;
            }
            "--workers" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --workers".to_string());
                };
                max_workers = value
                    .parse::<usize>()
                    .map_err(|_| "invalid integer for --workers".to_string())?;
                if max_workers == 0 {
                    return Err("--workers must be at least 1".to_string());
                }
                index += 1;
            }
            "--chunk-duration" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --chunk-duration".to_string());
                };
                chunk_duration_secs = value
                    .parse::<f64>()
                    .map_err(|_| "invalid float for --chunk-duration".to_string())?;
                if chunk_duration_secs < 30.0 {
                    return Err("--chunk-duration must be at least 30 seconds".to_string());
                }
                index += 1;
            }
            "--mt-model" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --mt-model".to_string());
                };
                mt_model = Some(value.clone());
                index += 1;
            }
            "--mt-batch-size" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --mt-batch-size".to_string());
                };
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| "invalid integer for --mt-batch-size".to_string())?;
                if parsed == 0 {
                    return Err("--mt-batch-size must be at least 1".to_string());
                }
                mt_batch_size = Some(parsed);
                index += 1;
            }
            "--mt-max-batch-tokens" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --mt-max-batch-tokens".to_string());
                };
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| "invalid integer for --mt-max-batch-tokens".to_string())?;
                if parsed == 0 {
                    return Err("--mt-max-batch-tokens must be at least 1".to_string());
                }
                mt_max_batch_tokens = Some(parsed);
                index += 1;
            }
            "--mt-oom-retries" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --mt-oom-retries".to_string());
                };
                mt_oom_retries = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| "invalid integer for --mt-oom-retries".to_string())?,
                );
                index += 1;
            }
            "--mt-allow-cpu-fallback" => {
                mt_allow_cpu_fallback = true;
                index += 1;
            }
            "--mt-no-cpu-fallback" => {
                mt_allow_cpu_fallback = false;
                index += 1;
            }
            "--mt-force-cpu" => {
                mt_force_cpu = true;
                index += 1;
            }
            "--mt-daemon" => {
                mt_daemon = true;
                index += 1;
            }
            "--no-mt-daemon" => {
                mt_daemon = false;
                index += 1;
            }
            "--mt-no-quality-floor" => {
                mt_enforce_quality_floor = false;
                index += 1;
            }
            "-l" | "--lang" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --lang".to_string());
                };
                target_lang = value.clone();
                index += 1;
            }
            "--source-lang" => {
                index += 1;
                let Some(value) = raw.get(index) else {
                    return Err("missing value for --source-lang".to_string());
                };
                source_lang = value.clone();
                index += 1;
            }
            "-i" | "--input" => {
                index += 1;
                while index < raw.len() && !raw[index].starts_with('-') {
                    inputs.push(PathBuf::from(&raw[index]));
                    index += 1;
                }
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown flag: {flag}"));
            }
            value => {
                inputs.push(PathBuf::from(value));
                index += 1;
            }
        }
    }

    if inputs.is_empty() {
        return Err("no input files were provided.\n\n".to_string() + &help_text());
    }
    if source_lang.trim().is_empty() {
        return Err("--source-lang cannot be empty".to_string());
    }
    if target_lang.trim().is_empty() {
        return Err("--lang cannot be empty".to_string());
    }
    if !vad_threshold_db.is_finite() {
        return Err("--vad-threshold-db must be a finite number".to_string());
    }
    if !vad_min_silence.is_finite() || vad_min_silence <= 0.0 {
        return Err("--vad-min-silence must be > 0".to_string());
    }
    if !vad_pad.is_finite() || vad_pad < 0.0 {
        return Err("--vad-pad must be >= 0".to_string());
    }
    if !verify_min_speech_overlap.is_finite() || !(0.0..=1.0).contains(&verify_min_speech_overlap) {
        return Err("--verify-min-speech-overlap must be between 0 and 1".to_string());
    }
    if events_file.is_some() && !events_json {
        return Err("--events-file requires --events-json".to_string());
    }
    if !(2..=16).contains(&speaker_max_speakers) {
        return Err("--speaker-max-speakers must be between 2 and 16".to_string());
    }

    Ok(Args {
        inputs,
        source_lang,
        target_lang,
        offline,
        transcribe,
        whisper_bin,
        whisper_model,
        whisper_args,
        audio_lang,
        audio_stream_index,
        skip_existing,
        vad,
        vad_threshold_db,
        vad_min_silence,
        vad_pad,
        verify,
        verify_min_speech_overlap,
        gpu,
        require_gpu,
        parallel,
        stream,
        stream_async,
        max_workers,
        chunk_duration_secs,
        force_phrase_table,
        speaker_aware,
        speaker_diarize,
        speaker_max_speakers,
        mt_model,
        mt_batch_size,
        mt_max_batch_tokens,
        mt_oom_retries,
        mt_allow_cpu_fallback,
        mt_force_cpu,
        mt_daemon,
        mt_enforce_quality_floor,
        auto_repair_sidecar,
        doom_qlock,
        trace_runtime,
        events_json,
        events_file,
        http_events_addr,
        ws_events_addr,
        quality_profile,
    })
}

fn help_text() -> String {
    [
        "Sub-Zero - Offline Subtitle Translator",
        "",
        "Usage:",
        "  sub-zero -i <file> [<file> ...] [options]",
        "",
        "Core Options:",
        "  --source-lang <code>     Source language (default: ja)",
        "  -l, --lang <code>        Target language (default: en)",
        "  --profile <name>         Quality profile: fast|balanced|strict (default: balanced)",
        "  --offline                Use offline-only backends",
        "  --skip-existing          Skip if output already exists",
        "",
        "Transcription:",
        "  --transcribe             Transcribe audio to SRT (default: on for video inputs)",
        "  --no-transcribe          Use existing sidecar SRT for video inputs when available",
        "  --audio-lang <code>      Prefer audio track language when extracting (example: jpn, eng)",
        "  --audio-stream <index>   Force ffmpeg stream index for audio extraction (overrides --audio-lang)",
        "  --whisper-bin <path>     Path to whisper.cpp binary",
        "  --whisper-model <path>   Path to whisper GGML model",
        "  --whisper-arg <x>        Pass-through arg to whisper backend",
        "",
        "Speed Pipeline:",
        "  --parallel               Enable parallel chunked transcription",
        "  --stream                 Stream chunk-by-chunk transcription+translation (writes output SRT progressively)",
        "  --stream-async           Streaming mode with bounded channels (overlap transcribe + translate)",
        "  --workers <N>            Max parallel whisper workers (default: CPU count)",
        "  --chunk-duration <secs>  Target chunk size in seconds (default: 300)",
        "",
        "Accuracy Pipeline:",
        "  --mt-model <name>        Neural MT model (default: nllb-200-distilled-600M)",
        "  --mt-batch-size <N>      MT batch size override (profile default when omitted)",
        "  --mt-max-batch-tokens <N> MT max token batch size override",
        "  --mt-oom-retries <N>     MT CUDA OOM retries before final failure/fallback",
        "  --mt-allow-cpu-fallback  Allow CPU fallback on MT CUDA OOM (default: on)",
        "  --mt-no-cpu-fallback     Disable MT CPU fallback on CUDA OOM",
        "  --mt-force-cpu           Force MT to run on CPU even when --gpu is set (frees VRAM for ASR)",
        "  --mt-daemon              Reuse a persistent MT Python daemon (default: off)",
        "  --no-mt-daemon           Disable MT daemon reuse",
        "  --mt-no-quality-floor    Do not fail when MT heuristic quality is below the profile floor (best-effort)",
        "  --phrase-table           Force phrase-table fallback (skip neural MT)",
        "  --speaker-aware          Enable speaker-aware discourse translation (Phase E)",
        "  --no-speaker-aware       Disable speaker-aware translation (default)",
        "  --speaker-diarize        Enable audio-based speaker diarization (fills missing speaker tags; implies --speaker-aware)",
        "  --no-speaker-diarize     Disable audio diarization (default)",
        "  --speaker-max-speakers <n>  Upper bound for diarization clusters (default: 4)",
        "  --doom-qlock             Enable adaptive IBVoid DOOM-QLOCK runtime policy (default: on)",
        "  --no-doom-qlock          Disable IBVoid DOOM-QLOCK and use fixed CLI settings",
        "  --trace-runtime         Emit <input>.sub-zero.trace.json with per-stage timings",
        "  --events-json           Emit JSON-lines event stream (for streaming/integration)",
        "  --events-file <path>    Append JSON-lines events to a file (requires --events-json)",
        "  --http-events <addr>    Start local HTTP sidecar (SSE) at addr; enables --events-json",
        "  --ws-events <addr>      Start local WebSocket sidecar at addr (path /ws); enables --events-json",
        "  --no-auto-repair-sidecar  Disable automatic rescue transcription when sidecar SRT looks degraded",
        "",
        "Audio/VAD:",
        "  --vad                    Enable VAD silence skipping (default: on)",
        "  --no-vad                 Disable VAD silence skipping",
        "  --vad-threshold-db <dB>  Silence threshold (default: -35)",
        "  --vad-min-silence <sec>  Minimum silence duration (default: 0.35)",
        "  --vad-pad <sec>          Padding around speech (default: 0.20)",
        "",
        "GPU:",
        "  --gpu                    Request GPU acceleration (fallback to CPU)",
        "  --require-gpu            Fail if CUDA is unavailable",
        "",
        "Verification:",
        "  --verify                 Check subtitle timing vs detected speech",
        "  --verify-min-speech-overlap <ratio>  Minimum overlap ratio (default: 0.35)",
        "",
        "Output: <input_stem>.<target_lang>.srt",
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::{is_help_requested, parse_args};
    use std::path::PathBuf;

    #[test]
    fn parse_positional_inputs() {
        let args = parse_args(vec![
            "a.srt".to_string(),
            "b.mkv".to_string(),
            "--source-lang".to_string(),
            "ja".to_string(),
            "--lang".to_string(),
            "en".to_string(),
            "--offline".to_string(),
            "--skip-existing".to_string(),
            "--vad".to_string(),
            "--verify".to_string(),
            "--vad-threshold-db".to_string(),
            "-32".to_string(),
            "--vad-min-silence".to_string(),
            "0.5".to_string(),
            "--vad-pad".to_string(),
            "0.25".to_string(),
            "--verify-min-speech-overlap".to_string(),
            "0.4".to_string(),
            "--require-gpu".to_string(),
            "--transcribe".to_string(),
            "--audio-lang".to_string(),
            "jpn".to_string(),
            "--whisper-bin".to_string(),
            "/tmp/whisper-cli".to_string(),
            "--whisper-model".to_string(),
            "/tmp/ggml-model.bin".to_string(),
            "--whisper-arg".to_string(),
            "-t".to_string(),
            "--whisper-arg".to_string(),
            "8".to_string(),
        ])
        .expect("parse should succeed");

        assert_eq!(args.inputs.len(), 2);
        assert_eq!(args.source_lang, "ja");
        assert_eq!(args.target_lang, "en");
        assert!(args.offline);
        assert!(args.transcribe);
        assert_eq!(args.audio_lang.as_deref(), Some("jpn"));
        assert_eq!(args.whisper_bin.unwrap(), PathBuf::from("/tmp/whisper-cli"));
        assert_eq!(
            args.whisper_model.unwrap(),
            PathBuf::from("/tmp/ggml-model.bin")
        );
        assert_eq!(args.whisper_args, vec!["-t".to_string(), "8".to_string()]);
        assert!(args.skip_existing);
        assert!(args.vad);
        assert!(args.verify);
        assert_eq!(args.vad_threshold_db, -32.0);
        assert_eq!(args.vad_min_silence, 0.5);
        assert_eq!(args.vad_pad, 0.25);
        assert_eq!(args.verify_min_speech_overlap, 0.4);
        assert!(args.gpu);
        assert!(args.require_gpu);
    }

    #[test]
    fn parse_audio_stream_flag() {
        let args = parse_args(vec![
            "video.mkv".to_string(),
            "--audio-stream".to_string(),
            "2".to_string(),
        ])
        .expect("parse should succeed");
        assert_eq!(args.audio_stream_index, Some(2));
    }

    #[test]
    fn parse_pipeline_flags() {
        let args = parse_args(vec![
            "video.mkv".to_string(),
            "--parallel".to_string(),
            "--workers".to_string(),
            "8".to_string(),
            "--chunk-duration".to_string(),
            "180".to_string(),
            "--mt-model".to_string(),
            "nllb-200-distilled-1.3B".to_string(),
            "--phrase-table".to_string(),
        ])
        .expect("parse should succeed");

        assert!(args.parallel);
        assert_eq!(args.max_workers, 8);
        assert_eq!(args.chunk_duration_secs, 180.0);
        assert_eq!(args.mt_model.unwrap(), "nllb-200-distilled-1.3B");
        assert!(args.force_phrase_table);
        assert_eq!(args.quality_profile, super::QualityProfile::Balanced);
    }

    #[test]
    fn parse_strict_profile_flag() {
        let args = parse_args(vec![
            "video.mkv".to_string(),
            "--profile".to_string(),
            "strict".to_string(),
        ])
        .expect("parse should succeed");
        assert_eq!(args.quality_profile, super::QualityProfile::Strict);
    }

    #[test]
    fn parse_mt_runtime_flags() {
        let args = parse_args(vec![
            "video.mkv".to_string(),
            "--mt-batch-size".to_string(),
            "12".to_string(),
            "--mt-max-batch-tokens".to_string(),
            "4096".to_string(),
            "--mt-oom-retries".to_string(),
            "4".to_string(),
            "--mt-no-cpu-fallback".to_string(),
            "--mt-force-cpu".to_string(),
            "--mt-daemon".to_string(),
            "--mt-no-quality-floor".to_string(),
        ])
        .expect("parse should succeed");

        assert_eq!(args.mt_batch_size, Some(12));
        assert_eq!(args.mt_max_batch_tokens, Some(4096));
        assert_eq!(args.mt_oom_retries, Some(4));
        assert!(!args.mt_allow_cpu_fallback);
        assert!(args.mt_force_cpu);
        assert!(args.mt_daemon);
        assert!(!args.mt_enforce_quality_floor);
    }

    #[test]
    fn detect_help_flags() {
        assert!(is_help_requested(&["--help".to_string()]));
        assert!(is_help_requested(&["-h".to_string()]));
        assert!(!is_help_requested(&["--offline".to_string()]));
    }

    #[test]
    fn parse_no_vad_flag() {
        let args = parse_args(vec!["video.mkv".to_string(), "--no-vad".to_string()])
            .expect("parse should succeed");
        assert!(!args.vad);
    }

    #[test]
    fn parse_no_transcribe_flag() {
        let args = parse_args(vec!["video.mkv".to_string(), "--no-transcribe".to_string()])
            .expect("parse should succeed");
        assert!(!args.transcribe);
    }

    #[test]
    fn parse_no_auto_repair_sidecar_flag() {
        let args = parse_args(vec![
            "video.mkv".to_string(),
            "--no-auto-repair-sidecar".to_string(),
        ])
        .expect("parse should succeed");
        assert!(!args.auto_repair_sidecar);
    }

    #[test]
    fn parse_no_doom_qlock_flag() {
        let args = parse_args(vec!["video.mkv".to_string(), "--no-doom-qlock".to_string()])
            .expect("parse should succeed");
        assert!(!args.doom_qlock);
    }

    #[test]
    fn parse_trace_runtime_flag() {
        let args = parse_args(vec!["video.mkv".to_string(), "--trace-runtime".to_string()])
            .expect("parse should succeed");
        assert!(args.trace_runtime);
    }

    #[test]
    fn parse_stream_flag() {
        let args = parse_args(vec!["video.mkv".to_string(), "--stream".to_string()])
            .expect("parse should succeed");
        assert!(args.stream);
    }

    #[test]
    fn parse_stream_async_flag_implies_stream() {
        let args = parse_args(vec!["video.mkv".to_string(), "--stream-async".to_string()])
            .expect("parse should succeed");
        assert!(args.stream);
        assert!(args.stream_async);
    }

    #[test]
    fn parse_speaker_aware_flag() {
        let args = parse_args(vec!["video.mkv".to_string(), "--speaker-aware".to_string()])
            .expect("parse should succeed");
        assert!(args.speaker_aware);
    }

    #[test]
    fn parse_speaker_diarize_flag_enables_speaker_aware() {
        let args = parse_args(vec![
            "video.mkv".to_string(),
            "--speaker-diarize".to_string(),
            "--speaker-max-speakers".to_string(),
            "5".to_string(),
        ])
        .expect("parse should succeed");
        assert!(args.speaker_aware);
        assert!(args.speaker_diarize);
        assert_eq!(args.speaker_max_speakers, 5);
    }

    #[test]
    fn parse_events_json_flag() {
        let args = parse_args(vec!["video.mkv".to_string(), "--events-json".to_string()])
            .expect("parse should succeed");
        assert!(args.events_json);
    }

    #[test]
    fn parse_events_file_flag() {
        let args = parse_args(vec![
            "video.mkv".to_string(),
            "--events-json".to_string(),
            "--events-file".to_string(),
            "events.jsonl".to_string(),
        ])
        .expect("parse should succeed");
        assert_eq!(args.events_file, Some(PathBuf::from("events.jsonl")));
    }

    #[test]
    fn parse_http_events_flag_enables_events_json() {
        let args = parse_args(vec![
            "video.mkv".to_string(),
            "--http-events".to_string(),
            "127.0.0.1:9411".to_string(),
        ])
        .expect("parse should succeed");
        assert!(args.events_json);
        assert_eq!(args.http_events_addr, Some("127.0.0.1:9411".to_string()));
    }

    #[test]
    fn parse_ws_events_flag_enables_events_json() {
        let args = parse_args(vec![
            "video.mkv".to_string(),
            "--ws-events".to_string(),
            "127.0.0.1:9412".to_string(),
        ])
        .expect("parse should succeed");
        assert!(args.events_json);
        assert_eq!(args.ws_events_addr, Some("127.0.0.1:9412".to_string()));
    }
}
