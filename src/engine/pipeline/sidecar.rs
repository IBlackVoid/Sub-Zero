use super::paths::checkpoint_dir_for;
use super::scenes::{scene_floor_for_difficulty, scene_quality, split_scenes};
use crate::engine::chunker::AudioChunk;
use crate::engine::srt::SubtitleCue;
use std::path::Path;

pub(super) fn write_parallel_confidence_sidecar(
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

pub(super) fn write_runtime_trace_sidecar(
    trace_path: &Path,
    payload: &serde_json::Value,
) -> Result<(), String> {
    let serialized = serde_json::to_string_pretty(payload)
        .map_err(|e| format!("{} serialize trace: {e}", trace_path.display()))?;
    std::fs::write(trace_path, serialized).map_err(|e| format!("{}: {e}", trace_path.display()))
}

pub(super) fn build_scene_metadata(cues: &[SubtitleCue]) -> Vec<serde_json::Value> {
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

pub(super) fn load_checkpoint_summary(input: &Path) -> Option<serde_json::Value> {
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
                        "chunk": entry.get("chunk_index").cloned().unwrap_or(serde_json::Value::Null),
                        "event": "chunk_failure",
                        "reason": entry.get("reason").cloned().unwrap_or_else(|| serde_json::json!("unknown")),
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
