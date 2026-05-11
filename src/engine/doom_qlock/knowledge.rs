use super::history::{record_backend, record_vram_mb, HistoryStore, RunRecord, HISTORY_VERSION};
use super::plan::ExecutionPlan;
use super::util::{
    bucket_cpu_cores, bucket_ram_mb, bucket_vram_mb, now_epoch_secs, parse_prefixed_u64,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub(super) fn default_knowledge_path(history_path: &Path) -> PathBuf {
    history_path.with_file_name("knowledge.json")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct KnowledgeSnapshot {
    pub(super) version: u32,
    pub(super) generated_at_epoch_secs: u64,
    pub(super) devices: HashMap<String, DeviceKnowledge>,
    pub(super) language_pairs: HashMap<String, LanguagePairKnowledge>,
    pub(super) content_kinds: HashMap<String, ContentKnowledge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct DeviceKnowledge {
    pub(super) runs: usize,
    pub(super) success_rate: f64,
    pub(super) avg_elapsed_secs: f64,
    pub(super) best_elapsed_secs: Option<f64>,
    pub(super) best_plan: Option<ExecutionPlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct LanguagePairKnowledge {
    pub(super) runs: usize,
    pub(super) success_rate: f64,
    pub(super) avg_elapsed_secs: f64,
    pub(super) avg_duration_secs: f64,
    pub(super) avg_elapsed_per_audio_hour_secs: Option<f64>,
    pub(super) best_elapsed_secs: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct ContentKnowledge {
    pub(super) runs: usize,
    pub(super) success_rate: f64,
    pub(super) avg_elapsed_secs: f64,
    pub(super) avg_difficulty: Option<f64>,
    pub(super) avg_scene_count: Option<f64>,
}

#[derive(Debug, Clone)]
struct KnowledgeAccum {
    runs: usize,
    success: usize,
    elapsed_sum: f64,
    best_elapsed: Option<f64>,
    best_plan: Option<ExecutionPlan>,
    duration_sum: f64,
    duration_count: usize,
    difficulty_sum: f64,
    difficulty_count: usize,
    scene_sum: f64,
    scene_count: usize,
}

impl KnowledgeAccum {
    fn observe(&mut self, record: &RunRecord) {
        self.runs += 1;
        if record.success {
            self.success += 1;
        }
        if record.elapsed_secs.is_finite() && record.elapsed_secs >= 0.0 {
            self.elapsed_sum += record.elapsed_secs;
            if record.success {
                let is_better = self
                    .best_elapsed
                    .map(|best| record.elapsed_secs < best)
                    .unwrap_or(true);
                if is_better {
                    self.best_elapsed = Some(record.elapsed_secs);
                    self.best_plan = Some(record.plan.clone());
                }
            }
        }
        if let Some(duration) = record.audio_duration_secs {
            if duration.is_finite() && duration > 0.0 {
                self.duration_sum += duration;
                self.duration_count += 1;
            }
        }
        if let Some(difficulty) = record.avg_difficulty {
            if difficulty.is_finite() {
                self.difficulty_sum += difficulty;
                self.difficulty_count += 1;
            }
        }
        if let Some(scene_count) = record.scene_count {
            self.scene_sum += scene_count as f64;
            self.scene_count += 1;
        }
    }

    fn success_rate(&self) -> f64 {
        if self.runs == 0 {
            0.0
        } else {
            self.success as f64 / self.runs as f64
        }
    }

    fn avg_elapsed_secs(&self) -> f64 {
        if self.runs == 0 {
            0.0
        } else {
            self.elapsed_sum / self.runs as f64
        }
    }
}

pub(super) fn build_knowledge_snapshot(history: &HistoryStore) -> KnowledgeSnapshot {
    let mut device_accum = HashMap::<String, KnowledgeAccum>::new();
    let mut pair_accum = HashMap::<String, KnowledgeAccum>::new();
    let mut content_accum = HashMap::<String, KnowledgeAccum>::new();

    for record in &history.records {
        device_accum
            .entry(device_knowledge_key(record))
            .or_insert_with(|| KnowledgeAccum {
                runs: 0,
                success: 0,
                elapsed_sum: 0.0,
                best_elapsed: None,
                best_plan: None,
                duration_sum: 0.0,
                duration_count: 0,
                difficulty_sum: 0.0,
                difficulty_count: 0,
                scene_sum: 0.0,
                scene_count: 0,
            })
            .observe(record);

        let pair_key = format!(
            "{}->{}",
            record.source_lang.to_ascii_lowercase(),
            record.target_lang.to_ascii_lowercase()
        );
        pair_accum
            .entry(pair_key)
            .or_insert_with(|| KnowledgeAccum {
                runs: 0,
                success: 0,
                elapsed_sum: 0.0,
                best_elapsed: None,
                best_plan: None,
                duration_sum: 0.0,
                duration_count: 0,
                difficulty_sum: 0.0,
                difficulty_count: 0,
                scene_sum: 0.0,
                scene_count: 0,
            })
            .observe(record);

        content_accum
            .entry(record.input_kind.clone())
            .or_insert_with(|| KnowledgeAccum {
                runs: 0,
                success: 0,
                elapsed_sum: 0.0,
                best_elapsed: None,
                best_plan: None,
                duration_sum: 0.0,
                duration_count: 0,
                difficulty_sum: 0.0,
                difficulty_count: 0,
                scene_sum: 0.0,
                scene_count: 0,
            })
            .observe(record);
    }

    let mut devices = HashMap::<String, DeviceKnowledge>::new();
    for (key, accum) in device_accum {
        devices.insert(
            key,
            DeviceKnowledge {
                runs: accum.runs,
                success_rate: accum.success_rate(),
                avg_elapsed_secs: accum.avg_elapsed_secs(),
                best_elapsed_secs: accum.best_elapsed,
                best_plan: accum.best_plan,
            },
        );
    }

    let mut language_pairs = HashMap::<String, LanguagePairKnowledge>::new();
    for (key, accum) in pair_accum {
        let avg_duration_secs = if accum.duration_count == 0 {
            0.0
        } else {
            accum.duration_sum / accum.duration_count as f64
        };
        let avg_elapsed_per_audio_hour_secs =
            if accum.duration_count == 0 || avg_duration_secs <= 0.0 {
                None
            } else {
                Some((accum.avg_elapsed_secs() / avg_duration_secs) * 3600.0)
            };
        language_pairs.insert(
            key,
            LanguagePairKnowledge {
                runs: accum.runs,
                success_rate: accum.success_rate(),
                avg_elapsed_secs: accum.avg_elapsed_secs(),
                avg_duration_secs,
                avg_elapsed_per_audio_hour_secs,
                best_elapsed_secs: accum.best_elapsed,
            },
        );
    }

    let mut content_kinds = HashMap::<String, ContentKnowledge>::new();
    for (key, accum) in content_accum {
        content_kinds.insert(
            key,
            ContentKnowledge {
                runs: accum.runs,
                success_rate: accum.success_rate(),
                avg_elapsed_secs: accum.avg_elapsed_secs(),
                avg_difficulty: if accum.difficulty_count == 0 {
                    None
                } else {
                    Some(accum.difficulty_sum / accum.difficulty_count as f64)
                },
                avg_scene_count: if accum.scene_count == 0 {
                    None
                } else {
                    Some(accum.scene_sum / accum.scene_count as f64)
                },
            },
        );
    }

    KnowledgeSnapshot {
        version: HISTORY_VERSION,
        generated_at_epoch_secs: now_epoch_secs(),
        devices,
        language_pairs,
        content_kinds,
    }
}

pub(super) fn save_knowledge_snapshot(path: &Path, history: &HistoryStore) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create knowledge directory {}: {error}",
                parent.display()
            )
        })?;
    }
    let snapshot = build_knowledge_snapshot(history);
    let serialized = serde_json::to_string_pretty(&snapshot)
        .map_err(|error| format!("failed to serialize knowledge snapshot: {error}"))?;
    fs::write(path, serialized).map_err(|error| {
        format!(
            "failed to write knowledge snapshot {}: {error}",
            path.display()
        )
    })
}

fn device_knowledge_key(record: &RunRecord) -> String {
    let cpu = record
        .cpu_cores
        .or_else(|| parse_prefixed_u64(&record.device_fingerprint, "cpu").map(|v| v as usize))
        .unwrap_or(0);
    let ram_mb = record
        .total_ram_mb
        .or_else(|| parse_prefixed_u64(&record.device_fingerprint, "-ram"));
    let backend = record_backend(record).unwrap_or_else(|| "unknown".to_string());
    let vram_mb = record_vram_mb(record).unwrap_or(0);
    let cpu_bucket = bucket_cpu_cores(cpu);
    let ram_bucket = bucket_ram_mb(ram_mb.unwrap_or(0));
    let vram_bucket = bucket_vram_mb(vram_mb);
    format!(
        "cpu{}-ram{}-gpu{}-vram{}",
        cpu_bucket, ram_bucket, backend, vram_bucket
    )
}
