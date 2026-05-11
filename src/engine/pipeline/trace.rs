use super::util::now_epoch_secs;
use super::PipelineConfig;
use serde_json::json;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone)]
pub(super) struct RuntimeTrace {
    started_at_epoch_secs: u64,
    run_started: Instant,
    stages: Vec<RuntimeTraceStage>,
}

#[derive(Debug, Clone)]
struct RuntimeTraceStage {
    name: String,
    elapsed_secs: f64,
    details: serde_json::Value,
}

impl RuntimeTrace {
    pub(super) fn new() -> Self {
        Self {
            started_at_epoch_secs: now_epoch_secs(),
            run_started: Instant::now(),
            stages: Vec::new(),
        }
    }

    pub(super) fn record_stage(
        &mut self,
        name: &str,
        started: Instant,
        details: serde_json::Value,
    ) {
        self.stages.push(RuntimeTraceStage {
            name: name.to_string(),
            elapsed_secs: started.elapsed().as_secs_f64(),
            details,
        });
    }

    pub(super) fn as_json(
        &self,
        input: &Path,
        output: &Path,
        config: &PipelineConfig,
    ) -> serde_json::Value {
        json!({
            "version": "1.0",
            "trace_kind": "runtime-performance",
            "algorithm": "IBVoid DOOM-QLOCK",
            "source_file": input.display().to_string(),
            "output_file": output.display().to_string(),
            "source_language": config.source_lang,
            "target_language": config.target_lang,
            "quality_profile": config.quality_profile.as_str(),
            "started_at_epoch_secs": self.started_at_epoch_secs,
            "finished_at_epoch_secs": now_epoch_secs(),
            "total_elapsed_secs": self.run_started.elapsed().as_secs_f64(),
            "plan_used": {
                "parallel": config.parallel,
                "workers": config.max_workers,
                "chunk_duration_secs": config.chunk_duration_secs,
                "mt_batch_size": config.mt_batch_size,
                "mt_max_batch_tokens": config.mt_max_batch_tokens,
                "mt_oom_retries": config.mt_oom_retries,
                "mt_allow_cpu_fallback": config.mt_allow_cpu_fallback,
                "mt_daemon": config.mt_daemon,
            },
            "stages": self.stages.iter().map(|stage| {
                json!({
                    "name": stage.name,
                    "elapsed_secs": stage.elapsed_secs,
                    "details": stage.details,
                })
            }).collect::<Vec<_>>(),
        })
    }
}
