use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("failed to initialize subtitle pipeline: {message}")]
    Initialization { message: String },
    #[error("failed to process {}: {message}", input.display())]
    ProcessInput { input: PathBuf, message: String },
}

pub type PipelineResult<T> = Result<T, PipelineError>;
