use std::path::PathBuf;

#[derive(Debug)]
pub enum PipelineError {
    Initialization { source: String },
    ProcessInput { input: PathBuf, source: String },
}

pub type PipelineResult<T> = Result<T, PipelineError>;

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initialization { source } => {
                write!(f, "failed to initialize subtitle pipeline: {source}")
            }
            Self::ProcessInput { input, source } => {
                write!(f, "failed to process {}: {source}", input.display())
            }
        }
    }
}

impl std::error::Error for PipelineError {}
