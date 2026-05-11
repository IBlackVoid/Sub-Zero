use std::path::PathBuf;

#[derive(Debug)]
pub enum DoomQlockError {
    WorkloadProbe { input: PathBuf, source: String },
    PlanValidation { input: PathBuf, source: String },
}

pub type DoomQlockResult<T> = Result<T, DoomQlockError>;

impl std::fmt::Display for DoomQlockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorkloadProbe { input, source } => {
                write!(
                    f,
                    "failed to profile workload for {}: {source}",
                    input.display()
                )
            }
            Self::PlanValidation { input, source } => {
                write!(
                    f,
                    "failed to validate execution plan for {}: {source}",
                    input.display()
                )
            }
        }
    }
}

impl std::error::Error for DoomQlockError {}
