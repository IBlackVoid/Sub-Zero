use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DoomQlockError {
    #[error("failed to profile workload for {}: {message}", input.display())]
    WorkloadProbe { input: PathBuf, message: String },
    #[error("failed to validate execution plan for {}: {message}", input.display())]
    PlanValidation { input: PathBuf, message: String },
}

pub type DoomQlockResult<T> = Result<T, DoomQlockError>;
