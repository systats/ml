use std::fmt;

/// Errors returned by ml operations.
#[derive(Debug)]
pub enum MlError {
    /// Matrix is singular or not positive definite (e.g. alpha=0 with collinear features).
    SingularMatrix,
    /// Training data has zero rows.
    EmptyData,
    /// Input dimension does not match what the model was fitted on.
    DimensionMismatch { expected: usize, got: usize },
    /// Iterative solver failed to converge within the iteration budget.
    ConvergenceFailure(String),
    /// Invalid hyperparameter value (e.g. negative learning rate, zero trees).
    InvalidParameter(String),
    /// Serialization or deserialization failed (serde).
    SerializationError(String),
}

impl fmt::Display for MlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlError::SingularMatrix => {
                write!(f, "matrix is singular or not positive definite")
            }
            MlError::EmptyData => write!(f, "training data has zero rows"),
            MlError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            MlError::ConvergenceFailure(msg) => {
                write!(f, "convergence failure: {msg}")
            }
            MlError::InvalidParameter(msg) => {
                write!(f, "invalid parameter: {msg}")
            }
            MlError::SerializationError(msg) => {
                write!(f, "serialization error: {msg}")
            }
        }
    }
}

impl std::error::Error for MlError {}
