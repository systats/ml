//! Tests for MlError variants and their Display implementations.

use ml::error::MlError;

#[test]
fn test_singular_matrix_display() {
    let e = MlError::SingularMatrix;
    let msg = format!("{e}");
    assert_eq!(msg, "matrix is singular or not positive definite");
}

#[test]
fn test_empty_data_display() {
    let e = MlError::EmptyData;
    let msg = format!("{e}");
    assert_eq!(msg, "training data has zero rows");
}

#[test]
fn test_dimension_mismatch_display() {
    let e = MlError::DimensionMismatch { expected: 100, got: 42 };
    let msg = format!("{e}");
    assert_eq!(msg, "dimension mismatch: expected 100, got 42");
}

#[test]
fn test_convergence_failure_display() {
    let e = MlError::ConvergenceFailure("L-BFGS did not converge in 1000 iterations".to_string());
    let msg = format!("{e}");
    assert_eq!(msg, "convergence failure: L-BFGS did not converge in 1000 iterations");
}

#[test]
fn test_invalid_parameter_display() {
    let e = MlError::InvalidParameter("learning_rate must be > 0, got -0.1".to_string());
    let msg = format!("{e}");
    assert_eq!(msg, "invalid parameter: learning_rate must be > 0, got -0.1");
}

#[test]
fn test_serialization_error_display() {
    let e = MlError::SerializationError("invalid JSON at byte 42".to_string());
    let msg = format!("{e}");
    assert_eq!(msg, "serialization error: invalid JSON at byte 42");
}

#[test]
fn test_convergence_failure_empty_message() {
    let e = MlError::ConvergenceFailure(String::new());
    let msg = format!("{e}");
    assert_eq!(msg, "convergence failure: ");
}

#[test]
fn test_invalid_parameter_empty_message() {
    let e = MlError::InvalidParameter(String::new());
    let msg = format!("{e}");
    assert_eq!(msg, "invalid parameter: ");
}

#[test]
fn test_serialization_error_empty_message() {
    let e = MlError::SerializationError(String::new());
    let msg = format!("{e}");
    assert_eq!(msg, "serialization error: ");
}

#[test]
fn test_error_is_debug() {
    // All variants must implement Debug (used by assert!, unwrap, etc.)
    let errors: Vec<MlError> = vec![
        MlError::SingularMatrix,
        MlError::EmptyData,
        MlError::DimensionMismatch { expected: 1, got: 2 },
        MlError::ConvergenceFailure("test".to_string()),
        MlError::InvalidParameter("test".to_string()),
        MlError::SerializationError("test".to_string()),
    ];
    for e in &errors {
        let debug = format!("{e:?}");
        assert!(!debug.is_empty(), "Debug output should not be empty");
    }
}

#[test]
fn test_error_implements_std_error() {
    // Verify the std::error::Error trait is implemented
    fn assert_std_error<E: std::error::Error>(_e: &E) {}

    assert_std_error(&MlError::SingularMatrix);
    assert_std_error(&MlError::EmptyData);
    assert_std_error(&MlError::DimensionMismatch { expected: 1, got: 2 });
    assert_std_error(&MlError::ConvergenceFailure("test".to_string()));
    assert_std_error(&MlError::InvalidParameter("test".to_string()));
    assert_std_error(&MlError::SerializationError("test".to_string()));
}
