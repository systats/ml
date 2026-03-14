//! Integration tests for `LogisticModel` (public API only).

use ml::error::MlError;
use ml::logistic::LogisticModel;
use nalgebra::DMatrix;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Linearly separable binary data.
/// Class 0 (rows 0..30): x1 centered at -3.
/// Class 1 (rows 30..60): x1 centered at +3.
fn make_binary_data() -> (DMatrix<f64>, Vec<i64>) {
    let n = 60;
    let x = DMatrix::from_fn(n, 2, |i, j| {
        if j == 0 {
            if i < 30 {
                -3.0 + (i % 5) as f64 * 0.1
            } else {
                3.0 + ((i - 30) % 5) as f64 * 0.1
            }
        } else {
            // Small variation in second feature
            (i as f64 * 0.13).sin() * 0.3
        }
    });
    let y: Vec<i64> = (0..n).map(|i| if i < 30 { 0 } else { 1 }).collect();
    (x, y)
}

/// Three clearly separated clusters (3-class problem).
/// Class 0: centered at (0, 3), Class 1: (-3, -2), Class 2: (3, -2).
fn make_multiclass_data() -> (DMatrix<f64>, Vec<i64>) {
    let n = 90; // 30 per class
    let centers = [(0.0f64, 3.0f64), (-3.0, -2.0), (3.0, -2.0)];
    let x = DMatrix::from_fn(n, 2, |i, j| {
        let class_idx = i / 30;
        let (cx, cy) = centers[class_idx];
        let base = if j == 0 { cx } else { cy };
        base + (i as f64 * 0.17 + j as f64 * 0.31).sin() * 0.3
    });
    let y: Vec<i64> = (0..n).map(|i| (i / 30) as i64).collect();
    (x, y)
}

fn accuracy(y_true: &[i64], y_pred: &[i64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len());
    let correct = y_true.iter().zip(y_pred).filter(|(a, b)| a == b).count();
    correct as f64 / y_true.len() as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_binary_accuracy() {
    let (x, y) = make_binary_data();
    let mut model = LogisticModel::new(1.0, 500);
    model.fit(&x, &y, None).unwrap();
    let preds = model.predict(&x);
    let acc = accuracy(&y, &preds);
    assert!(acc > 0.9, "binary accuracy {acc:.3} < 0.9");
}

#[test]
fn test_multiclass_accuracy() {
    let (x, y) = make_multiclass_data();
    let mut model = LogisticModel::new(1.0, 500);
    model.fit(&x, &y, None).unwrap();
    let preds = model.predict(&x);
    let acc = accuracy(&y, &preds);
    assert!(acc > 0.8, "multiclass accuracy {acc:.3} < 0.8");
}

#[test]
fn test_n_classes_set_correctly() {
    let (x, y) = make_binary_data();
    let mut model = LogisticModel::new(1.0, 200);
    model.fit(&x, &y, None).unwrap();
    assert_eq!(model.n_classes, 2);

    let (x3, y3) = make_multiclass_data();
    let mut model3 = LogisticModel::new(1.0, 200);
    model3.fit(&x3, &y3, None).unwrap();
    assert_eq!(model3.n_classes, 3);
}

#[test]
fn test_predict_proba_shape() {
    let (x, y) = make_binary_data();
    let mut model = LogisticModel::new(1.0, 200);
    model.fit(&x, &y, None).unwrap();
    let proba = model.predict_proba(&x);
    assert_eq!(proba.nrows(), x.nrows(), "proba row count");
    assert_eq!(proba.ncols(), 2, "binary proba should have 2 columns");
}

#[test]
fn test_predict_proba_rows_sum_to_one() {
    let (x, y) = make_binary_data();
    let mut model = LogisticModel::new(1.0, 200);
    model.fit(&x, &y, None).unwrap();
    let proba = model.predict_proba(&x);
    for i in 0..proba.nrows() {
        let row_sum: f64 = proba.row(i).iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-10,
            "row {i} sum = {row_sum:.8} (expected 1.0)"
        );
    }
}

#[test]
fn test_predict_proba_multiclass_rows_sum_to_one() {
    let (x, y) = make_multiclass_data();
    let mut model = LogisticModel::new(1.0, 200);
    model.fit(&x, &y, None).unwrap();
    let proba = model.predict_proba(&x);
    assert_eq!(proba.ncols(), 3, "3-class proba should have 3 columns");
    for i in 0..proba.nrows() {
        let row_sum: f64 = proba.row(i).iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-10,
            "multiclass row {i} sum = {row_sum:.8}"
        );
    }
}

#[test]
fn test_predict_proba_values_in_range() {
    let (x, y) = make_binary_data();
    let mut model = LogisticModel::new(1.0, 200);
    model.fit(&x, &y, None).unwrap();
    let proba = model.predict_proba(&x);
    for p in proba.iter() {
        assert!(*p >= 0.0 && *p <= 1.0, "probability {p} out of [0, 1]");
    }
}

#[test]
fn test_higher_regularization_smaller_coef_norm() {
    // Lower C = stronger regularization → smaller coefficient norms.
    let (x, y) = make_binary_data();
    let mut m_weak = LogisticModel::new(100.0, 500); // weak regularization
    let mut m_strong = LogisticModel::new(0.01, 500); // strong regularization
    m_weak.fit(&x, &y, None).unwrap();
    m_strong.fit(&x, &y, None).unwrap();

    // Compare feature weights (index 1..) for the single OvR classifier
    let norm_weak: f64 = m_weak.coefs[0].rows(1, x.ncols()).norm();
    let norm_strong: f64 = m_strong.coefs[0].rows(1, x.ncols()).norm();
    assert!(
        norm_strong < norm_weak,
        "strong reg norm {norm_strong:.4} >= weak reg norm {norm_weak:.4}"
    );
}

#[test]
fn test_reproducibility() {
    let (x, y) = make_binary_data();
    let mut m1 = LogisticModel::new(1.0, 200);
    let mut m2 = LogisticModel::new(1.0, 200);
    m1.fit(&x, &y, None).unwrap();
    m2.fit(&x, &y, None).unwrap();

    let p1 = m1.predict(&x);
    let p2 = m2.predict(&x);
    assert_eq!(p1, p2, "identical fits must produce identical predictions");
}

#[test]
fn test_empty_data_returns_error() {
    let x = DMatrix::<f64>::zeros(0, 2);
    let y: Vec<i64> = vec![];
    let mut model = LogisticModel::new(1.0, 100);
    let result = model.fit(&x, &y, None);
    assert!(
        matches!(result, Err(MlError::EmptyData)),
        "expected EmptyData error"
    );
}

#[test]
fn test_dimension_mismatch_returns_error() {
    let x = DMatrix::<f64>::zeros(10, 2);
    let y: Vec<i64> = vec![0; 7]; // wrong length
    let mut model = LogisticModel::new(1.0, 100);
    let result = model.fit(&x, &y, None);
    assert!(
        matches!(
            result,
            Err(MlError::DimensionMismatch {
                expected: 10,
                got: 7
            })
        ),
        "expected DimensionMismatch(10, 7)"
    );
}

// ---------------------------------------------------------------------------
// Probability clipping tests (Wave 3B: numerical stability)
// ---------------------------------------------------------------------------

/// Helper: create extreme data that drives sigmoid toward 0 or 1.
/// Class 0 at x = -100, class 1 at x = +100 — after fitting, the model
/// will produce near-saturated probabilities.
fn make_extreme_data() -> (DMatrix<f64>, Vec<i64>) {
    let n = 40;
    let x = DMatrix::from_fn(n, 1, |i, _| {
        if i < 20 { -100.0 } else { 100.0 }
    });
    let y: Vec<i64> = (0..n).map(|i| if i < 20 { 0 } else { 1 }).collect();
    (x, y)
}

#[test]
fn test_predict_proba_never_exact_zero_or_one_binary() {
    let (x, y) = make_extreme_data();
    let mut model = LogisticModel::new(1.0, 500);
    model.fit(&x, &y, None).unwrap();

    let proba = model.predict_proba(&x);
    let eps = 1e-15;
    for i in 0..proba.nrows() {
        for j in 0..proba.ncols() {
            let p = proba[(i, j)];
            assert!(
                p >= eps,
                "proba[{i},{j}] = {p} is below eps={eps} (exact 0.0 causes log(0)=-inf)"
            );
            assert!(
                p <= 1.0 - eps,
                "proba[{i},{j}] = {p} is above 1-eps={} (exact 1.0 causes log(0)=-inf for complement)",
                1.0 - eps
            );
        }
    }
}

#[test]
fn test_predict_proba_never_exact_zero_or_one_multiclass() {
    // 3-class with extreme separation
    let n = 60;
    let x = DMatrix::from_fn(n, 2, |i, j| {
        let cls = i / 20;
        let center = match cls {
            0 => (-100.0, 0.0),
            1 => (100.0, -100.0),
            _ => (100.0, 100.0),
        };
        if j == 0 { center.0 } else { center.1 }
    });
    let y: Vec<i64> = (0..n).map(|i| (i / 20) as i64).collect();

    let mut model = LogisticModel::new(1.0, 500);
    model.fit(&x, &y, None).unwrap();

    let proba = model.predict_proba(&x);
    let eps = 1e-15;
    for i in 0..proba.nrows() {
        for j in 0..proba.ncols() {
            let p = proba[(i, j)];
            assert!(
                p >= eps && p <= 1.0 - eps,
                "multiclass proba[{i},{j}] = {p} outside [{eps}, {}]",
                1.0 - eps
            );
        }
    }
}

#[test]
fn test_predict_proba_row_major_never_exact_zero_or_one_binary() {
    let n = 40;
    let p = 1;
    let x_rm: Vec<f64> = (0..n).map(|i| if i < 20 { -100.0 } else { 100.0 }).collect();
    let y: Vec<i64> = (0..n).map(|i| if i < 20 { 0 } else { 1 }).collect();

    let mut model = LogisticModel::new(1.0, 500);
    model.fit_row_major(&x_rm, &y, n, p, None).unwrap();

    let proba = model.predict_proba_row_major(&x_rm, n, p);
    let eps = 1e-15;
    let k = 2;
    for i in 0..n {
        for j in 0..k {
            let prob = proba[i * k + j];
            assert!(
                prob >= eps && prob <= 1.0 - eps,
                "row_major proba[{i},{j}] = {prob} outside [{eps}, {}]",
                1.0 - eps
            );
        }
    }
}

#[test]
fn test_predict_proba_log_safe() {
    // The primary purpose of clipping: log(proba) must not produce -inf or NaN.
    let (x, y) = make_extreme_data();
    let mut model = LogisticModel::new(1.0, 500);
    model.fit(&x, &y, None).unwrap();

    let proba = model.predict_proba(&x);
    for i in 0..proba.nrows() {
        for j in 0..proba.ncols() {
            let log_p = proba[(i, j)].ln();
            assert!(
                log_p.is_finite(),
                "log(proba[{i},{j}]) = {log_p} is not finite (proba = {})",
                proba[(i, j)]
            );
        }
    }
}
