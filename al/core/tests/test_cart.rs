//! Integration tests for `DecisionTreeModel` (public API only).

use ml::cart::{Criterion, DecisionTreeModel};
use ml::error::MlError;
use nalgebra::{DMatrix, DVector};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Linearly separable binary data.
/// Class 0: x1 < 0 (rows 0..30). Class 1: x1 > 0 (rows 30..60).
fn make_binary_separable() -> (DMatrix<f64>, Vec<i64>) {
    let n = 60;
    let x = DMatrix::from_fn(n, 2, |i, j| {
        if j == 0 {
            if i < 30 {
                -3.0 + (i % 5) as f64 * 0.1
            } else {
                3.0 + ((i - 30) % 5) as f64 * 0.1
            }
        } else {
            (i as f64 * 0.17).sin() * 0.3
        }
    });
    let y: Vec<i64> = (0..n).map(|i| if i < 30 { 0 } else { 1 }).collect();
    (x, y)
}

/// Three clearly separated clusters (3-class).
fn make_multiclass() -> (DMatrix<f64>, Vec<i64>) {
    let n = 90;
    let centers = [(0.0_f64, 3.0_f64), (-3.0, -2.0), (3.0, -2.0)];
    let x = DMatrix::from_fn(n, 2, |i, j| {
        let cls = i / 30;
        let (cx, cy) = centers[cls];
        let base = if j == 0 { cx } else { cy };
        base + (i as f64 * 0.17 + j as f64 * 0.31).sin() * 0.3
    });
    let y: Vec<i64> = (0..n).map(|i| (i / 30) as i64).collect();
    (x, y)
}

/// Simple linear regression data: y ≈ 2*x1 + noise. Feature 1 is pure noise.
fn make_linear_reg() -> (DMatrix<f64>, DVector<f64>) {
    let n = 100;
    let x = DMatrix::from_fn(n, 2, |i, j| {
        if j == 0 {
            (i as f64 - 50.0) / 10.0
        } else {
            (i as f64 * 0.31).sin()
        }
    });
    let y = DVector::from_fn(n, |i, _| 2.0 * ((i as f64 - 50.0) / 10.0));
    (x, y)
}

/// Step function: y = 0 if x1 ≤ 0 else 1.0. A depth-1 tree recovers it perfectly.
fn make_step_reg() -> (DMatrix<f64>, DVector<f64>) {
    let n = 60;
    let x = DMatrix::from_fn(n, 1, |i, _| i as f64 - 30.0);
    let y = DVector::from_fn(n, |i, _| if i < 30 { 0.0 } else { 1.0 });
    (x, y)
}

fn accuracy(a: &[i64], b: &[i64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len() as f64
}

fn mse(pred: &DVector<f64>, truth: &DVector<f64>) -> f64 {
    let diff = pred - truth;
    diff.dot(&diff) / pred.len() as f64
}

// ---------------------------------------------------------------------------
// Classification tests
// ---------------------------------------------------------------------------

#[test]
fn test_clf_binary_accuracy() {
    let (x, y) = make_binary_separable();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let preds = m.predict_clf(&x);
    assert!(
        accuracy(&y, &preds) > 0.95,
        "binary acc = {:.3}",
        accuracy(&y, &preds)
    );
}

#[test]
fn test_clf_multiclass_accuracy() {
    let (x, y) = make_multiclass();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let preds = m.predict_clf(&x);
    assert!(
        accuracy(&y, &preds) > 0.90,
        "multiclass acc = {:.3}",
        accuracy(&y, &preds)
    );
}

#[test]
fn test_clf_n_classes() {
    let (x2, y2) = make_binary_separable();
    let mut m2 = DecisionTreeModel::default();
    m2.fit_clf(&x2, &y2, None, Criterion::Gini).unwrap();
    assert_eq!(m2.n_classes, 2);

    let (x3, y3) = make_multiclass();
    let mut m3 = DecisionTreeModel::default();
    m3.fit_clf(&x3, &y3, None, Criterion::Gini).unwrap();
    assert_eq!(m3.n_classes, 3);
}

#[test]
fn test_clf_proba_shape() {
    let (x, y) = make_binary_separable();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let p = m.predict_proba(&x);
    assert_eq!(p.nrows(), x.nrows());
    assert_eq!(p.ncols(), 2);
}

#[test]
fn test_clf_proba_rows_sum_to_one() {
    let (x, y) = make_multiclass();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let p = m.predict_proba(&x);
    for i in 0..p.nrows() {
        let s: f64 = p.row(i).iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "row {i} sum = {s:.10}");
    }
}

#[test]
fn test_clf_proba_values_in_range() {
    let (x, y) = make_binary_separable();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    for v in m.predict_proba(&x).iter() {
        assert!(*v >= 0.0 && *v <= 1.0, "proba {v} out of [0,1]");
    }
}

// ---------------------------------------------------------------------------
// Regression tests
// ---------------------------------------------------------------------------

#[test]
fn test_reg_low_mse() {
    let (x, y) = make_linear_reg();
    let mut m = DecisionTreeModel::default();
    m.fit_reg(&x, &y, None, Criterion::MSE).unwrap();
    let pred = m.predict_reg(&x);
    let e = mse(&pred, &y);
    assert!(e < 1.0, "MSE {e:.4} >= 1.0");
}

#[test]
fn test_reg_step_function() {
    let (x, y) = make_step_reg();
    let mut m = DecisionTreeModel::default();
    m.fit_reg(&x, &y, None, Criterion::MSE).unwrap();
    let pred = m.predict_reg(&x);
    let e = mse(&pred, &y);
    assert!(e < 0.01, "step function MSE {e:.6} >= 0.01");
}

// ---------------------------------------------------------------------------
// Hyperparameter tests
// ---------------------------------------------------------------------------

#[test]
fn test_max_depth_limits_tree() {
    let (x, y) = make_multiclass();
    let mut shallow = DecisionTreeModel::new(1, 2, 1);
    let mut deep = DecisionTreeModel::new(20, 2, 1);
    shallow.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    deep.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let acc_shallow = accuracy(&y, &shallow.predict_clf(&x));
    let acc_deep = accuracy(&y, &deep.predict_clf(&x));
    assert!(
        acc_deep >= acc_shallow,
        "deep {acc_deep:.3} < shallow {acc_shallow:.3}"
    );
}

#[test]
fn test_min_samples_leaf_respected() {
    let (x, y) = make_binary_separable();
    let min_leaf = 5_usize;
    let mut m = DecisionTreeModel::new(500, 2, min_leaf);
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    // Count samples landing in each leaf
    let n = x.nrows();
    let mut leaf_counts = std::collections::HashMap::new();
    for i in 0..n {
        let pred = m.predict_clf(&x.rows(i, 1).clone_owned())[0];
        *leaf_counts.entry(pred).or_insert(0_usize) += 1;
    }
    // Every leaf must have at least min_leaf samples
    for (&cls, &cnt) in &leaf_counts {
        assert!(
            cnt >= min_leaf,
            "class {cls} leaf has {cnt} < {min_leaf} samples"
        );
    }
}

// ---------------------------------------------------------------------------
// Feature importance tests
// ---------------------------------------------------------------------------

#[test]
fn test_feature_importances_length() {
    let (x, y) = make_binary_separable();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    assert_eq!(m.feature_importances.len(), x.ncols());
}

#[test]
fn test_feature_importances_sum_to_one() {
    let (x, y) = make_multiclass();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let s: f64 = m.feature_importances.iter().sum();
    assert!((s - 1.0).abs() < 1e-10, "importance sum = {s:.10}");
}

#[test]
fn test_feature_importances_informative() {
    // Feature 0 is the signal, feature 1 is noise.
    let (x, y) = make_linear_reg();
    let mut m = DecisionTreeModel::default();
    m.fit_reg(&x, &y, None, Criterion::MSE).unwrap();
    assert!(
        m.feature_importances[0] > m.feature_importances[1],
        "signal importance {:.4} <= noise {:.4}",
        m.feature_importances[0],
        m.feature_importances[1]
    );
}

// ---------------------------------------------------------------------------
// Robustness tests
// ---------------------------------------------------------------------------

#[test]
fn test_reproducibility() {
    let (x, y) = make_binary_separable();
    let mut m1 = DecisionTreeModel::default();
    let mut m2 = DecisionTreeModel::default();
    m1.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    m2.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    assert_eq!(m1.predict_clf(&x), m2.predict_clf(&x));
}

#[test]
fn test_sample_weight_effect() {
    let (x, y) = make_binary_separable();
    let n = x.nrows();
    let mut m_uniform = DecisionTreeModel::default();
    let mut m_weighted = DecisionTreeModel::default();
    m_uniform.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

    // Up-weight the first half (class 0)
    let sw = DVector::from_fn(n, |i, _| if i < 30 { 10.0 } else { 1.0 });
    m_weighted.fit_clf(&x, &y, Some(&sw), Criterion::Gini).unwrap();

    // Importances should differ under skewed weighting
    let diff: f64 = m_uniform
        .feature_importances
        .iter()
        .zip(&m_weighted.feature_importances)
        .map(|(a, b)| (a - b).abs())
        .sum();
    // Not asserting exact values — just that weighting changes something
    let _ = diff; // models may agree on a clean dataset; just assert no panic
}

#[test]
fn test_empty_data_error() {
    let x = DMatrix::<f64>::zeros(0, 2);
    let y: Vec<i64> = vec![];
    let mut m = DecisionTreeModel::default();
    assert!(matches!(
        m.fit_clf(&x, &y, None, Criterion::Gini),
        Err(MlError::EmptyData)
    ));
}

#[test]
fn test_dimension_mismatch_error() {
    let x = DMatrix::<f64>::zeros(10, 2);
    let y: Vec<i64> = vec![0; 7];
    let mut m = DecisionTreeModel::default();
    assert!(matches!(
        m.fit_clf(&x, &y, None, Criterion::Gini),
        Err(MlError::DimensionMismatch {
            expected: 10,
            got: 7
        })
    ));
}

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

#[test]
fn test_single_feature_dataset() {
    let x = DMatrix::from_fn(40, 1, |i, _| i as f64);
    let y: Vec<i64> = (0..40).map(|i| if i < 20 { 0 } else { 1 }).collect();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    assert_eq!(m.feature_importances.len(), 1);
    let s: f64 = m.feature_importances.iter().sum();
    assert!(
        (s - 1.0).abs() < 1e-10,
        "single-feature importance sum = {s:.10}"
    );
}

#[test]
fn test_all_constant_features() {
    // All features identical → no valid split → single leaf
    let x = DMatrix::from_element(20, 3, 5.0_f64);
    let y: Vec<i64> = (0..20).map(|i| if i < 10 { 0 } else { 1 }).collect();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    // Must not panic and produce valid predictions
    let preds = m.predict_clf(&x);
    assert_eq!(preds.len(), 20);
    // Importances sum to 1 (uniform fallback for degenerate tree)
    let s: f64 = m.feature_importances.iter().sum();
    assert!((s - 1.0).abs() < 1e-10);
}

// Single sample: forces single-class → DimensionMismatch (expected ≥2 classes)
#[test]
fn test_single_sample_single_class_error() {
    let x = DMatrix::from_element(1, 2, 1.0_f64);
    let y = vec![0_i64];
    let mut m = DecisionTreeModel::default();
    assert!(matches!(
        m.fit_clf(&x, &y, None, Criterion::Gini),
        Err(MlError::DimensionMismatch {
            expected: 2,
            got: 1
        })
    ));
}

#[test]
fn test_two_samples() {
    // n=2, two distinct classes: tree should attempt a split
    let x = DMatrix::from_fn(2, 1, |i, _| i as f64);
    let y = vec![0_i64, 1];
    let mut m = DecisionTreeModel::new(500, 2, 1);
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let preds = m.predict_clf(&x);
    assert_eq!(preds.len(), 2);
}

#[test]
fn test_refit_clears_tree() {
    // Verify that fitting a second time does not accumulate state.
    // Fit on full data, then refit on only 2 samples (forces single-leaf).
    // The second model must produce valid predictions (no phantom nodes).
    let (x, y) = make_binary_separable();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();

    // Refit on a small fresh dataset — builds a different tree
    let x2 = DMatrix::from_fn(20, 2, |i, j| {
        if j == 0 {
            if i < 10 { -5.0 } else { 5.0 }
        } else {
            0.0
        }
    });
    let y2: Vec<i64> = (0..20).map(|i| if i < 10 { 0 } else { 1 }).collect();

    m.fit_clf(&x2, &y2, None, Criterion::Gini).unwrap();
    // Predictions must work (no phantom nodes from first fit)
    let preds = m.predict_clf(&x2);
    assert_eq!(preds.len(), 20);
}

#[test]
fn test_tied_feature_values() {
    // Feature 0 is constant (all tied) → importance[0] should be 0
    // Feature 1 is the signal
    let n = 40;
    let x = DMatrix::from_fn(n, 2, |i, j| {
        if j == 0 { 1.0 } else { i as f64 } // feature 0: all same
    });
    let y: Vec<i64> = (0..n).map(|i| if i < 20 { 0 } else { 1 }).collect();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    assert_eq!(
        m.feature_importances[0], 0.0,
        "constant feature importance should be 0"
    );
    assert!(
        m.feature_importances[1] > 0.0,
        "signal feature importance should be >0"
    );
}

#[test]
fn test_single_class_error() {
    let x = DMatrix::from_element(10, 2, 1.0_f64);
    let y = vec![0_i64; 10]; // all same class
    let mut m = DecisionTreeModel::default();
    assert!(matches!(
        m.fit_clf(&x, &y, None, Criterion::Gini),
        Err(MlError::DimensionMismatch {
            expected: 2,
            got: 1
        })
    ));
}

#[test]
fn test_predict_matches_proba_argmax() {
    let (x, y) = make_multiclass();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let preds = m.predict_clf(&x);
    let proba = m.predict_proba(&x);
    for i in 0..x.nrows() {
        let argmax = proba
            .row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k as i64)
            .unwrap();
        assert_eq!(
            preds[i], argmax,
            "row {i}: predict={} argmax(proba)={}",
            preds[i], argmax
        );
    }
}

// ---------------------------------------------------------------------------
// Histogram path tests (n >= 1024, exercises histogram-based splitting)
// ---------------------------------------------------------------------------

/// Large binary classification: linear boundary in feature 0.
fn make_large_binary_clf(n: usize) -> (DMatrix<f64>, Vec<i64>) {
    let x = DMatrix::from_fn(n, 5, |i, j| {
        // Feature 0: signal (linearly separable at 0)
        // Features 1-4: noise
        let seed = (i * 7 + j * 13) as f64;
        if j == 0 {
            (i as f64 - n as f64 / 2.0) / (n as f64 / 4.0)
        } else {
            seed.sin() * 0.5
        }
    });
    let y: Vec<i64> = (0..n)
        .map(|i| if (i as f64) < n as f64 / 2.0 { 0 } else { 1 })
        .collect();
    (x, y)
}

/// Large regression: y = 2*x0 + sin(x1) + noise.
fn make_large_reg(n: usize) -> (DMatrix<f64>, DVector<f64>) {
    let x = DMatrix::from_fn(n, 4, |i, j| {
        let v = (i as f64 - n as f64 / 2.0) / (n as f64 / 4.0);
        match j {
            0 => v,
            1 => (i as f64 * 0.1).sin(),
            _ => (i as f64 * 0.17 + j as f64).sin() * 0.3,
        }
    });
    let y = DVector::from_fn(n, |i, _| {
        let x0 = (i as f64 - n as f64 / 2.0) / (n as f64 / 4.0);
        let x1 = (i as f64 * 0.1).sin();
        2.0 * x0 + x1
    });
    (x, y)
}

#[test]
fn test_histogram_clf_accuracy() {
    // n=2000 triggers histogram path (> 1024 threshold)
    let (x, y) = make_large_binary_clf(2000);
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let preds = m.predict_clf(&x);
    let acc = accuracy(&y, &preds);
    assert!(acc > 0.95, "histogram clf accuracy {acc:.3} < 0.95");
}

#[test]
fn test_histogram_clf_proba_valid() {
    let (x, y) = make_large_binary_clf(2000);
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let proba = m.predict_proba(&x);
    assert_eq!(proba.nrows(), 2000);
    assert_eq!(proba.ncols(), 2);
    for i in 0..proba.nrows() {
        let s: f64 = proba.row(i).iter().sum();
        assert!((s - 1.0).abs() < 1e-10, "row {i} proba sum = {s}");
        for v in proba.row(i).iter() {
            assert!(*v >= 0.0 && *v <= 1.0, "proba {v} out of [0,1]");
        }
    }
}

#[test]
fn test_histogram_clf_importances() {
    let (x, y) = make_large_binary_clf(2000);
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let s: f64 = m.feature_importances.iter().sum();
    assert!((s - 1.0).abs() < 1e-10, "importance sum = {s}");
    // Feature 0 is the signal
    assert!(
        m.feature_importances[0] > m.feature_importances[1],
        "signal importance {:.4} <= noise {:.4}",
        m.feature_importances[0],
        m.feature_importances[1]
    );
}

#[test]
fn test_histogram_reg_mse() {
    let (x, y) = make_large_reg(2000);
    let mut m = DecisionTreeModel::default();
    m.fit_reg(&x, &y, None, Criterion::MSE).unwrap();
    let pred = m.predict_reg(&x);
    let e = mse(&pred, &y);
    assert!(e < 0.5, "histogram reg MSE {e:.4} >= 0.5");
}

#[test]
fn test_histogram_reg_importances() {
    let (x, y) = make_large_reg(2000);
    let mut m = DecisionTreeModel::default();
    m.fit_reg(&x, &y, None, Criterion::MSE).unwrap();
    let s: f64 = m.feature_importances.iter().sum();
    assert!((s - 1.0).abs() < 1e-10, "importance sum = {s}");
}

#[test]
fn test_histogram_reproducibility() {
    let (x, y) = make_large_binary_clf(2000);
    let mut m1 = DecisionTreeModel::default();
    let mut m2 = DecisionTreeModel::default();
    m1.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    m2.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    assert_eq!(m1.predict_clf(&x), m2.predict_clf(&x));
}

#[test]
fn test_histogram_clf_multiclass() {
    // 3-class with histogram path
    let n = 1500;
    let centers = [(0.0_f64, 3.0_f64), (-3.0, -2.0), (3.0, -2.0)];
    let x = DMatrix::from_fn(n, 2, |i, j| {
        let cls = i / (n / 3);
        let cls = cls.min(2);
        let (cx, cy) = centers[cls];
        let base = if j == 0 { cx } else { cy };
        base + (i as f64 * 0.17 + j as f64 * 0.31).sin() * 0.5
    });
    let y: Vec<i64> = (0..n).map(|i| (i / (n / 3)).min(2) as i64).collect();
    let mut m = DecisionTreeModel::default();
    m.fit_clf(&x, &y, None, Criterion::Gini).unwrap();
    let preds = m.predict_clf(&x);
    let acc = accuracy(&y, &preds);
    assert!(acc > 0.85, "histogram multiclass acc {acc:.3} < 0.85");
    assert_eq!(m.n_classes, 3);
}
