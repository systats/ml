//! Integration tests for `RandomForestModel` (public API only).

use ml::forest::RandomForestModel;
use nalgebra::{DMatrix, DVector};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Linearly separable binary data (200 samples for RF).
fn make_binary_clf() -> (DMatrix<f64>, Vec<i64>) {
    let n = 200;
    let x = DMatrix::from_fn(n, 4, |i, j| {
        let base = if i < 100 { -2.0 } else { 2.0 };
        if j == 0 {
            base + (i as f64 * 0.17).sin() * 0.5
        } else {
            (i as f64 * (0.13 + j as f64 * 0.07)).sin() * 2.0
        }
    });
    let y: Vec<i64> = (0..n).map(|i| if i < 100 { 0 } else { 1 }).collect();
    (x, y)
}

/// Three-class classification (300 samples).
fn make_multiclass_clf() -> (DMatrix<f64>, Vec<i64>) {
    let n = 300;
    let centers = [(0.0_f64, 4.0_f64), (-4.0, -2.0), (4.0, -2.0)];
    let x = DMatrix::from_fn(n, 3, |i, j| {
        let cls = i / 100;
        let (cx, cy) = centers[cls];
        let base = match j {
            0 => cx,
            1 => cy,
            _ => 0.0,
        };
        base + (i as f64 * 0.17 + j as f64 * 0.31).sin() * 0.5
    });
    let y: Vec<i64> = (0..n).map(|i| (i / 100) as i64).collect();
    (x, y)
}

/// Linear regression data: y ≈ 3*x0 - 2*x1 + noise (200 samples).
fn make_regression() -> (DMatrix<f64>, DVector<f64>) {
    let n = 200;
    let x = DMatrix::from_fn(n, 4, |i, j| {
        (i as f64 * (0.13 + j as f64 * 0.07)).sin() * 3.0
    });
    let y = DVector::from_fn(n, |i, _| {
        let x0 = x[(i, 0)];
        let x1 = x[(i, 1)];
        3.0 * x0 - 2.0 * x1 + (i as f64 * 0.31).sin() * 0.3
    });
    (x, y)
}

// ---------------------------------------------------------------------------
// Classification tests
// ---------------------------------------------------------------------------

#[test]
fn test_rf_clf_accuracy() {
    let (x, y) = make_binary_clf();
    let mut rf = RandomForestModel::new(10, 10, 2, 1, 42);
    rf.fit_clf(&x, &y, None).unwrap();

    let preds = rf.predict_clf(&x);
    let correct: usize = preds.iter().zip(&y).filter(|&(&p, &t)| p == t).count();
    let acc = correct as f64 / y.len() as f64;
    assert!(acc > 0.90, "RF clf accuracy = {acc:.3}, expected > 0.90");
}

#[test]
fn test_rf_clf_multiclass() {
    let (x, y) = make_multiclass_clf();
    let mut rf = RandomForestModel::new(20, 10, 2, 1, 42);
    rf.fit_clf(&x, &y, None).unwrap();

    let preds = rf.predict_clf(&x);
    let correct: usize = preds.iter().zip(&y).filter(|&(&p, &t)| p == t).count();
    let acc = correct as f64 / y.len() as f64;
    assert!(acc > 0.90, "RF multiclass accuracy = {acc:.3}, expected > 0.90");
}

#[test]
fn test_rf_clf_proba_valid() {
    let (x, y) = make_binary_clf();
    let mut rf = RandomForestModel::new(10, 10, 2, 1, 42);
    rf.fit_clf(&x, &y, None).unwrap();

    let proba = rf.predict_proba(&x);
    assert_eq!(proba.nrows(), x.nrows());
    assert_eq!(proba.ncols(), 2);

    for i in 0..proba.nrows() {
        let row_sum: f64 = (0..proba.ncols()).map(|j| proba[(i, j)]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-10,
            "Row {i} proba sum = {row_sum}"
        );
        for j in 0..proba.ncols() {
            assert!(proba[(i, j)] >= 0.0);
            assert!(proba[(i, j)] <= 1.0);
        }
    }
}

#[test]
fn test_rf_clf_multiclass_proba() {
    let (x, y) = make_multiclass_clf();
    let mut rf = RandomForestModel::new(10, 10, 2, 1, 42);
    rf.fit_clf(&x, &y, None).unwrap();

    let proba = rf.predict_proba(&x);
    assert_eq!(proba.ncols(), 3);

    for i in 0..proba.nrows() {
        let row_sum: f64 = (0..3).map(|j| proba[(i, j)]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-10,
            "Row {i} proba sum = {row_sum}"
        );
    }
}

// ---------------------------------------------------------------------------
// Regression tests
// ---------------------------------------------------------------------------

#[test]
fn test_rf_reg_mse() {
    let (x, y) = make_regression();
    let mut rf = RandomForestModel::new(10, 10, 2, 1, 42);
    rf.fit_reg(&x, &y, None).unwrap();

    let preds = rf.predict_reg(&x);
    let mse: f64 = (0..y.len())
        .map(|i| (preds[i] - y[i]).powi(2))
        .sum::<f64>()
        / y.len() as f64;

    // Variance of y
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let y_var: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / y.len() as f64;

    assert!(
        mse < y_var * 0.3,
        "RF reg MSE = {mse:.4}, y_var = {y_var:.4}, expected MSE < 30% of variance"
    );
}

// ---------------------------------------------------------------------------
// Feature importances
// ---------------------------------------------------------------------------

#[test]
fn test_rf_clf_importances_sum_to_one() {
    let (x, y) = make_binary_clf();
    let mut rf = RandomForestModel::new(10, 10, 2, 1, 42);
    rf.fit_clf(&x, &y, None).unwrap();

    let s: f64 = rf.feature_importances.iter().sum();
    assert!(
        (s - 1.0).abs() < 1e-10,
        "Importances sum = {s}, expected 1.0"
    );
    assert_eq!(rf.feature_importances.len(), 4);
}

#[test]
fn test_rf_reg_importances_sum_to_one() {
    let (x, y) = make_regression();
    let mut rf = RandomForestModel::new(10, 10, 2, 1, 42);
    rf.fit_reg(&x, &y, None).unwrap();

    let s: f64 = rf.feature_importances.iter().sum();
    assert!(
        (s - 1.0).abs() < 1e-10,
        "Importances sum = {s}, expected 1.0"
    );
    assert_eq!(rf.feature_importances.len(), 4);
}

#[test]
fn test_rf_clf_informative_feature_dominant() {
    let (x, y) = make_binary_clf();
    let mut rf = RandomForestModel::new(20, 10, 2, 1, 42);
    rf.fit_clf(&x, &y, None).unwrap();

    // Feature 0 is the informative feature (separates classes)
    let f0 = rf.feature_importances[0];
    let max_other = rf.feature_importances[1..]
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);
    assert!(
        f0 > max_other,
        "Feature 0 importance ({f0:.3}) should be highest but max other = {max_other:.3}"
    );
}

// ---------------------------------------------------------------------------
// OOB score
// ---------------------------------------------------------------------------

#[test]
fn test_rf_clf_oob_score() {
    let (x, y) = make_binary_clf();
    let mut rf = RandomForestModel::new(50, 10, 2, 1, 42);
    rf.fit_clf(&x, &y, None).unwrap();

    let oob = rf.oob_score.expect("OOB score should be computed");
    assert!(oob > 0.80, "OOB accuracy = {oob:.3}, expected > 0.80");
    assert!(oob <= 1.0);
}

#[test]
fn test_rf_reg_oob_score() {
    let (x, y) = make_regression();
    let mut rf = RandomForestModel::new(50, 10, 2, 1, 42);
    rf.fit_reg(&x, &y, None).unwrap();

    let oob = rf.oob_score.expect("OOB R² should be computed");
    assert!(oob > 0.30, "OOB R² = {oob:.3}, expected > 0.30");
    assert!(oob <= 1.0);
}

// ---------------------------------------------------------------------------
// Reproducibility
// ---------------------------------------------------------------------------

#[test]
fn test_rf_clf_reproducibility() {
    let (x, y) = make_binary_clf();

    let mut rf1 = RandomForestModel::new(10, 10, 2, 1, 42);
    rf1.fit_clf(&x, &y, None).unwrap();
    let p1 = rf1.predict_clf(&x);

    let mut rf2 = RandomForestModel::new(10, 10, 2, 1, 42);
    rf2.fit_clf(&x, &y, None).unwrap();
    let p2 = rf2.predict_clf(&x);

    assert_eq!(p1, p2, "Same seed should produce identical predictions");
}

#[test]
fn test_rf_reg_reproducibility() {
    let (x, y) = make_regression();

    let mut rf1 = RandomForestModel::new(10, 10, 2, 1, 42);
    rf1.fit_reg(&x, &y, None).unwrap();
    let p1 = rf1.predict_reg(&x);

    let mut rf2 = RandomForestModel::new(10, 10, 2, 1, 42);
    rf2.fit_reg(&x, &y, None).unwrap();
    let p2 = rf2.predict_reg(&x);

    for i in 0..p1.len() {
        assert!(
            (p1[i] - p2[i]).abs() < 1e-10,
            "Predictions differ at {i}: {} vs {}",
            p1[i],
            p2[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

#[test]
fn test_rf_n_features_and_n_classes() {
    let (x, y) = make_multiclass_clf();
    let mut rf = RandomForestModel::new(10, 10, 2, 1, 42);
    rf.fit_clf(&x, &y, None).unwrap();

    assert_eq!(rf.n_features, 3);
    assert_eq!(rf.n_classes, 3);
}

#[test]
fn test_rf_different_seeds_differ() {
    let (x, y) = make_binary_clf();

    let mut rf1 = RandomForestModel::new(10, 10, 2, 1, 42);
    rf1.fit_clf(&x, &y, None).unwrap();

    let mut rf2 = RandomForestModel::new(10, 10, 2, 1, 99);
    rf2.fit_clf(&x, &y, None).unwrap();

    // Importances should differ (different bootstrap samples)
    let imp_diff: f64 = rf1
        .feature_importances
        .iter()
        .zip(&rf2.feature_importances)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        imp_diff > 1e-6,
        "Different seeds should produce different importances"
    );
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn test_rf_empty_data_error() {
    let x = DMatrix::<f64>::zeros(0, 3);
    let y: Vec<i64> = vec![];

    let mut rf = RandomForestModel::new(10, 10, 2, 1, 42);
    assert!(rf.fit_clf(&x, &y, None).is_err());
}

#[test]
fn test_rf_dimension_mismatch_error() {
    let x = DMatrix::from_fn(100, 3, |i, j| (i + j) as f64);
    let y: Vec<i64> = vec![0; 50]; // wrong length

    let mut rf = RandomForestModel::new(10, 10, 2, 1, 42);
    assert!(rf.fit_clf(&x, &y, None).is_err());
}

#[test]
fn test_rf_reg_empty_data_error() {
    let x = DMatrix::<f64>::zeros(0, 3);
    let y = DVector::<f64>::zeros(0);

    let mut rf = RandomForestModel::new(10, 10, 2, 1, 42);
    assert!(rf.fit_reg(&x, &y, None).is_err());
}

// ---------------------------------------------------------------------------
// Histogram threshold propagation
// ---------------------------------------------------------------------------

#[test]
fn test_rf_histogram_threshold_propagation() {
    let (x, y) = make_binary_clf();
    let mut rf = RandomForestModel::new(10, 10, 2, 1, 42);
    rf.histogram_threshold = 50; // Low threshold to trigger histogram in trees
    rf.fit_clf(&x, &y, None).unwrap();

    let preds = rf.predict_clf(&x);
    let correct: usize = preds.iter().zip(&y).filter(|&(&p, &t)| p == t).count();
    let acc = correct as f64 / y.len() as f64;
    assert!(acc > 0.85, "RF with histogram threshold accuracy = {acc:.3}");
}

// ---------------------------------------------------------------------------
// RF regression: fit_reg_prepared uses max_features=p (not p/3)
// ---------------------------------------------------------------------------

#[test]
fn test_rf_reg_prepared_default_max_features() {
    // Council C3: regression test to prevent re-introduction of p/3 default.
    // With 5 features, p/3=2 produces significantly worse R² than p=5.
    let n = 200;
    let p = 5;
    let x = DMatrix::from_fn(n, p, |i, j| {
        (i as f64 * (0.13 + j as f64 * 0.07)).sin() * 3.0
    });
    let y = DVector::from_fn(n, |i, _| {
        // Strong signal from ALL 5 features
        x[(i, 0)] * 3.0 + x[(i, 1)] * 2.5 + x[(i, 2)] * 2.0
            + x[(i, 3)] * 1.5 + x[(i, 4)] * 1.0
            + (i as f64 * 0.31).sin() * 0.3
    });

    let mut rf = RandomForestModel::new(50, 500, 2, 1, 42);
    // Use prepared path (same as PyO3 binding)
    let cm = ml::cart::ColMajorMatrix::from_dmatrix(&x);
    rf.fit_reg_prepared(&cm, &y, None).unwrap();
    let preds = rf.predict_reg(&x);

    // Compute R²
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().enumerate().map(|(i, &yi)| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = y.iter().enumerate().map(|(i, &yi)| (yi - preds[i]).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;

    // With max_features=5 (all): R² > 0.95 on training data
    // With max_features=2 (p/3): R² would be ~0.80-0.85
    assert!(r2 > 0.90, "RF reg prepared R²={r2:.3} — max_features default may be wrong (p/3 instead of p)");
}
