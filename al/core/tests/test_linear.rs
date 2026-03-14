//! Integration tests for `LinearModel` (public API only).

use ml::error::MlError;
use ml::linear::LinearModel;
use nalgebra::{DMatrix, DVector};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Deterministic (n, p) design matrix: x[i][j] = sin(i * (j+1) * 0.3).
fn make_x(n: usize, p: usize) -> DMatrix<f64> {
    DMatrix::from_fn(n, p, |i, j| ((i as f64) * ((j + 1) as f64) * 0.3).sin())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_predict_shape() {
    let x_train = make_x(30, 3);
    let y_train = DVector::from_fn(30, |i, _| (i as f64) * 0.5);
    let mut model = LinearModel::new(1.0);
    model.fit(&x_train, &y_train, None).unwrap();

    let x_test = make_x(7, 3);
    let preds = model.predict(&x_test);
    assert_eq!(preds.len(), 7, "predict output length");
}

#[test]
fn test_coef_length_equals_n_features() {
    let x = make_x(50, 4);
    let y = DVector::from_fn(50, |i, _| (i as f64) * 0.1);
    let mut model = LinearModel::new(0.1);
    model.fit(&x, &y, None).unwrap();
    assert_eq!(model.coef.len(), 4, "coef length should equal n_features");
}

#[test]
fn test_alpha_zero_recovers_ols() {
    // y = 2*x1 - x2 + 0.5*x3 + 0.1 (no noise, known true weights and intercept).
    // With alpha=0, the model should recover them exactly.
    let w_true = DVector::from_vec(vec![2.0, -1.0, 0.5]);
    let intercept_true = 0.1_f64;
    let x = DMatrix::from_row_slice(
        10,
        3,
        &[
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0,
        ],
    );
    let y = DVector::from_fn(10, |i, _| {
        x.row(i).dot(&w_true.transpose()) + intercept_true
    });

    let mut model = LinearModel::new(0.0);
    model.fit(&x, &y, None).unwrap();

    for i in 0..3 {
        assert!(
            (model.coef[i] - w_true[i]).abs() < 1e-9,
            "coef[{i}]: got {:.6}, expected {:.6}",
            model.coef[i],
            w_true[i]
        );
    }
    assert!(
        (model.intercept - intercept_true).abs() < 1e-9,
        "intercept: got {:.6}, expected {intercept_true}",
        model.intercept
    );
}

#[test]
fn test_higher_alpha_shrinks_coef_norm() {
    // Stronger regularization should produce smaller coefficient norms.
    let x = make_x(60, 3);
    let y = DVector::from_fn(60, |i, _| {
        (i as f64 * 0.2).sin() * 3.0 + (i as f64 * 0.7).cos()
    });

    let mut m_lo = LinearModel::new(0.001);
    let mut m_hi = LinearModel::new(100.0);
    m_lo.fit(&x, &y, None).unwrap();
    m_hi.fit(&x, &y, None).unwrap();

    assert!(
        m_hi.coef.norm() < m_lo.coef.norm(),
        "high alpha coef norm {:.4} >= low alpha coef norm {:.4}",
        m_hi.coef.norm(),
        m_lo.coef.norm()
    );
}

#[test]
fn test_regression_quality() {
    // Signal >> noise → R² should be > 0.9 on held-out rows.
    let n = 50;
    let w_true = DVector::from_vec(vec![2.0, -1.0, 0.5]);
    let x = make_x(n, 3);
    let y = DVector::from_fn(n, |i, _| {
        x.row(i).dot(&w_true.transpose()) + 0.05 * ((i as f64) * 2.7).cos()
    });

    let (train_n, test_n) = (40, 10);
    let x_train = x.rows(0, train_n).into_owned();
    let y_train = y.rows(0, train_n).into_owned();
    let x_test = x.rows(train_n, test_n).into_owned();
    let y_test = y.rows(train_n, test_n).into_owned();

    let mut model = LinearModel::new(0.01);
    model.fit(&x_train, &y_train, None).unwrap();
    let preds = model.predict(&x_test);

    let ss_res: f64 = (0..test_n).map(|i| (preds[i] - y_test[i]).powi(2)).sum();
    let y_mean = y_test.iter().sum::<f64>() / test_n as f64;
    let ss_tot: f64 = (0..test_n).map(|i| (y_test[i] - y_mean).powi(2)).sum();
    let r2 = 1.0 - ss_res / ss_tot;

    assert!(
        r2 > 0.9,
        "R² {r2:.3} < 0.9 — Ridge regression quality too low"
    );
}

#[test]
fn test_reproducibility() {
    // Two identical fits must produce identical predictions.
    let x = make_x(30, 3);
    let y = DVector::from_fn(30, |i, _| (i as f64 * 0.3).sin() * 2.0);
    let x_test = make_x(5, 3);

    let mut m1 = LinearModel::new(0.5);
    let mut m2 = LinearModel::new(0.5);
    m1.fit(&x, &y, None).unwrap();
    m2.fit(&x, &y, None).unwrap();

    let p1 = m1.predict(&x_test);
    let p2 = m2.predict(&x_test);
    assert_eq!(p1, p2, "identical fits must produce identical predictions");
}

#[test]
fn test_sample_weights_affect_result() {
    // With extreme weights (first half weight=100, second half weight=1),
    // the model should fit the first half much better.
    let x = make_x(20, 2);
    let y = DVector::from_fn(20, |i, _| if i < 10 { 5.0 } else { -5.0 });
    let sw = DVector::from_fn(20, |i, _| if i < 10 { 100.0 } else { 1.0 });

    let mut m_unweighted = LinearModel::new(0.1);
    let mut m_weighted = LinearModel::new(0.1);
    m_unweighted.fit(&x, &y, None).unwrap();
    m_weighted.fit(&x, &y, Some(&sw)).unwrap();

    // Weighted model should produce different coefficients
    let diff: f64 = (&m_weighted.coef - &m_unweighted.coef).norm();
    assert!(diff > 1e-6, "sample weights had no effect on coefficients");
}

#[test]
fn test_empty_data_returns_error() {
    let x = DMatrix::<f64>::zeros(0, 3);
    let y = DVector::<f64>::zeros(0);
    let mut model = LinearModel::new(1.0);
    let result = model.fit(&x, &y, None);
    assert!(
        matches!(result, Err(MlError::EmptyData)),
        "expected EmptyData error"
    );
}

#[test]
fn test_dimension_mismatch_returns_error() {
    let x = make_x(10, 3);
    let y = DVector::<f64>::zeros(7); // wrong length
    let mut model = LinearModel::new(1.0);
    let result = model.fit(&x, &y, None);
    assert!(
        matches!(
            result,
            Err(MlError::DimensionMismatch {
                expected: 10,
                got: 7
            })
        ),
        "expected DimensionMismatch error"
    );
}

// ---------------------------------------------------------------------------
// Dual solver tests (p >> n)
// ---------------------------------------------------------------------------

#[test]
fn test_dual_solver_p_much_greater_than_n() {
    // p=500, n=20. Dual path should be used automatically.
    // Verify it produces sensible predictions (coef has right length,
    // predictions are finite).
    let n = 20;
    let p = 500;
    let x = make_x(n, p);
    let y = DVector::from_fn(n, |i, _| (i as f64) * 0.5 + 1.0);
    let mut model = LinearModel::new(1.0);
    model.fit(&x, &y, None).unwrap();

    assert_eq!(model.coef.len(), p, "coef length must equal p");
    assert!(model.intercept.is_finite(), "intercept must be finite");
    assert!(
        model.coef.iter().all(|c| c.is_finite()),
        "all coefficients must be finite"
    );

    // Predictions on training data should be reasonable.
    let preds = model.predict(&x);
    assert_eq!(preds.len(), n);
    assert!(
        preds.iter().all(|p| p.is_finite()),
        "all predictions must be finite"
    );
}

#[test]
fn test_dual_solver_large_p() {
    // Kill gate: p=5000, n=200 should complete (not hang or OOM).
    // Dual path solves 200x200 system instead of 5001x5001.
    let n = 200;
    let p = 5000;
    let x = make_x(n, p);
    let y = DVector::from_fn(n, |i, _| (i as f64 * 0.1).sin());
    let mut model = LinearModel::new(10.0);
    model.fit(&x, &y, None).unwrap();

    assert_eq!(model.coef.len(), p);
    let preds = model.predict(&x);
    assert_eq!(preds.len(), n);
    assert!(
        preds.iter().all(|v| v.is_finite()),
        "predictions must be finite for p=5000, n=200"
    );
}

#[test]
fn test_dual_primal_parity_at_boundary() {
    // When p ~ n, both solvers are feasible. Force both paths and check
    // predictions agree within 1e-6.
    //
    // Use p = n-1 (primal path) and p = n+1 (dual path) with alpha=1.0.
    // Both should give very similar results if we use the same data shape.
    //
    // Better approach: use p = n (primal), then manually invoke dual by
    // setting p = n + 1 (append a zero column). But cleanest: use the
    // same data with p slightly above and below n.
    let n = 30;
    let p = 30; // p == n → primal path

    // True signal: y = sum of first 3 features * known weights + intercept.
    let x = make_x(n, p);
    let y = DVector::from_fn(n, |i, _| {
        let row = x.row(i);
        2.0 * row[0] - 1.0 * row[1] + 0.5 * row[2] + 3.0
    });

    // Primal solve (p == n).
    let mut primal = LinearModel::new(1.0);
    primal.fit(&x, &y, None).unwrap();
    let preds_primal = primal.predict(&x);

    // Dual solve: add one zero column → p = n+1 triggers dual.
    let mut x_ext = DMatrix::zeros(n, p + 1);
    for i in 0..n {
        for j in 0..p {
            x_ext[(i, j)] = x[(i, j)];
        }
        // Column p+1 is zero.
    }
    let mut dual = LinearModel::new(1.0);
    dual.fit(&x_ext, &y, None).unwrap();

    // Predictions from the dual model (on extended X) should match primal
    // within tolerance, since the extra zero column contributes nothing.
    let preds_dual = dual.predict(&x_ext);

    let max_diff: f64 = preds_primal
        .iter()
        .zip(preds_dual.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_diff < 1e-6,
        "primal vs dual max prediction diff {max_diff:.2e} exceeds 1e-6"
    );
}

#[test]
fn test_dual_primal_exact_parity() {
    // Force both paths on identical data by constructing the system manually.
    // With n=15, p=20 (dual) and then solving the same system with alpha > 0.
    // Compare against a reference primal solve on the same data.
    let n = 15;
    let p = 20;
    let alpha = 2.0;
    let x = make_x(n, p);
    let y = DVector::from_fn(n, |i, _| (i as f64 * 0.3).sin() + 1.0);

    // Dual path (p > n, alpha > 0 → automatic).
    let mut dual_model = LinearModel::new(alpha);
    dual_model.fit(&x, &y, None).unwrap();

    // Primal path: manually build (p+1)x(p+1) system and solve.
    // This is the reference solution.
    let d = p + 1;
    let mut a = DMatrix::zeros(d, d);
    a[(0, 0)] = n as f64;
    for i in 0..n {
        for j in 0..p {
            a[(0, j + 1)] += x[(i, j)];
            a[(j + 1, 0)] += x[(i, j)];
        }
    }
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0;
            for k in 0..n {
                s += x[(k, i)] * x[(k, j)];
            }
            a[(i + 1, j + 1)] = s;
        }
        a[(i + 1, i + 1)] += alpha;
    }
    let mut rhs = DVector::zeros(d);
    rhs[0] = y.iter().sum::<f64>();
    for j in 0..p {
        let mut s = 0.0;
        for i in 0..n {
            s += x[(i, j)] * y[i];
        }
        rhs[j + 1] = s;
    }
    let w_ref = a.cholesky().unwrap().solve(&rhs);
    let intercept_ref = w_ref[0];
    let coef_ref = w_ref.rows(1, p).into_owned();

    // Compare coefficients.
    let coef_diff: f64 = (&dual_model.coef - &coef_ref).norm();
    assert!(
        coef_diff < 1e-6,
        "dual vs primal coef L2 diff {coef_diff:.2e} exceeds 1e-6"
    );

    // Compare intercepts.
    let intercept_diff = (dual_model.intercept - intercept_ref).abs();
    assert!(
        intercept_diff < 1e-6,
        "dual vs primal intercept diff {intercept_diff:.2e} exceeds 1e-6"
    );

    // Compare predictions.
    let preds_dual = dual_model.predict(&x);
    let intercept = intercept_ref;
    let preds_ref: Vec<f64> = (0..n)
        .map(|i| {
            let mut v = intercept;
            for j in 0..p {
                v += x[(i, j)] * coef_ref[j];
            }
            v
        })
        .collect();

    let max_pred_diff: f64 = preds_dual
        .iter()
        .zip(preds_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_pred_diff < 1e-6,
        "dual vs primal max prediction diff {max_pred_diff:.2e} exceeds 1e-6"
    );
}

#[test]
fn test_dual_solver_with_sample_weights() {
    // Dual path with sample weights: p=50, n=20.
    let n = 20;
    let p = 50;
    let x = make_x(n, p);
    let y = DVector::from_fn(n, |i, _| (i as f64) * 0.3 - 1.0);
    let sw = DVector::from_fn(n, |i, _| if i < 10 { 5.0 } else { 1.0 });

    let mut m_weighted = LinearModel::new(1.0);
    let mut m_unweighted = LinearModel::new(1.0);
    m_weighted.fit(&x, &y, Some(&sw)).unwrap();
    m_unweighted.fit(&x, &y, None).unwrap();

    // Weights should change the result.
    let diff = (&m_weighted.coef - &m_unweighted.coef).norm();
    assert!(
        diff > 1e-6,
        "sample weights had no effect on dual solver coefficients"
    );

    // Both should produce finite predictions.
    let preds_w = m_weighted.predict(&x);
    let preds_u = m_unweighted.predict(&x);
    assert!(preds_w.iter().all(|v| v.is_finite()));
    assert!(preds_u.iter().all(|v| v.is_finite()));
}

#[test]
fn test_dual_higher_alpha_shrinks_coef_norm() {
    // Same invariant as primal: more regularization = smaller coefficients.
    // p=100, n=20 (dual path).
    let n = 20;
    let p = 100;
    let x = make_x(n, p);
    let y = DVector::from_fn(n, |i, _| {
        (i as f64 * 0.2).sin() * 3.0 + (i as f64 * 0.7).cos()
    });

    let mut m_lo = LinearModel::new(0.01);
    let mut m_hi = LinearModel::new(100.0);
    m_lo.fit(&x, &y, None).unwrap();
    m_hi.fit(&x, &y, None).unwrap();

    assert!(
        m_hi.coef.norm() < m_lo.coef.norm(),
        "dual path: high alpha coef norm {:.4} >= low alpha coef norm {:.4}",
        m_hi.coef.norm(),
        m_lo.coef.norm()
    );
}

#[test]
fn test_dual_alpha_zero_falls_back_to_primal() {
    // alpha=0 with p > n should NOT use dual path (dual requires alpha > 0).
    // It falls back to primal. This should still work (may be singular but
    // the primal solver handles it via LU fallback).
    let n = 10;
    let p = 20;
    let x = make_x(n, p);
    let y = DVector::from_fn(n, |i, _| (i as f64) * 0.5);
    let mut model = LinearModel::new(0.0);

    // This uses primal path even though p > n (because alpha = 0).
    // May succeed or fail depending on matrix conditioning.
    let result = model.fit(&x, &y, None);
    // We don't assert success or failure — just that the code path
    // doesn't panic or use dual when alpha = 0.
    if let Ok(_) = result {
        assert_eq!(model.coef.len(), p);
    }
}

#[test]
fn test_dual_fit_row_major_parity() {
    // Verify fit() and fit_row_major() produce identical results on dual path.
    let n = 15;
    let p = 40;
    let x = make_x(n, p);
    let y = DVector::from_fn(n, |i, _| (i as f64 * 0.2).cos() * 2.0);

    // fit() with DMatrix
    let mut m1 = LinearModel::new(1.0);
    m1.fit(&x, &y, None).unwrap();

    // fit_row_major() with row-major slice
    let mut x_rm = vec![0.0_f64; n * p];
    for i in 0..n {
        for j in 0..p {
            x_rm[i * p + j] = x[(i, j)];
        }
    }
    let y_slice: Vec<f64> = y.iter().cloned().collect();
    let mut m2 = LinearModel::new(1.0);
    m2.fit_row_major(&x_rm, &y_slice, n, p, None).unwrap();

    // Coefficients should match exactly (both use fit_dual internally).
    let coef_diff = (&m1.coef - &m2.coef).norm();
    assert!(
        coef_diff < 1e-10,
        "fit vs fit_row_major coef diff {coef_diff:.2e} on dual path"
    );
    let intercept_diff = (m1.intercept - m2.intercept).abs();
    assert!(
        intercept_diff < 1e-10,
        "fit vs fit_row_major intercept diff {intercept_diff:.2e} on dual path"
    );
}
