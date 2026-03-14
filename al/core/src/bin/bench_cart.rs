//! Benchmark binary: ml CART vs sklearn baseline.
//!
//! Generates deterministic data via integer arithmetic (identical to bench_cart.py),
//! times fit + predict, prints JSON for the Python comparison script.
//!
//! Usage: bench_cart <n_samples>
//!   n_samples defaults to 10000

use ml::cart::DecisionTreeModel;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;

/// Deterministic feature matrix: X[i, j] = ((i * 7 + j * 13) % 100) as f64 / 100.0
/// Identical formula used in bench_cart.py — integer ops, exact same f64 result.
fn make_x(n: usize, p: usize) -> DMatrix<f64> {
    DMatrix::from_fn(n, p, |i, j| ((i * 7 + j * 13) % 100) as f64 / 100.0)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);
    let p = 5;

    let x = make_x(n, p);

    // Classification: y = 1 if X[:,0] + X[:,1] > X[:,2], else 0
    let y_clf: Vec<i64> = (0..n)
        .map(|i| {
            if x[(i, 0)] + x[(i, 1)] > x[(i, 2)] {
                1
            } else {
                0
            }
        })
        .collect();

    // Regression: y = 2*X[:,0] + X[:,1] - 0.5*X[:,2]
    let y_reg = DVector::from_fn(n, |i, _| 2.0 * x[(i, 0)] + x[(i, 1)] - 0.5 * x[(i, 2)]);

    let train_n = n * 8 / 10;
    let x_train = x.rows(0, train_n).into_owned();
    let x_test = x.rows(train_n, n - train_n).into_owned();
    let y_clf_train = &y_clf[..train_n];
    let y_clf_test = &y_clf[train_n..];
    let y_reg_train = y_reg.rows(0, train_n).into_owned();
    let y_reg_test = y_reg.rows(train_n, n - train_n).into_owned();

    // --- Classification ---
    let mut clf = DecisionTreeModel::default();
    let t0 = Instant::now();
    clf.fit_clf(&x_train, y_clf_train, None, ml::cart::Criterion::Gini).unwrap();
    let fit_clf_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let preds_clf = clf.predict_clf(&x_test);
    let predict_clf_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let n_correct: usize = preds_clf
        .iter()
        .zip(y_clf_test)
        .filter(|(a, b)| **a == **b)
        .count();
    let accuracy = n_correct as f64 / y_clf_test.len() as f64;

    // --- Regression ---
    let mut reg = DecisionTreeModel::default();
    let t2 = Instant::now();
    reg.fit_reg(&x_train, &y_reg_train, None, ml::cart::Criterion::MSE).unwrap();
    let fit_reg_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = Instant::now();
    let preds_reg = reg.predict_reg(&x_test);
    let predict_reg_ms = t3.elapsed().as_secs_f64() * 1000.0;

    let mse: f64 = preds_reg
        .iter()
        .zip(y_reg_test.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f64>()
        / y_reg_test.len() as f64;

    // Emit JSON for bench_cart.py to parse
    println!("{{");
    println!("  \"n\": {n},");
    println!("  \"fit_clf_ms\": {fit_clf_ms:.4},");
    println!("  \"predict_clf_ms\": {predict_clf_ms:.4},");
    println!("  \"accuracy\": {accuracy:.6},");
    println!("  \"fit_reg_ms\": {fit_reg_ms:.4},");
    println!("  \"predict_reg_ms\": {predict_reg_ms:.4},");
    println!("  \"mse\": {mse:.6}");
    println!("}}");
}
