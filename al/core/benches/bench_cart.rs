//! Criterion benchmarks for CART decision tree (classification + regression).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ml::cart::{Criterion as SplitCriterion, DecisionTreeModel};
use nalgebra::{DMatrix, DVector};

// ---------------------------------------------------------------------------
// Deterministic LCG data generator (no rand dependency)
// ---------------------------------------------------------------------------

fn lcg_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

fn make_clf_data(n: usize, p: usize, seed: u64) -> (DMatrix<f64>, Vec<i64>) {
    let mut state = seed;
    let data: Vec<f64> = (0..n * p).map(|_| lcg_f64(&mut state)).collect();
    let x = DMatrix::from_row_slice(n, p, &data);
    let y: Vec<i64> = (0..n).map(|i| if data[i * p] > 0.5 { 1 } else { 0 }).collect();
    (x, y)
}

fn make_reg_data(n: usize, p: usize, seed: u64) -> (DMatrix<f64>, DVector<f64>) {
    let mut state = seed;
    let data: Vec<f64> = (0..n * p).map(|_| lcg_f64(&mut state)).collect();
    let x = DMatrix::from_row_slice(n, p, &data);
    let y = DVector::from_iterator(n, (0..n).map(|i| data[i * p] + 2.0 * data[i * p + 1]));
    (x, y)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_cart_fit_clf(c: &mut Criterion) {
    let mut group = c.benchmark_group("cart_fit_clf");
    for &n in &[1_000, 10_000] {
        let (x, y) = make_clf_data(n, 20, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut tree = DecisionTreeModel::new(10, 2, 1);
                tree.fit_clf(&x, &y, None, SplitCriterion::Gini).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_cart_fit_reg(c: &mut Criterion) {
    let mut group = c.benchmark_group("cart_fit_reg");
    for &n in &[1_000, 10_000] {
        let (x, y) = make_reg_data(n, 20, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut tree = DecisionTreeModel::new(10, 2, 1);
                tree.fit_reg(&x, &y, None, SplitCriterion::MSE).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_cart_predict_clf(c: &mut Criterion) {
    let mut group = c.benchmark_group("cart_predict_clf");
    for &n in &[1_000, 10_000] {
        let (x, y) = make_clf_data(n, 20, 42);
        let mut tree = DecisionTreeModel::new(10, 2, 1);
        tree.fit_clf(&x, &y, None, SplitCriterion::Gini).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                tree.predict_clf(&x);
            });
        });
    }
    group.finish();
}

fn bench_cart_predict_reg(c: &mut Criterion) {
    let mut group = c.benchmark_group("cart_predict_reg");
    for &n in &[1_000, 10_000] {
        let (x, y) = make_reg_data(n, 20, 42);
        let mut tree = DecisionTreeModel::new(10, 2, 1);
        tree.fit_reg(&x, &y, None, SplitCriterion::MSE).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                tree.predict_reg(&x);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cart_fit_clf,
    bench_cart_fit_reg,
    bench_cart_predict_clf,
    bench_cart_predict_reg,
);
criterion_main!(benches);
