//! Criterion benchmarks for SVM classifier (linear SMO).
//!
//! SVM uses flat row-major `&[f64]` for x, NOT `DMatrix`.
//! SVM at n=5K is the upper size since SMO is O(n^2) per iteration.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ml::svm::SvmClassifier;

// ---------------------------------------------------------------------------
// Deterministic LCG data generator (no rand dependency)
// ---------------------------------------------------------------------------

fn lcg_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

fn make_clf_data_flat(n: usize, p: usize, seed: u64) -> (Vec<f64>, Vec<i64>) {
    let mut state = seed;
    let x: Vec<f64> = (0..n * p).map(|_| lcg_f64(&mut state)).collect();
    let y: Vec<i64> = (0..n).map(|i| if x[i * p] > 0.5 { 1 } else { 0 }).collect();
    (x, y)
}

fn make_multiclass_data_flat(n: usize, p: usize, k: usize, seed: u64) -> (Vec<f64>, Vec<i64>) {
    let mut state = seed;
    let x: Vec<f64> = (0..n * p).map(|_| lcg_f64(&mut state)).collect();
    let y: Vec<i64> = (0..n).map(|i| (i % k) as i64).collect();
    (x, y)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_svm_fit_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("svm_fit_binary");
    for &n in &[1_000, 5_000] {
        let p = 20;
        let (x, y) = make_clf_data_flat(n, p, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut clf = SvmClassifier::new(1.0, 1e-3, 200);
                clf.fit(&x, &y, n, p, None);
            });
        });
    }
    group.finish();
}

fn bench_svm_fit_multiclass(c: &mut Criterion) {
    let mut group = c.benchmark_group("svm_fit_multiclass");
    for &n in &[1_000, 5_000] {
        let p = 20;
        let (x, y) = make_multiclass_data_flat(n, p, 5, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut clf = SvmClassifier::new(1.0, 1e-3, 200);
                clf.fit(&x, &y, n, p, None);
            });
        });
    }
    group.finish();
}

fn bench_svm_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("svm_predict");
    for &n in &[1_000, 5_000] {
        let p = 20;
        let (x, y) = make_clf_data_flat(n, p, 42);
        let mut clf = SvmClassifier::new(1.0, 1e-3, 200);
        clf.fit(&x, &y, n, p, None);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                clf.predict(&x, n, p);
            });
        });
    }
    group.finish();
}

fn bench_svm_predict_proba(c: &mut Criterion) {
    let mut group = c.benchmark_group("svm_predict_proba");
    for &n in &[1_000, 5_000] {
        let p = 20;
        let (x, y) = make_clf_data_flat(n, p, 42);
        let mut clf = SvmClassifier::new(1.0, 1e-3, 200);
        clf.fit(&x, &y, n, p, None);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                clf.predict_proba(&x, n, p);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_svm_fit_binary,
    bench_svm_fit_multiclass,
    bench_svm_predict,
    bench_svm_predict_proba,
);
criterion_main!(benches);
