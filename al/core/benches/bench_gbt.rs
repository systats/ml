//! Criterion benchmarks for Gradient-Boosted Trees (classification + regression).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ml::gbt::GBTModel;
use nalgebra::DMatrix;

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

fn make_multiclass_data(n: usize, p: usize, k: usize, seed: u64) -> (DMatrix<f64>, Vec<i64>) {
    let mut state = seed;
    let data: Vec<f64> = (0..n * p).map(|_| lcg_f64(&mut state)).collect();
    let x = DMatrix::from_row_slice(n, p, &data);
    let y: Vec<i64> = (0..n).map(|i| (i % k) as i64).collect();
    (x, y)
}

fn make_reg_data(n: usize, p: usize, seed: u64) -> (DMatrix<f64>, Vec<f64>) {
    let mut state = seed;
    let data: Vec<f64> = (0..n * p).map(|_| lcg_f64(&mut state)).collect();
    let x = DMatrix::from_row_slice(n, p, &data);
    let y: Vec<f64> = (0..n).map(|i| data[i * p] + 2.0 * data[i * p + 1]).collect();
    (x, y)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_gbt_fit_clf_binary(c: &mut Criterion) {
    let mut group = c.benchmark_group("gbt_fit_clf_binary");
    for &n in &[1_000, 10_000] {
        let (x, y) = make_clf_data(n, 20, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut model = GBTModel::new(50, 0.1, 5, 2, 1, 1.0, 42);
                model.fit_clf(&x, &y, None).unwrap();
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// GOSS vs no-GOSS: validates the goss_min_n gate and speedup at large n
// ---------------------------------------------------------------------------

fn bench_lgbm_goss_100k(c: &mut Criterion) {
    let mut group = c.benchmark_group("lgbm_goss");
    // n=100K, p=20, 200 trees, max_leaves=31 (lossguide = LightGBM-style)
    let (x, y) = make_clf_data(100_000, 20, 42);

    group.bench_function("goss_top20_other10", |b| {
        b.iter(|| {
            let mut model = GBTModel::new(200, 0.1, 6, 2, 1, 1.0, 42);
            model.grow_policy = ml::gbt::GrowPolicy::Lossguide;
            model.max_leaves = 31;
            model.goss_top_rate = 0.2;
            model.goss_other_rate = 0.1;
            model.goss_min_n = 0; // force GOSS always
            model.fit_clf(&x, &y, None).unwrap();
        });
    });

    group.bench_function("no_goss_subsample1", |b| {
        b.iter(|| {
            let mut model = GBTModel::new(200, 0.1, 6, 2, 1, 1.0, 42);
            model.grow_policy = ml::gbt::GrowPolicy::Lossguide;
            model.max_leaves = 31;
            // goss_top_rate defaults to 1.0 → GOSS disabled
            model.fit_clf(&x, &y, None).unwrap();
        });
    });

    group.finish();
}

fn bench_lgbm_goss_10k(c: &mut Criterion) {
    let mut group = c.benchmark_group("lgbm_goss_small");
    // n=10K: expect GOSS to be slower (sort overhead > histogram savings)
    let (x, y) = make_clf_data(10_000, 20, 42);

    group.bench_function("goss_top20_other10", |b| {
        b.iter(|| {
            let mut model = GBTModel::new(200, 0.1, 6, 2, 1, 1.0, 42);
            model.grow_policy = ml::gbt::GrowPolicy::Lossguide;
            model.max_leaves = 31;
            model.goss_top_rate = 0.2;
            model.goss_other_rate = 0.1;
            model.goss_min_n = 0;
            model.fit_clf(&x, &y, None).unwrap();
        });
    });

    group.bench_function("no_goss_subsample1", |b| {
        b.iter(|| {
            let mut model = GBTModel::new(200, 0.1, 6, 2, 1, 1.0, 42);
            model.grow_policy = ml::gbt::GrowPolicy::Lossguide;
            model.max_leaves = 31;
            model.fit_clf(&x, &y, None).unwrap();
        });
    });

    group.finish();
}

fn bench_gbt_fit_clf_multiclass(c: &mut Criterion) {
    let mut group = c.benchmark_group("gbt_fit_clf_multiclass");
    for &n in &[1_000, 10_000] {
        let (x, y) = make_multiclass_data(n, 20, 5, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut model = GBTModel::new(50, 0.1, 5, 2, 1, 1.0, 42);
                model.fit_clf(&x, &y, None).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_gbt_fit_reg(c: &mut Criterion) {
    let mut group = c.benchmark_group("gbt_fit_reg");
    for &n in &[1_000, 10_000] {
        let (x, y) = make_reg_data(n, 20, 42);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut model = GBTModel::new(50, 0.1, 5, 2, 1, 1.0, 42);
                model.fit_reg(&x, &y, None).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_gbt_predict_clf(c: &mut Criterion) {
    let mut group = c.benchmark_group("gbt_predict_clf");
    for &n in &[1_000, 10_000] {
        let (x, y) = make_clf_data(n, 20, 42);
        let mut model = GBTModel::new(50, 0.1, 5, 2, 1, 1.0, 42);
        model.fit_clf(&x, &y, None).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                model.predict_clf(&x);
            });
        });
    }
    group.finish();
}

fn bench_gbt_predict_reg(c: &mut Criterion) {
    let mut group = c.benchmark_group("gbt_predict_reg");
    for &n in &[1_000, 10_000] {
        let (x, y) = make_reg_data(n, 20, 42);
        let mut model = GBTModel::new(50, 0.1, 5, 2, 1, 1.0, 42);
        model.fit_reg(&x, &y, None).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                model.predict_reg(&x);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_gbt_fit_clf_binary,
    bench_gbt_fit_clf_multiclass,
    bench_gbt_fit_reg,
    bench_gbt_predict_clf,
    bench_gbt_predict_reg,
    bench_lgbm_goss_100k,
    bench_lgbm_goss_10k,
);
criterion_main!(benches);
