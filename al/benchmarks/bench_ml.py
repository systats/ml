#!/usr/bin/env python3
"""Benchmark: ml CART + RF vs sklearn.

5 representative datasets × 3 sizes.
CART: ml (sort, auto) vs sklearn DecisionTree.
RF:   ml RandomForest vs sklearn RandomForestClassifier/Regressor.

All data generated with numpy only (deterministic, no sklearn generators).
Timed via in-process PyO3 calls — zero subprocess overhead.
"""
import time

import numpy as np
from ml_py import DecisionTree, RandomForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ---------------------------------------------------------------------------
# Dataset generators (numpy-only, deterministic)
# ---------------------------------------------------------------------------

def make_noisy_xor(n: int, seed: int = 42):
    """XOR with noise: y = (x0 > 0) XOR (x1 > 0), 5 features, 20% label noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y_clean = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    flip = rng.rand(n) < 0.20
    y = y_clean.copy()
    y[flip] = 1 - y[flip]
    return X, y


def make_overlapping(n: int, seed: int = 42):
    """3-class, 10 features, 3 informative, overlapping clusters."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 10)
    centers = np.array([[2, 0, -1], [-1, 2, 1], [0, -2, 1]], dtype=np.float64)
    y = rng.randint(0, 3, size=n).astype(np.int64)
    for c in range(3):
        mask = y == c
        X[mask, :3] += centers[c]
    return X, y


def make_wide_sparse(n: int, seed: int = 42):
    """Binary, 50 features, 5 informative, rest pure noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 50)
    signal = X[:, 0] + 0.8 * X[:, 1] - 0.6 * X[:, 2] + 0.4 * X[:, 3] + 0.3 * X[:, 4]
    y = (signal > 0).astype(np.int64)
    flip = rng.rand(n) < 0.10
    y[flip] = 1 - y[flip]
    return X, y


def make_sinusoidal_reg(n: int, seed: int = 42):
    """y = sin(x0) + 0.5*cos(x1*x2) + noise, 8 features."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 8)
    y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1] * X[:, 2]) + 0.3 * rng.randn(n)
    return X, y


def make_friedman1(n: int, seed: int = 42):
    """Friedman #1: y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + noise.
    10 features (5 informative + 5 noise), uniform [0,1]."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 10)
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3]
         + 5 * X[:, 4]
         + rng.randn(n))
    return X, y


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

DATASETS_CLF = [
    ("D1 noisy_xor  ", make_noisy_xor,    [1_000, 10_000, 100_000]),
    ("D2 overlapping ", make_overlapping,   [1_000, 10_000, 100_000]),
    ("D3 wide_sparse ", make_wide_sparse,   [1_000, 10_000, 100_000]),
]

DATASETS_REG = [
    ("D4 sinusoidal  ", make_sinusoidal_reg, [1_000, 10_000, 100_000]),
    ("D5 friedman1   ", make_friedman1,      [1_000, 10_000, 100_000]),
]

WARMUP_RUNS = 1
TIMED_RUNS = 3

NO_HIST = 2**63  # force sort-based


def _time_cart_clf(X_tr, yc_tr, X_te, yc_te, engine="sklearn", hist_thresh=1024):
    """Time fit+predict for CART classification."""
    fits, preds_t = [], []
    last_preds = None
    runs = WARMUP_RUNS + TIMED_RUNS
    for r in range(runs):
        if engine == "sklearn":
            m = DecisionTreeClassifier(random_state=0)
            t0 = time.perf_counter()
            m.fit(X_tr, yc_tr)
            fit_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            last_preds = m.predict(X_te)
            pred_ms = (time.perf_counter() - t0) * 1000
        else:
            m = DecisionTree(histogram_threshold=hist_thresh)
            t0 = time.perf_counter()
            m.fit_clf(np.ascontiguousarray(X_tr), yc_tr)
            fit_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            last_preds = m.predict_clf(np.ascontiguousarray(X_te))
            pred_ms = (time.perf_counter() - t0) * 1000
        if r >= WARMUP_RUNS:
            fits.append(fit_ms)
            preds_t.append(pred_ms)
    acc = float((last_preds == yc_te).mean())
    return np.median(fits), np.median(preds_t), acc


def _time_cart_reg(X_tr, yr_tr, X_te, yr_te, engine="sklearn", hist_thresh=1024):
    """Time fit+predict for CART regression."""
    fits, preds_t = [], []
    last_preds = None
    runs = WARMUP_RUNS + TIMED_RUNS
    for r in range(runs):
        if engine == "sklearn":
            m = DecisionTreeRegressor(random_state=0)
            t0 = time.perf_counter()
            m.fit(X_tr, yr_tr)
            fit_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            last_preds = m.predict(X_te)
            pred_ms = (time.perf_counter() - t0) * 1000
        else:
            m = DecisionTree(histogram_threshold=hist_thresh)
            t0 = time.perf_counter()
            m.fit_reg(np.ascontiguousarray(X_tr), yr_tr)
            fit_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            last_preds = m.predict_reg(np.ascontiguousarray(X_te))
            pred_ms = (time.perf_counter() - t0) * 1000
        if r >= WARMUP_RUNS:
            fits.append(fit_ms)
            preds_t.append(pred_ms)
    mse_val = float(np.mean((last_preds - yr_te) ** 2))
    return np.median(fits), np.median(preds_t), mse_val


def _time_rf_clf(X_tr, yc_tr, X_te, yc_te, engine="sklearn", n_trees=100):
    """Time fit+predict for RF classification."""
    fits, preds_t = [], []
    last_preds = None
    runs = WARMUP_RUNS + TIMED_RUNS
    for r in range(runs):
        if engine == "sklearn":
            m = RandomForestClassifier(n_estimators=n_trees, random_state=0, n_jobs=-1)
            t0 = time.perf_counter()
            m.fit(X_tr, yc_tr)
            fit_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            last_preds = m.predict(X_te)
            pred_ms = (time.perf_counter() - t0) * 1000
        else:
            m = RandomForest(n_trees=n_trees, seed=42)
            t0 = time.perf_counter()
            m.fit_clf(np.ascontiguousarray(X_tr), yc_tr)
            fit_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            last_preds = m.predict_clf(np.ascontiguousarray(X_te))
            pred_ms = (time.perf_counter() - t0) * 1000
        if r >= WARMUP_RUNS:
            fits.append(fit_ms)
            preds_t.append(pred_ms)
    acc = float((last_preds == yc_te).mean())
    return np.median(fits), np.median(preds_t), acc


def _time_rf_reg(X_tr, yr_tr, X_te, yr_te, engine="sklearn", n_trees=100):
    """Time fit+predict for RF regression."""
    fits, preds_t = [], []
    last_preds = None
    runs = WARMUP_RUNS + TIMED_RUNS
    for r in range(runs):
        if engine == "sklearn":
            m = RandomForestRegressor(n_estimators=n_trees, random_state=0, n_jobs=-1)
            t0 = time.perf_counter()
            m.fit(X_tr, yr_tr)
            fit_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            last_preds = m.predict(X_te)
            pred_ms = (time.perf_counter() - t0) * 1000
        else:
            m = RandomForest(n_trees=n_trees, seed=42)
            t0 = time.perf_counter()
            m.fit_reg(np.ascontiguousarray(X_tr), yr_tr)
            fit_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            last_preds = m.predict_reg(np.ascontiguousarray(X_te))
            pred_ms = (time.perf_counter() - t0) * 1000
        if r >= WARMUP_RUNS:
            fits.append(fit_ms)
            preds_t.append(pred_ms)
    mse_val = float(np.mean((last_preds - yr_te) ** 2))
    return np.median(fits), np.median(preds_t), mse_val


def main():
    n_trees = 100

    # ============================= CART ==============================
    print()
    print("=" * 120)
    print("  PART 1: CART — ml (sort, auto) vs sklearn DecisionTree")
    print(f"  Timing: median of {TIMED_RUNS} runs (after {WARMUP_RUNS} warmup)")
    print("=" * 120)

    # --- CART Classification ---
    print(f"\n{'Dataset':<18} {'N':>7}  {'sk fit':>9}  {'sort fit':>9}  "
          f"{'auto fit':>9}  {'sk/sort':>7}  {'sk/auto':>7}  "
          f"{'sk acc':>7}  {'sort acc':>7}  {'auto acc':>7}")
    print("-" * 120)

    for name, gen_fn, sizes in DATASETS_CLF:
        for n in sizes:
            X, y = gen_fn(n)
            train_n = int(n * 0.8)
            X_tr, X_te = X[:train_n], X[train_n:]
            y_tr, y_te = y[:train_n], y[train_n:]

            sk_fit, _, sk_acc = _time_cart_clf(X_tr, y_tr, X_te, y_te, "sklearn")
            sort_fit, _, sort_acc = _time_cart_clf(X_tr, y_tr, X_te, y_te, "ml", NO_HIST)
            auto_fit, _, auto_acc = _time_cart_clf(X_tr, y_tr, X_te, y_te, "ml", 1024)

            sk_sort = sk_fit / max(sort_fit, 0.001)
            sk_auto = sk_fit / max(auto_fit, 0.001)

            print(f"{name} {n:>7}  {sk_fit:>8.2f}ms  {sort_fit:>8.2f}ms  "
                  f"{auto_fit:>8.2f}ms  {sk_sort:>6.2f}x  {sk_auto:>6.2f}x  "
                  f"{sk_acc:>7.4f}  {sort_acc:>7.4f}  {auto_acc:>7.4f}")
        print()

    # --- CART Regression ---
    print(f"{'Dataset':<18} {'N':>7}  {'sk fit':>9}  {'sort fit':>9}  "
          f"{'auto fit':>9}  {'sk/sort':>7}  {'sk/auto':>7}  "
          f"{'sk mse':>10}  {'sort mse':>10}  {'auto mse':>10}")
    print("-" * 120)

    for name, gen_fn, sizes in DATASETS_REG:
        for n in sizes:
            X, y = gen_fn(n)
            train_n = int(n * 0.8)
            X_tr, X_te = X[:train_n], X[train_n:]
            y_tr, y_te = y[:train_n], y[train_n:]

            sk_fit, _, sk_mse = _time_cart_reg(X_tr, y_tr, X_te, y_te, "sklearn")
            sort_fit, _, sort_mse = _time_cart_reg(X_tr, y_tr, X_te, y_te, "ml", NO_HIST)
            auto_fit, _, auto_mse = _time_cart_reg(X_tr, y_tr, X_te, y_te, "ml", 1024)

            sk_sort = sk_fit / max(sort_fit, 0.001)
            sk_auto = sk_fit / max(auto_fit, 0.001)

            print(f"{name} {n:>7}  {sk_fit:>8.2f}ms  {sort_fit:>8.2f}ms  "
                  f"{auto_fit:>8.2f}ms  {sk_sort:>6.2f}x  {sk_auto:>6.2f}x  "
                  f"{sk_mse:>10.4f}  {sort_mse:>10.4f}  {auto_mse:>10.4f}")
        print()

    # ======================== Random Forest ==========================
    print()
    print("=" * 120)
    print(f"  PART 2: Random Forest ({n_trees} trees) — ml vs sklearn")
    print("  sklearn uses n_jobs=-1 (all cores). ml uses rayon (all cores).")
    print(f"  Timing: median of {TIMED_RUNS} runs (after {WARMUP_RUNS} warmup)")
    print("=" * 120)

    # --- RF Classification ---
    print(f"\n{'Dataset':<18} {'N':>7}  {'sk fit':>10}  {'ax fit':>10}  "
          f"{'speedup':>7}  {'sk acc':>7}  {'ax acc':>7}")
    print("-" * 80)

    for name, gen_fn, sizes in DATASETS_CLF:
        for n in sizes:
            X, y = gen_fn(n)
            train_n = int(n * 0.8)
            X_tr, X_te = X[:train_n], X[train_n:]
            y_tr, y_te = y[:train_n], y[train_n:]

            sk_fit, _, sk_acc = _time_rf_clf(X_tr, y_tr, X_te, y_te, "sklearn", n_trees)
            ax_fit, _, ax_acc = _time_rf_clf(X_tr, y_tr, X_te, y_te, "ml", n_trees)

            speedup = sk_fit / max(ax_fit, 0.001)

            print(f"{name} {n:>7}  {sk_fit:>9.1f}ms  {ax_fit:>9.1f}ms  "
                  f"{speedup:>6.2f}x  {sk_acc:>7.4f}  {ax_acc:>7.4f}")
        print()

    # --- RF Regression ---
    print(f"{'Dataset':<18} {'N':>7}  {'sk fit':>10}  {'ax fit':>10}  "
          f"{'speedup':>7}  {'sk mse':>10}  {'ax mse':>10}")
    print("-" * 90)

    for name, gen_fn, sizes in DATASETS_REG:
        for n in sizes:
            X, y = gen_fn(n)
            train_n = int(n * 0.8)
            X_tr, X_te = X[:train_n], X[train_n:]
            y_tr, y_te = y[:train_n], y[train_n:]

            sk_fit, _, sk_mse = _time_rf_reg(X_tr, y_tr, X_te, y_te, "sklearn", n_trees)
            ax_fit, _, ax_mse = _time_rf_reg(X_tr, y_tr, X_te, y_te, "ml", n_trees)

            speedup = sk_fit / max(ax_fit, 0.001)

            print(f"{name} {n:>7}  {sk_fit:>9.1f}ms  {ax_fit:>9.1f}ms  "
                  f"{speedup:>6.2f}x  {sk_mse:>10.4f}  {ax_mse:>10.4f}")
        print()

    print("Legend:")
    print("  sk = sklearn, ax = ml")
    print("  speedup = sklearn_fit / ml_fit  (>1 = ml faster)")
    print("  CART auto = histogram for nodes >= 1024 samples, sort below")
    print(f"  RF: {n_trees} trees, both using all available cores")
    print()


if __name__ == "__main__":
    main()
