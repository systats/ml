#!/usr/bin/env python3
"""Benchmark: ml CART (sort vs histogram) vs sklearn DecisionTree.

5 representative datasets × 3 sizes.
Three ml modes:
  - sort:  histogram_threshold=2**63  (force sort-based splitting)
  - hist:  histogram_threshold=0       (force histogram-based splitting)
  - auto:  histogram_threshold=1024    (default: sort for small nodes, hist for large)

All data generated with numpy only (deterministic, no sklearn generators).
Timed via in-process PyO3 calls — zero subprocess overhead.
"""
import time

import numpy as np
from ml_py import DecisionTree
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
FORCE_HIST = 0   # force histogram-based


def _time_clf(X_tr, yc_tr, X_te, yc_te, engine="sklearn", hist_thresh=1024):
    """Time fit+predict for classification. Returns (fit_ms, predict_ms, accuracy, preds)."""
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
    return np.median(fits), np.median(preds_t), acc, last_preds


def _time_reg(X_tr, yr_tr, X_te, yr_te, engine="sklearn", hist_thresh=1024):
    """Time fit+predict for regression. Returns (fit_ms, predict_ms, mse)."""
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


def main():
    print()
    print("=" * 130)
    print("  algoX CART v3 — sort vs histogram vs sklearn")
    print(f"  Timing: median of {TIMED_RUNS} runs (after {WARMUP_RUNS} warmup)")
    print("=" * 130)

    # --- Classification ---
    print(f"\n{'Dataset':<18} {'N':>7}  {'sk fit':>9}  {'sort fit':>9}  "
          f"{'auto fit':>9}  {'sk/sort':>7}  {'sk/auto':>7}  "
          f"{'sk acc':>7}  {'sort acc':>7}  {'auto acc':>7}")
    print("-" * 130)

    for name, gen_fn, sizes in DATASETS_CLF:
        for n in sizes:
            X, y = gen_fn(n)
            train_n = int(n * 0.8)
            X_tr, X_te = X[:train_n], X[train_n:]
            y_tr, y_te = y[:train_n], y[train_n:]

            sk_fit, _, sk_acc, _ = _time_clf(X_tr, y_tr, X_te, y_te, "sklearn")
            sort_fit, _, sort_acc, _ = _time_clf(X_tr, y_tr, X_te, y_te, "ml", NO_HIST)
            auto_fit, _, auto_acc, _ = _time_clf(X_tr, y_tr, X_te, y_te, "ml", 1024)

            sk_sort = sk_fit / max(sort_fit, 0.001)
            sk_auto = sk_fit / max(auto_fit, 0.001)

            print(f"{name} {n:>7}  {sk_fit:>8.2f}ms  {sort_fit:>8.2f}ms  "
                  f"{auto_fit:>8.2f}ms  {sk_sort:>6.2f}x  {sk_auto:>6.2f}x  "
                  f"{sk_acc:>7.4f}  {sort_acc:>7.4f}  {auto_acc:>7.4f}")
        print()

    # --- Regression ---
    print(f"{'Dataset':<18} {'N':>7}  {'sk fit':>9}  {'sort fit':>9}  "
          f"{'auto fit':>9}  {'sk/sort':>7}  {'sk/auto':>7}  "
          f"{'sk mse':>10}  {'sort mse':>10}  {'auto mse':>10}")
    print("-" * 130)

    for name, gen_fn, sizes in DATASETS_REG:
        for n in sizes:
            X, y = gen_fn(n)
            train_n = int(n * 0.8)
            X_tr, X_te = X[:train_n], X[train_n:]
            y_tr, y_te = y[:train_n], y[train_n:]

            sk_fit, _, sk_mse = _time_reg(X_tr, y_tr, X_te, y_te, "sklearn")
            sort_fit, _, sort_mse = _time_reg(X_tr, y_tr, X_te, y_te, "ml", NO_HIST)
            auto_fit, _, auto_mse = _time_reg(X_tr, y_tr, X_te, y_te, "ml", 1024)

            sk_sort = sk_fit / max(sort_fit, 0.001)
            sk_auto = sk_fit / max(auto_fit, 0.001)

            print(f"{name} {n:>7}  {sk_fit:>8.2f}ms  {sort_fit:>8.2f}ms  "
                  f"{auto_fit:>8.2f}ms  {sk_sort:>6.2f}x  {sk_auto:>6.2f}x  "
                  f"{sk_mse:>10.4f}  {sort_mse:>10.4f}  {auto_mse:>10.4f}")
        print()

    print("Legend:")
    print("  sk/sort = sklearn_fit / ml_sort_fit  (>1 = ml faster)")
    print("  sk/auto = sklearn_fit / ml_auto_fit  (>1 = ml faster)")
    print("  sort = force sort-based splitting (v2 algorithm)")
    print("  auto = histogram for nodes >= 1024 samples, sort below (v3 default)")
    print()


if __name__ == "__main__":
    main()
