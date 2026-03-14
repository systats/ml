# Benchmark: ml (Rust) vs sklearn

Mode: **singlecore**. Median of 20 runs after 3 warmup.
Real-world datasets only (N >= 1000, except tips for bundled coverage).

## Summary

| Algorithm | Speedup | Min | Max | Max |delta| | Note |
|-----------|--------:|----:|----:|--------:|------|
| gradient_boosting | **5.0x** | 2.0x | 11.8x | 0.0067 |  * |
| elastic_net | **2.7x** | 1.2x | 13.4x | 0.0349 |  |
| adaboost | **1.6x** | 1.2x | 2.0x | 0.0000 |  |
| logistic | **1.5x** | 1.0x | 3.2x | 0.0000 |  |
| decision_tree | **1.4x** | 1.1x | 2.2x | 0.0187 |  |
| extra_trees | **1.4x** | 0.9x | 4.4x | 0.0623 |  |
| random_forest | **1.3x** | 0.7x | 3.8x | 0.0315 |  |
| linear | **1.2x** | 1.1x | 1.3x | 0.0000 |  |
| naive_bayes | **1.1x** | 1.1x | 1.1x | 0.0000 |  |
| knn | **1.0x** | 0.4x | 1.9x | 0.0008 |  |

\* Rust=histogram GBT, sklearn=exact greedy — different algorithms, speedup partly reflects algorithm choice not just implementation.

## Detailed results

### churn (N=7032, binary, 7043×20, mixed types)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml accuracy | sk accuracy | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 32.7 | 35.1 | **1.1x** | 0.6972 | 0.7122 |  |
| random_forest | 222.5 | 238.9 | **1.1x** | 0.7797 | 0.7747 |  |
| extra_trees | 256.1 | 352.4 | **1.4x** | 0.7775 | 0.7704 |  |
| gradient_boosting | 270.2 | 532.6 | **2.0x** | 0.7868 | 0.7868 |  * |
| logistic | 34.8 | 36.9 | **1.1x** | 0.7946 | 0.7946 |  |
| knn | 37.3 | 30.7 | **0.8x** | 0.7576 | 0.7584 |  |
| naive_bayes | 19.5 | 21.2 | **1.1x** | 0.7257 | 0.7257 |  |
| adaboost | 85.0 | 159.6 | **1.9x** | 0.7946 | 0.7946 |  |

### adult (N=45222, binary, 48842×14, categoricals)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml accuracy | sk accuracy | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 98.2 | 111.5 | **1.1x** | 0.8129 | 0.8187 |  |
| random_forest | 1535.4 | 1092.3 | **0.7x** | 0.8597 | 0.8500 |  |
| extra_trees | 1040.6 | 921.2 | **0.9x** | 0.8560 | 0.8418 |  |
| gradient_boosting | 264.7 | 1005.6 | **3.8x** | 0.8659 | 0.8655 |  * |
| logistic | 189.1 | 602.8 | **3.2x** | 0.8514 | 0.8514 |  |
| knn | 137.8 | 140.2 | **1.0x** | 0.8223 | 0.8230 |  |
| naive_bayes | 64.0 | 71.7 | **1.1x** | 0.7972 | 0.7972 |  |
| adaboost | 235.2 | 281.4 | **1.2x** | 0.8493 | 0.8493 |  |

### ecommerce (N=12330, binary, 12330×18, mixed types)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml accuracy | sk accuracy | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 34.0 | 49.2 | **1.4x** | 0.8650 | 0.8532 |  |
| random_forest | 405.5 | 450.9 | **1.1x** | 0.9047 | 0.9043 |  |
| extra_trees | 542.9 | 489.4 | **0.9x** | 0.8974 | 0.8925 |  |
| gradient_boosting | 213.4 | 1157.9 | **5.4x** | 0.9006 | 0.9031 |  * |
| logistic | 25.4 | 25.1 | **1.0x** | 0.8828 | 0.8828 |  |
| knn | 52.7 | 22.0 | **0.4x** | 0.8710 | 0.8710 |  |
| naive_bayes | 14.0 | 15.8 | **1.1x** | 0.8402 | 0.8402 |  |
| adaboost | 140.8 | 278.6 | **2.0x** | 0.8873 | 0.8873 |  |

### houses (N=20640, continuous, 20640×9)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml r2 | sk r2 | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 29.7 | 65.9 | **2.2x** | 0.5894 | 0.6023 |  |
| random_forest | 2469.6 | 3816.2 | **1.6x** | 0.8015 | 0.8026 |  |
| extra_trees | 1027.4 | 1440.5 | **1.4x** | 0.8157 | 0.8075 |  |
| gradient_boosting | 115.8 | 1371.6 | **11.8x** | 0.7869 | 0.7822 |  * |
| linear | 5.1 | 5.7 | **1.1x** | 0.4928 | 0.4928 |  |
| knn | 5.3 | 10.2 | **1.9x** | 0.7292 | 0.7292 |  |
| elastic_net | 5.6 | 74.8 | **13.4x** | 0.2019 | 0.2019 |  |

### diamonds (N=53940, continuous, 53940×10)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml r2 | sk r2 | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 72.3 | 119.8 | **1.7x** | 0.9632 | 0.9626 |  |
| random_forest | 5133.9 | 6248.7 | **1.2x** | 0.9802 | 0.9803 |  |
| extra_trees | 2780.2 | 3488.0 | **1.2x** | 0.9800 | 0.9801 |  |
| gradient_boosting | 200.4 | 1733.7 | **8.7x** | 0.9700 | 0.9709 |  * |
| linear | 40.2 | 43.1 | **1.1x** | 0.9182 | 0.9182 |  |
| knn | 42.7 | 44.8 | **1.1x** | 0.9463 | 0.9463 |  |
| elastic_net | 43.4 | 52.7 | **1.2x** | 0.8721 | 0.8372 |  |

### tips (N=244, continuous, 244×7, bundled)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml r2 | sk r2 | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 3.6 | 4.7 | **1.3x** | 0.1072 | 0.1259 |  |
| random_forest | 11.1 | 41.9 | **3.8x** | 0.3451 | 0.3766 |  |
| extra_trees | 11.6 | 51.6 | **4.4x** | 0.2149 | 0.1526 |  |
| gradient_boosting | 8.1 | 29.1 | **3.6x** | 0.3773 | 0.3840 |  * |
| linear | 4.1 | 5.5 | **1.3x** | 0.3753 | 0.3753 |  |
| knn | 4.5 | 5.5 | **1.2x** | 0.3186 | 0.3186 |  |
| elastic_net | 4.2 | 5.2 | **1.2x** | 0.2401 | 0.2401 |  |

## Environment

- Machine: x86_64 (24 cores)
- OS: Linux x86_64
- Python: 3.12.3
- sklearn: 1.8.0
- ml: 1.0.0
- Rayon threads: 1
- Mode: singlecore

## Methodology

- `ml.fit(engine='ml')` vs `ml.fit(engine='sklearn')` — same high-level API
- Includes all Python overhead, preprocessing, model construction
- This is what the user experiences, not a micro-benchmark
- Median of 20 runs after 3 warmup runs, with gc.collect() between runs
- Same seed (42), same split, same hyperparameter defaults
- Datasets loaded via `ml.dataset()` — real-world data, no synthetic generators
- NaN rows dropped before split (ensures both engines get identical clean data)
- Single-core: RAYON_NUM_THREADS=1, ml.config(n_jobs=1)
- Isolates algorithmic efficiency from parallelism

## Disclaimers

- `gradient_boosting`: Rust uses histogram splits (like LightGBM), sklearn uses exact greedy. Different algorithms — speedup reflects both implementation AND algorithm choice.
- `svm`: excluded — Rust linear SMO accuracy not competitive on several datasets (convergence fix pending).
- Accuracy deltas reflect different implementations with different defaults, not 'parity'. Both produce competitive results on each dataset.
- Small datasets (N<1000) excluded: Python overhead dominates fit time, making speedup measurements unreliable.
