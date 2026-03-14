# Benchmark: ml (Rust) vs sklearn

Mode: **allcores**. Median of 20 runs after 3 warmup.
Real-world datasets only (N >= 1000, except tips for bundled coverage).

## Summary

| Algorithm | Speedup | Min | Max | Max |delta| | Note |
|-----------|--------:|----:|----:|--------:|------|
| gradient_boosting | **4.0x** | 1.5x | 10.6x | 0.0067 |  * |
| random_forest | **3.0x** | 1.7x | 12.1x | 0.0315 |  |
| elastic_net | **2.7x** | 1.1x | 13.3x | 0.0349 |  |
| extra_trees | **2.7x** | 1.2x | 15.6x | 0.0623 |  |
| adaboost | **1.6x** | 1.2x | 2.0x | 0.0000 |  |
| decision_tree | **1.4x** | 1.1x | 2.2x | 0.0187 |  |
| logistic | **1.4x** | 1.0x | 2.4x | 0.0000 |  |
| knn | **1.3x** | 0.6x | 3.9x | 0.0008 |  |
| linear | **1.1x** | 1.0x | 1.3x | 0.0000 |  |
| naive_bayes | **1.0x** | 0.9x | 1.1x | 0.0000 |  |

\* Rust=histogram GBT, sklearn=exact greedy — different algorithms, speedup partly reflects algorithm choice not just implementation.

## Detailed results

### churn (N=7032, binary, 7043×20, mixed types)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml accuracy | sk accuracy | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 33.1 | 35.9 | **1.1x** | 0.6972 | 0.7122 |  |
| random_forest | 34.0 | 113.3 | **3.3x** | 0.7797 | 0.7747 |  |
| extra_trees | 37.4 | 139.6 | **3.7x** | 0.7775 | 0.7704 |  |
| gradient_boosting | 300.2 | 535.2 | **1.8x** | 0.7868 | 0.7868 |  * |
| logistic | 36.5 | 38.1 | **1.1x** | 0.7946 | 0.7946 |  |
| knn | 33.4 | 30.7 | **0.9x** | 0.7576 | 0.7584 |  |
| naive_bayes | 19.3 | 21.2 | **1.1x** | 0.7257 | 0.7257 |  |
| adaboost | 85.2 | 158.3 | **1.9x** | 0.7946 | 0.7946 |  |

### adult (N=45222, binary, 48842×14, categoricals)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml accuracy | sk accuracy | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 98.0 | 112.4 | **1.1x** | 0.8129 | 0.8187 |  |
| random_forest | 184.1 | 313.2 | **1.7x** | 0.8597 | 0.8500 |  |
| extra_trees | 150.7 | 176.8 | **1.2x** | 0.8560 | 0.8418 |  |
| gradient_boosting | 277.7 | 1006.5 | **3.6x** | 0.8659 | 0.8655 |  * |
| logistic | 197.1 | 475.7 | **2.4x** | 0.8514 | 0.8514 |  |
| knn | 141.4 | 140.8 | **1.0x** | 0.8223 | 0.8230 |  |
| naive_bayes | 74.2 | 66.7 | **0.9x** | 0.7972 | 0.7972 |  |
| adaboost | 233.8 | 284.0 | **1.2x** | 0.8493 | 0.8493 |  |

### ecommerce (N=12330, binary, 12330×18, mixed types)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml accuracy | sk accuracy | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 35.2 | 47.9 | **1.4x** | 0.8650 | 0.8532 |  |
| random_forest | 42.1 | 126.9 | **3.0x** | 0.9047 | 0.9043 |  |
| extra_trees | 56.0 | 142.2 | **2.5x** | 0.8974 | 0.8925 |  |
| gradient_boosting | 238.0 | 1159.4 | **4.9x** | 0.9006 | 0.9031 |  * |
| logistic | 25.8 | 25.3 | **1.0x** | 0.8828 | 0.8828 |  |
| knn | 39.9 | 24.3 | **0.6x** | 0.8710 | 0.8710 |  |
| naive_bayes | 14.4 | 15.9 | **1.1x** | 0.8402 | 0.8402 |  |
| adaboost | 139.9 | 277.1 | **2.0x** | 0.8873 | 0.8873 |  |

### houses (N=20640, continuous, 20640×9)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml r2 | sk r2 | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 29.9 | 66.0 | **2.2x** | 0.5894 | 0.6023 |  |
| random_forest | 176.2 | 364.4 | **2.1x** | 0.8015 | 0.8026 |  |
| extra_trees | 76.8 | 122.4 | **1.6x** | 0.8157 | 0.8075 |  |
| gradient_boosting | 130.0 | 1373.6 | **10.6x** | 0.7869 | 0.7822 |  * |
| linear | 4.9 | 5.5 | **1.1x** | 0.4928 | 0.4928 |  |
| knn | 5.6 | 10.5 | **1.9x** | 0.7292 | 0.7292 |  |
| elastic_net | 5.6 | 74.8 | **13.3x** | 0.2019 | 0.2019 |  |

### diamonds (N=53940, continuous, 53940×10)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml r2 | sk r2 | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 72.7 | 119.9 | **1.6x** | 0.9632 | 0.9626 |  |
| random_forest | 385.0 | 652.8 | **1.7x** | 0.9802 | 0.9803 |  |
| extra_trees | 220.8 | 291.9 | **1.3x** | 0.9800 | 0.9801 |  |
| gradient_boosting | 206.1 | 1730.2 | **8.4x** | 0.9700 | 0.9709 |  * |
| linear | 42.4 | 42.6 | **1.0x** | 0.9182 | 0.9182 |  |
| knn | 43.7 | 43.4 | **1.0x** | 0.9463 | 0.9463 |  |
| elastic_net | 46.8 | 53.6 | **1.1x** | 0.8721 | 0.8372 |  |

### tips (N=244, continuous, 244×7, bundled)

| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml r2 | sk r2 | Note |
|-----------|--------:|-------------:|--------:|----------:|----------:|------|
| decision_tree | 3.7 | 4.8 | **1.3x** | 0.1072 | 0.1259 |  |
| random_forest | 4.4 | 53.5 | **12.1x** | 0.3451 | 0.3766 |  |
| extra_trees | 5.0 | 77.3 | **15.6x** | 0.2149 | 0.1526 |  |
| gradient_boosting | 19.4 | 29.1 | **1.5x** | 0.3773 | 0.3840 |  * |
| linear | 4.3 | 5.6 | **1.3x** | 0.3753 | 0.3753 |  |
| knn | 4.7 | 18.1 | **3.9x** | 0.3186 | 0.3186 |  |
| elastic_net | 4.5 | 5.7 | **1.3x** | 0.2401 | 0.2401 |  |

## Environment

- Machine: x86_64 (24 cores)
- OS: Linux x86_64
- Python: 3.12.3
- sklearn: 1.8.0
- ml: 1.0.0
- Rayon threads: all
- Mode: allcores

## Methodology

- `ml.fit(engine='ml')` vs `ml.fit(engine='sklearn')` — same high-level API
- Includes all Python overhead, preprocessing, model construction
- This is what the user experiences, not a micro-benchmark
- Median of 20 runs after 3 warmup runs, with gc.collect() between runs
- Same seed (42), same split, same hyperparameter defaults
- Datasets loaded via `ml.dataset()` — real-world data, no synthetic generators
- NaN rows dropped before split (ensures both engines get identical clean data)
- Both engines use all available cores (ml.config(n_jobs=-1))
- Rust parallelism via rayon, sklearn via joblib

## Disclaimers

- `gradient_boosting`: Rust uses histogram splits (like LightGBM), sklearn uses exact greedy. Different algorithms — speedup reflects both implementation AND algorithm choice.
- `svm`: excluded — Rust linear SMO accuracy not competitive on several datasets (convergence fix pending).
- Accuracy deltas reflect different implementations with different defaults, not 'parity'. Both produce competitive results on each dataset.
- Small datasets (N<1000) excluded: Python overhead dominates fit time, making speedup measurements unreliable.
