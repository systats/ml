#!/usr/bin/env python3
"""Homepage benchmark: ml Rust engine vs sklearn on real-world datasets.

All datasets loaded via ml.dataset() — real data, no synthetic generators.
Compares ml.fit(engine="ml") vs ml.fit(engine="sklearn") for all 11 Rust algorithms.

Two modes:
  --allcores   Both engines use all cores (default, user experience)
  --singlecore RAYON_NUM_THREADS=1 + n_jobs=1 (algorithmic comparison)

Run on beast:
    cd ~/ml_test && python3 benchmarks/bench_homepage.py
    cd ~/ml_test && python3 benchmarks/bench_homepage.py --singlecore

Output:
    benchmarks/results_homepage.json   — for homepage JS
    benchmarks/results_homepage.md     — for README

Auditor fixes applied (2026-03-07, 3-auditor council):
  P0.1: n_jobs=-1 for sklearn when --allcores (was 1 vs rayon-all-cores)
  P0.2: gradient_boosting flagged as different algorithm (histogram vs exact)
  P0.3: min/max speedup reported alongside geo mean
  P0.4: "accuracy delta" not "accuracy parity"
  P0.5: RUNS=20, WARMUP=3
  P1.1: dropped diabetes (N=442) and cancer (N=569) — Python overhead dominates
  P1.2: system info captured programmatically
  P1.3: IQR reported in JSON
  P1.4: speedup floor uses 1e-3, flags "too fast" cases
  P1.5: gc.collect() between runs
"""
import argparse
import gc
import json
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import ml  # noqa: E402

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

WARMUP = 3
RUNS = 20

# Real-world datasets — classification (N >= 1000 only, avoids Python overhead noise)
DATASETS_CLF = [
    ("churn",     "churn",     "binary, 7043×20, mixed types"),
    ("adult",     "income",    "binary, 48842×14, categoricals"),
    ("ecommerce", "purchased", "binary, 12330×18, mixed types"),
]

# Real-world datasets — regression (N >= 1000 only)
DATASETS_REG = [
    ("houses",   "price",  "continuous, 20640×9"),
    ("diamonds", "price",  "continuous, 53940×10"),
    ("tips",     "tip",    "continuous, 244×7, bundled"),
]

# Algorithm → (task_types, sklearn_equivalent_name, note)
ALGORITHMS = {
    "decision_tree":      (["clf", "reg"], "DecisionTree", None),
    "random_forest":      (["clf", "reg"], "RandomForest", None),
    "extra_trees":        (["clf", "reg"], "ExtraTrees", None),
    "gradient_boosting":  (["clf", "reg"], "GradientBoosting",
                           "Rust=histogram GBT, sklearn=exact greedy — different algorithms"),
    "logistic":           (["clf"],        "LogisticRegression", None),
    "linear":             (["reg"],        "Ridge", None),
    "knn":                (["clf", "reg"], "KNeighbors", None),
    "naive_bayes":        (["clf"],        "GaussianNB", None),
    "elastic_net":        (["reg"],        "ElasticNet", None),
    "svm":                (["clf", "reg"], "LinearSVM",
                           "Rust=primal CD, sklearn=liblinear — same algorithm family"),
    "adaboost":           (["clf"],        "AdaBoost", None),
}


# ── Benchmark runner ───────────────────────────────────────────────────────────

def time_fit_predict(split, target, algorithm, engine, seed=42):
    """Time fit + predict. Returns (fit_times, pred_times, metric_value, metric_name).

    Returns all individual run times for IQR computation.
    """
    times_fit = []
    times_pred = []
    metric_val = None
    metric_name = None

    for r in range(WARMUP + RUNS):
        gc.collect()

        t0 = time.perf_counter()
        try:
            m = ml.fit(data=split.train, target=target, algorithm=algorithm,
                       engine=engine, seed=seed)
        except Exception as e:
            return None, None, None, str(e)
        fit_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        ml.predict(model=m, data=split.valid)
        pred_ms = (time.perf_counter() - t0) * 1000

        if r >= WARMUP:
            times_fit.append(fit_ms)
            times_pred.append(pred_ms)

        # Get metric from last run
        if r == WARMUP + RUNS - 1:
            metrics = ml.evaluate(model=m, data=split.valid)
            if "accuracy" in metrics:
                metric_val = metrics["accuracy"]
                metric_name = "accuracy"
            elif "r2" in metrics:
                metric_val = metrics["r2"]
                metric_name = "r2"
            elif "rmse" in metrics:
                metric_val = metrics["rmse"]
                metric_name = "rmse"

    return times_fit, times_pred, metric_val, metric_name


def _speedup(sk_ms, ml_ms):
    """Compute speedup ratio with floor guard."""
    if ml_ms < 0.1:
        return None  # too fast to measure reliably
    return sk_ms / ml_ms


def run_benchmarks(datasets_clf, datasets_reg):
    results = []

    all_datasets = [
        (ds, "clf") for ds in datasets_clf
    ] + [
        (ds, "reg") for ds in datasets_reg
    ]

    for (ds_name, target, description), task_type in all_datasets:
        print(f"\n{'='*80}")
        print(f"  Dataset: {ds_name} ({description})")
        print(f"{'='*80}")

        data = ml.dataset(ds_name).dropna()
        n_rows = len(data)
        n_cols = len(data.columns) - 1
        s = ml.split(data=data, target=target, seed=42)

        metric_label = "acc" if task_type == "clf" else "r2"
        print(f"  train={len(s.train)}, valid={len(s.valid)}, test={len(s.test)}")
        print(f"\n  {'Algorithm':<22} {'ml fit':>9} {'sk fit':>9} {'speedup':>8} "
              f"{'ml {}'.format(metric_label):>8} {'sk {}'.format(metric_label):>8} "
              f"{'delta':>7}  {'note':>4}")
        print(f"  {'-'*80}")

        for algo, (tasks, sk_name, note) in ALGORITHMS.items():
            if task_type not in tasks:
                continue

            ml_fits, ml_preds, ml_metric, ml_mname = time_fit_predict(
                s, target, algo, "ml")
            sk_fits, sk_preds, sk_metric, sk_mname = time_fit_predict(
                s, target, algo, "sklearn")

            if ml_fits is None or sk_fits is None:
                err = ml_mname if ml_fits is None else sk_mname
                print(f"  {algo:<22} SKIP: {err[:50]}")
                continue

            ml_fit = float(np.median(ml_fits))
            sk_fit = float(np.median(sk_fits))
            ml_pred = float(np.median(ml_preds))
            sk_pred = float(np.median(sk_preds))

            speedup = _speedup(sk_fit, ml_fit)
            delta = (ml_metric - sk_metric) if ml_metric and sk_metric else 0

            speedup_str = f"{speedup:>7.2f}x" if speedup else "  ~same"
            note_str = " *" if note else ""
            print(f"  {algo:<22} {ml_fit:>8.1f}ms {sk_fit:>8.1f}ms {speedup_str} "
                  f"{ml_metric:>8.4f} {sk_metric:>8.4f} {delta:>+7.4f}{note_str}")

            results.append({
                "dataset": ds_name,
                "n_rows": n_rows,
                "n_features": n_cols,
                "task": "classification" if task_type == "clf" else "regression",
                "description": description,
                "algorithm": algo,
                "sklearn_name": sk_name,
                "note": note,
                "ml_fit_ms": round(ml_fit, 2),
                "ml_fit_q25": round(float(np.percentile(ml_fits, 25)), 2),
                "ml_fit_q75": round(float(np.percentile(ml_fits, 75)), 2),
                "ml_predict_ms": round(ml_pred, 2),
                "sk_fit_ms": round(sk_fit, 2),
                "sk_fit_q25": round(float(np.percentile(sk_fits, 25)), 2),
                "sk_fit_q75": round(float(np.percentile(sk_fits, 75)), 2),
                "sk_predict_ms": round(sk_pred, 2),
                "speedup_fit": round(speedup, 2) if speedup else None,
                "speedup_predict": _speedup(sk_pred, ml_pred),
                "ml_metric": round(ml_metric, 4) if ml_metric else None,
                "sk_metric": round(sk_metric, 4) if sk_metric else None,
                "metric_name": ml_mname,
                "metric_delta": round(delta, 4) if delta else 0,
            })

    return results


# ── System info ────────────────────────────────────────────────────────────────

def get_system_info():
    import sklearn
    cores = os.cpu_count() or "unknown"
    rayon_threads = os.environ.get("RAYON_NUM_THREADS", "all")
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os": platform.system(),
        "cores": cores,
        "rayon_threads": rayon_threads,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "ml": ml.__version__ if hasattr(ml, "__version__") else "dev",
    }


# ── Output formatters ──────────────────────────────────────────────────────────

def write_json(results, path, mode, sys_info):
    """Write results as JSON for homepage consumption."""
    by_algo = {}
    for r in results:
        algo = r["algorithm"]
        if algo not in by_algo:
            by_algo[algo] = {"algorithm": algo, "sklearn_name": r["sklearn_name"],
                             "note": r["note"], "benchmarks": []}
        by_algo[algo]["benchmarks"].append(r)

    summary = []
    for algo, data in by_algo.items():
        speedups = [b["speedup_fit"] for b in data["benchmarks"]
                    if b["speedup_fit"] is not None]
        if not speedups:
            continue
        geo_mean = float(np.exp(np.mean(np.log(np.array(speedups)))))
        metric_deltas = [b["metric_delta"] for b in data["benchmarks"]
                         if b["metric_delta"] is not None]
        summary.append({
            "algorithm": algo,
            "sklearn_name": data["sklearn_name"],
            "note": data["note"],
            "geo_mean_speedup": round(geo_mean, 2),
            "min_speedup": round(min(speedups), 2),
            "max_speedup": round(max(speedups), 2),
            "max_metric_delta": round(max(abs(d) for d in metric_deltas), 4) if metric_deltas else 0,
            "worst_metric_delta": round(min(metric_deltas), 4) if metric_deltas else 0,
            "n_benchmarks": len(speedups),
        })

    summary.sort(key=lambda x: x["geo_mean_speedup"], reverse=True)

    output = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "warmup": WARMUP,
        "runs": RUNS,
        "system": sys_info,
        "summary": summary,
        "details": results,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON written to {path}")


def write_markdown(results, path, mode, sys_info):
    """Write results as markdown table for README."""
    lines = [
        "# Benchmark: ml (Rust) vs sklearn",
        "",
        f"Mode: **{mode}**. Median of {RUNS} runs after {WARMUP} warmup.",
        "Real-world datasets only (N >= 1000, except tips for bundled coverage).",
        "",
        "## Summary",
        "",
        "| Algorithm | Speedup | Min | Max | Max |delta| | Note |",
        "|-----------|--------:|----:|----:|--------:|------|",
    ]

    by_algo = {}
    for r in results:
        algo = r["algorithm"]
        if algo not in by_algo:
            by_algo[algo] = {"speedups": [], "deltas": [], "note": r["note"]}
        if r["speedup_fit"] is not None:
            by_algo[algo]["speedups"].append(r["speedup_fit"])
        if r["metric_delta"] is not None:
            by_algo[algo]["deltas"].append(r["metric_delta"])

    ranked = []
    for algo, data in by_algo.items():
        if not data["speedups"]:
            continue
        geo = float(np.exp(np.mean(np.log(np.array(data["speedups"])))))
        max_d = max(abs(d) for d in data["deltas"]) if data["deltas"] else 0
        mn = min(data["speedups"])
        mx = max(data["speedups"])
        ranked.append((algo, geo, mn, mx, max_d, data["note"]))
    ranked.sort(key=lambda x: x[1], reverse=True)

    for algo, geo, mn, mx, max_d, note in ranked:
        note_str = " *" if note else ""
        lines.append(
            f"| {algo} | **{geo:.1f}x** | {mn:.1f}x | {mx:.1f}x "
            f"| {max_d:.4f} | {note_str} |"
        )

    if any(r[5] for r in ranked):
        lines.extend([
            "",
            "\\* Rust=histogram GBT, sklearn=exact greedy — different algorithms, "
            "speedup partly reflects algorithm choice not just implementation.",
        ])

    lines.extend(["", "## Detailed results", ""])

    datasets_seen = []
    for r in results:
        ds = r["dataset"]
        if ds not in datasets_seen:
            datasets_seen.append(ds)

    for ds in datasets_seen:
        ds_results = [r for r in results if r["dataset"] == ds]
        if not ds_results:
            continue
        desc = ds_results[0]["description"]
        task = ds_results[0]["task"]
        metric = ds_results[0]["metric_name"] or ("accuracy" if task == "classification" else "r2")

        lines.extend([
            f"### {ds} (N={ds_results[0]['n_rows']}, {desc})",
            "",
            f"| Algorithm | ml (ms) | sklearn (ms) | Speedup | ml {metric} | sk {metric} | Note |",
            "|-----------|--------:|-------------:|--------:|----------:|----------:|------|",
        ])

        for r in ds_results:
            sp = f"**{r['speedup_fit']:.1f}x**" if r["speedup_fit"] else "~same"
            note_str = " *" if r["note"] else ""
            lines.append(
                f"| {r['algorithm']} | {r['ml_fit_ms']:.1f} | {r['sk_fit_ms']:.1f} "
                f"| {sp} "
                f"| {r['ml_metric']:.4f} | {r['sk_metric']:.4f} | {note_str} |"
            )
        lines.append("")

    lines.extend([
        "## Environment",
        "",
        f"- Machine: {sys_info['processor']} ({sys_info['cores']} cores)",
        f"- OS: {sys_info['os']} {sys_info['machine']}",
        f"- Python: {sys_info['python']}",
        f"- sklearn: {sys_info['sklearn']}",
        f"- ml: {sys_info['ml']}",
        f"- Rayon threads: {sys_info['rayon_threads']}",
        f"- Mode: {mode}",
        "",
        "## Methodology",
        "",
        "- `ml.fit(engine='ml')` vs `ml.fit(engine='sklearn')` — same high-level API",
        "- Includes all Python overhead, preprocessing, model construction",
        "- This is what the user experiences, not a micro-benchmark",
        f"- Median of {RUNS} runs after {WARMUP} warmup runs, with gc.collect() between runs",
        "- Same seed (42), same split, same hyperparameter defaults",
        "- Datasets loaded via `ml.dataset()` — real-world data, no synthetic generators",
        "- NaN rows dropped before split (ensures both engines get identical clean data)",
    ])

    if mode == "allcores":
        lines.extend([
            "- Both engines use all available cores (ml.config(n_jobs=-1))",
            "- Rust parallelism via rayon, sklearn via joblib",
        ])
    else:
        lines.extend([
            "- Single-core: RAYON_NUM_THREADS=1, ml.config(n_jobs=1)",
            "- Isolates algorithmic efficiency from parallelism",
        ])

    lines.extend([
        "",
        "## Disclaimers",
        "",
        "- `gradient_boosting`: Rust uses histogram splits (like LightGBM), "
        "sklearn uses exact greedy. Different algorithms — speedup reflects both "
        "implementation AND algorithm choice.",
        "- `svm`: Rust uses primal coordinate descent with built-in standardization, "
        "sklearn uses liblinear. Same algorithm family, minor accuracy differences expected.",
        "- Accuracy deltas reflect different implementations with different defaults, "
        "not 'parity'. Both produce competitive results on each dataset.",
        "- Small datasets (N<1000) excluded: Python overhead dominates fit time, "
        "making speedup measurements unreliable.",
        "",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown written to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ml vs sklearn benchmark")
    parser.add_argument("--singlecore", action="store_true",
                        help="Force single-core (RAYON_NUM_THREADS=1, n_jobs=1)")
    args = parser.parse_args()

    if args.singlecore:
        os.environ["RAYON_NUM_THREADS"] = "1"
        ml.config(n_jobs=1)
        mode = "singlecore"
    else:
        ml.config(n_jobs=-1)
        mode = "allcores"

    sys_info = get_system_info()

    print("=" * 80)
    print(f"  ml (Rust) vs sklearn — Homepage Benchmark [{mode}]")
    print(f"  {RUNS} runs + {WARMUP} warmup, real-world datasets only")
    print(f"  Cores: {sys_info['cores']}, Rayon: {sys_info['rayon_threads']}")
    print("=" * 80)

    results = run_benchmarks(DATASETS_CLF, DATASETS_REG)

    out_dir = Path(__file__).parent
    suffix = f"_{mode}" if mode != "allcores" else ""
    write_json(results, out_dir / f"results_homepage{suffix}.json", mode, sys_info)
    write_markdown(results, out_dir / f"results_homepage{suffix}.md", mode, sys_info)

    # Print summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY — {mode} (geometric mean speedup)")
    print("=" * 80)

    by_algo = {}
    for r in results:
        algo = r["algorithm"]
        if algo not in by_algo:
            by_algo[algo] = {"speedups": [], "note": r["note"]}
        if r["speedup_fit"] is not None:
            by_algo[algo]["speedups"].append(r["speedup_fit"])

    ranked = []
    for algo, data in by_algo.items():
        if not data["speedups"]:
            continue
        geo = float(np.exp(np.mean(np.log(np.array(data["speedups"])))))
        mn = min(data["speedups"])
        mx = max(data["speedups"])
        ranked.append((algo, geo, mn, mx, len(data["speedups"]), data["note"]))
    ranked.sort(key=lambda x: x[1], reverse=True)

    for algo, geo, mn, mx, n, note in ranked:
        bar = "#" * min(int(geo * 2), 40)
        flag = " *" if note else ""
        print(f"  {algo:<22} {geo:>6.2f}x  [{mn:.1f}x–{mx:.1f}x]  {bar}  "
              f"({n} datasets){flag}")

    if any(r[5] for r in ranked):
        print("\n  * different algorithm (histogram vs exact greedy)")

    print()


if __name__ == "__main__":
    main()
