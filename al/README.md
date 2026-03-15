<p align="center">
  <a href="https://epagogy.ai">
    <img src="https://epagogy.ai/assets/img/hex-ml-logo.svg" width="140" alt="ml"/>
  </a>
</p>

<h1 align="center">al</h1>

<p align="center">
<strong>Algorithm Layer.</strong> Native Rust kernels for the ml ecosystem.<br/>
11 families. No BLAS. No system dependencies.
</p>

<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT"/></a>
  <a href="https://github.com/epagogy/ml"><img src="https://img.shields.io/badge/Rust-1.87+-DEA584" alt="Rust"/></a>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.19023838">Paper</a> · <a href="../python/">Python</a> · <a href="../r/">R</a> · <a href="https://epagogy.ai">epagogy.ai</a>
</p>

```
al/
├── core/     Pure Rust. nalgebra, rayon, serde.
├── py/       PyO3 bindings for Python.
└── r/        extendr bindings for R.
```

All models serialize to JSON. Users see `engine="auto"` (Rust) or `engine="sklearn"` / `engine="r"` (fallback).

## Algorithms

| Algorithm | Module | Clf | Reg |
|-----------|--------|:---:|:---:|
| Linear (Ridge) | `ml::linear` | . | Y |
| Logistic (OvR, L-BFGS) | `ml::logistic` | Y | . |
| Decision Tree (CART) | `ml::tree` | Y | Y |
| Random Forest | `ml::forest` | Y | Y |
| Extra Trees | `ml::forest` | Y | Y |
| Gradient Boosting | `ml::gbt` | Y | Y |
| KNN (KD-tree) | `ml::knn` | Y | Y |
| Naive Bayes | `ml::naive_bayes` | Y | . |
| Elastic Net | `ml::elastic_net` | . | Y |
| AdaBoost (SAMME) | `ml::adaboost` | Y | . |
| SVM (linear SMO) | `ml::svm` | Y | Y |

## Build

```bash
cargo build && cargo test && cargo clippy -- -D warnings
```

## License

MIT. [Simon Roth](https://epagogy.ai), 2026.

<p align="center"><a href="https://epagogy.ai"><img src="../logo-github.png" width="28" alt="epagogy"/></a></p>
