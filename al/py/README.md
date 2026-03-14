# ml-py

Rust kernels for the [ml](https://pypi.org/project/mlw/) machine learning toolkit.

This package provides PyO3 bindings to native Rust implementations of 12 algorithm families:
Decision Tree, Random Forest, Extra Trees, Gradient Boosting, Logistic Regression,
Ridge Regression, Elastic Net, K-Nearest Neighbors, Naive Bayes, AdaBoost, SVM (linear),
and histgradient.

## Usage

This package is a backend dependency of `mlw` — install it via:

```bash
pip install mlw
```

The Rust backend activates automatically when available. No code changes needed.

## License

MIT
