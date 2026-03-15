# ml-r

Rust algorithm kernels for the `ml` R package, via [extendr](https://extendr.github.io/).

Part of the [ml ecosystem](https://epagogy.ai). Paper: [Roth (2026)](https://doi.org/10.5281/zenodo.19023838).

## Algorithms

| Algorithm | Classification | Regression |
|-----------|:-:|:-:|
| Linear (Ridge) | — | Y |
| Logistic (OvR, L-BFGS) | Y | — |
| Decision Tree (CART) | Y | Y |
| Random Forest | Y | Y |
| Extra Trees | Y | Y |
| Gradient Boosting (Newton) | Y | Y |
| KNN (KD-tree) | Y | Y |
| Naive Bayes (Gaussian) | Y | — |
| Elastic Net | — | Y |
| AdaBoost (SAMME) | Y | — |
| SVM (linear SMO) | Y | Y |

## Build

Built automatically by the R package's `configure` script. Requires:

- Rust >= 1.87 (`rustup update stable`)
- Cargo

The R package detects Rust availability at install time. If Rust is not found, it falls back to R-native implementations.

## Architecture

- **`ml` (core)**: Pure Rust algorithm implementations. No FFI.
- **`ml-r` (this crate)**: extendr bindings. Translates R vectors to nalgebra matrices, calls core, returns JSON-serialized models.
- **`ml-py`**: PyO3 bindings for the same core. Shared algorithms, different FFI.

All models serialize to JSON for cross-language debuggability.

## License

MIT — [Simon Roth](https://epagogy.ai), 2026.
