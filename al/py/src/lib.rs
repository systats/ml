// ml-py: PyO3 bindings for ml crate.
// pyo3 0.22 proc-macros generate error conversions that Rust 1.93 clippy flags
#![allow(clippy::useless_conversion)]
//
// Exposes DecisionTree, RandomForest, ExtraTrees, Linear, Logistic, KNN, GradientBoosting.

use ml::cart::Criterion;
use nalgebra::{DMatrix, DVector};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod adaboost;
mod elastic_net;
mod forest;
mod gbt;
mod knn;
mod linear;
mod logistic;
mod naive_bayes;
mod shuffle;
mod svm;
mod tree;

// ---------------------------------------------------------------------------
// Shared helpers (used across submodules via crate::)
// ---------------------------------------------------------------------------

pub(crate) fn to_dvec(sw: Option<PyReadonlyArray1<f64>>) -> PyResult<Option<DVector<f64>>> {
    match sw {
        None => Ok(None),
        Some(arr) => Ok(Some(DVector::from_column_slice(arr.as_slice()?))),
    }
}

pub(crate) fn dmat_to_flat(m: &DMatrix<f64>) -> (Vec<f64>, usize, usize) {
    let (n, k) = (m.nrows(), m.ncols());
    let mut flat = Vec::with_capacity(n * k);
    for i in 0..n {
        for j in 0..k {
            flat.push(m[(i, j)]);
        }
    }
    (flat, n, k)
}

pub(crate) fn parse_criterion(s: &str) -> PyResult<Criterion> {
    match s {
        "gini" => Ok(Criterion::Gini),
        "entropy" => Ok(Criterion::Entropy),
        "mse" | "squared_error" => Ok(Criterion::MSE),
        "poisson" => Ok(Criterion::Poisson),
        _ => Err(PyValueError::new_err(format!(
            "Unknown criterion '{}'. Valid values: gini, entropy, mse, squared_error, poisson",
            s
        ))),
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// ml_py Python extension module (bridge to ml crate).
#[pymodule]
fn ml_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<tree::PyDecisionTree>()?;
    m.add_class::<forest::PyRandomForest>()?;
    m.add_class::<forest::PyExtraTrees>()?;
    m.add_class::<linear::PyLinear>()?;
    m.add_class::<logistic::PyLogistic>()?;
    m.add_class::<knn::PyKNN>()?;
    m.add_class::<gbt::PyGradientBoosting>()?;
    m.add_class::<naive_bayes::PyNaiveBayes>()?;
    m.add_class::<elastic_net::PyElasticNet>()?;
    m.add_class::<adaboost::PyAdaBoost>()?;
    m.add_class::<svm::PySvmClassifier>()?;
    m.add_class::<svm::PySvmRegressor>()?;
    m.add_function(wrap_pyfunction!(shuffle::shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle::partition_sizes, m)?)?;
    Ok(())
}
