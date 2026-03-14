use ml::svm::{KernelType, SvmClassifier, SvmRegressor};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Kernel parsing helper
// ---------------------------------------------------------------------------

/// Parse kernel string + optional params into KernelType.
/// When gamma is None, computes "scale" mode from the training data.
fn parse_kernel(
    kernel: &str,
    gamma: Option<f64>,
    degree: Option<u32>,
    coef0: Option<f64>,
) -> PyResult<KernelType> {
    match kernel {
        "linear" => Ok(KernelType::Linear),
        "rbf" => {
            // gamma will be resolved at fit-time if None (scale mode)
            Ok(KernelType::RBF { gamma: gamma.unwrap_or(0.0) })
        }
        "poly" | "polynomial" => {
            Ok(KernelType::Polynomial {
                gamma: gamma.unwrap_or(0.0),
                coef0: coef0.unwrap_or(0.0),
                degree: degree.unwrap_or(3),
            })
        }
        _ => Err(PyValueError::new_err(format!(
            "Unknown kernel '{}'. Valid values: linear, rbf, poly", kernel
        ))),
    }
}

/// Compute default gamma ("scale" mode): 1 / (n_features * X_variance).
fn compute_scale_gamma(x_slice: &[f64], n: usize, p: usize) -> f64 {
    if n == 0 || p == 0 { return 1.0; }
    let total = (n * p) as f64;
    let mean: f64 = x_slice.iter().sum::<f64>() / total;
    let var: f64 = x_slice.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / total;
    if var < 1e-15 { 1.0 } else { 1.0 / (p as f64 * var) }
}

/// Resolve gamma=0.0 (sentinel for "scale" mode) to actual value.
fn resolve_gamma(kt: &KernelType, x_slice: &[f64], n: usize, p: usize) -> KernelType {
    match kt {
        KernelType::RBF { gamma } if *gamma <= 0.0 => {
            KernelType::RBF { gamma: compute_scale_gamma(x_slice, n, p) }
        }
        KernelType::Polynomial { gamma, coef0, degree } if *gamma <= 0.0 => {
            KernelType::Polynomial {
                gamma: compute_scale_gamma(x_slice, n, p),
                coef0: *coef0,
                degree: *degree,
            }
        }
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// SVM Classifier (OvR, linear/RBF/polynomial kernel, Platt scaling)
// ---------------------------------------------------------------------------

/// SVM classifier (OvR multiclass, Platt probability calibration).
///
/// Parameters
/// ----------
/// c : float, default 1.0
///     Regularization parameter. Smaller = stronger regularization.
/// tol : float, default 1e-3
///     Convergence tolerance for SMO.
/// max_iter : int, default 1000
///     Maximum SMO iterations.
/// kernel : str, default "linear"
///     Kernel type: "linear", "rbf", or "poly".
/// gamma : float or None, default None
///     Kernel coefficient. None = "scale" mode (1 / (n_features * X_variance)).
/// degree : int or None, default 3
///     Degree for polynomial kernel.
/// coef0 : float or None, default 0.0
///     Independent term in polynomial kernel.
#[pyclass(name = "SvmClassifier")]
pub(crate) struct PySvmClassifier {
    model: SvmClassifier,
    gamma_auto: bool,  // true if gamma was None (use "scale" mode at fit time)
}

#[pymethods]
impl PySvmClassifier {
    #[new]
    #[pyo3(signature = (c=1.0, tol=1e-3, max_iter=1000, kernel="linear", gamma=None, degree=None, coef0=None))]
    fn new(
        c: f64, tol: f64, max_iter: usize,
        kernel: &str,
        gamma: Option<f64>,
        degree: Option<u32>,
        coef0: Option<f64>,
    ) -> PyResult<Self> {
        let gamma_auto = gamma.is_none() && kernel != "linear";
        let kt = parse_kernel(kernel, gamma, degree, coef0)?;
        Ok(PySvmClassifier {
            model: SvmClassifier::with_kernel(c, tol, max_iter, kt),
            gamma_auto,
        })
    }

    /// Fit classifier. y must be 0-based integer labels (dtype int64).
    ///
    /// sample_weight : optional 1-D float64 array of per-sample weights.
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<i64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        let sw_vec: Option<Vec<f64>> = match &sample_weight {
            Some(arr) => Some(arr.as_slice()?.to_vec()),
            None => None,
        };
        // Resolve gamma if auto
        if self.gamma_auto {
            self.model.kernel = resolve_gamma(&self.model.kernel, x_slice, n, p);
        }
        py.allow_threads(|| {
            self.model.fit(x_slice, y_slice, n, p, sw_vec.as_deref())
        }).map_err(|e| PyValueError::new_err(e))?;
        Ok(())
    }

    /// Predict class indices → 1-D int64 array.
    fn predict_clf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let v = py.allow_threads(|| self.model.predict(x_slice, n, p));
        Ok(Array1::from(v).into_pyarray_bound(py))
    }

    /// Predict class probabilities → 2-D float64 array (n_samples, n_classes).
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let k = self.model.n_classes;
        let x_slice = x.as_slice()?;
        let flat = py.allow_threads(|| self.model.predict_proba(x_slice, n, p));
        Ok(Array2::from_shape_vec((n, k), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray_bound(py))
    }

    #[getter]
    fn n_classes(&self) -> usize { self.model.n_classes }
    #[getter]
    fn n_features(&self) -> usize { self.model.n_features }

    fn _to_json(&self) -> PyResult<String> {
        self.model.to_json().map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn _from_json(s: &str) -> PyResult<Self> {
        SvmClassifier::from_json(s)
            .map(|model| PySvmClassifier { model, gamma_auto: false })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// SVM Regressor (epsilon-SVR)
// ---------------------------------------------------------------------------

/// Epsilon-SVR with linear/RBF/polynomial kernel.
///
/// Parameters
/// ----------
/// c : float, default 1.0
///     Regularization parameter.
/// epsilon : float, default 0.1
///     Epsilon-tube width (insensitive zone).
/// tol : float, default 1e-3
///     Convergence tolerance.
/// max_iter : int, default 1000
///     Maximum iterations.
/// kernel : str, default "linear"
///     Kernel type: "linear", "rbf", or "poly".
/// gamma : float or None, default None
///     Kernel coefficient. None = "scale" mode.
/// degree : int or None, default 3
///     Degree for polynomial kernel.
/// coef0 : float or None, default 0.0
///     Independent term in polynomial kernel.
#[pyclass(name = "SvmRegressor")]
pub(crate) struct PySvmRegressor {
    model: SvmRegressor,
    gamma_auto: bool,
}

#[pymethods]
impl PySvmRegressor {
    #[new]
    #[pyo3(signature = (c=1.0, epsilon=0.1, tol=1e-3, max_iter=1000, kernel="linear", gamma=None, degree=None, coef0=None))]
    fn new(
        c: f64, epsilon: f64, tol: f64, max_iter: usize,
        kernel: &str,
        gamma: Option<f64>,
        degree: Option<u32>,
        coef0: Option<f64>,
    ) -> PyResult<Self> {
        let gamma_auto = gamma.is_none() && kernel != "linear";
        let kt = parse_kernel(kernel, gamma, degree, coef0)?;
        Ok(PySvmRegressor {
            model: SvmRegressor::with_kernel(c, epsilon, tol, max_iter, kt),
            gamma_auto,
        })
    }

    /// Fit regressor. y must be float64.
    ///
    /// sample_weight : optional 1-D float64 array of per-sample weights.
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        let sw_vec: Option<Vec<f64>> = match &sample_weight {
            Some(arr) => Some(arr.as_slice()?.to_vec()),
            None => None,
        };
        // Resolve gamma if auto
        if self.gamma_auto {
            self.model.kernel = resolve_gamma(&self.model.kernel, x_slice, n, p);
        }
        py.allow_threads(|| {
            self.model.fit(x_slice, y_slice, n, p, sw_vec.as_deref())
        }).map_err(|e| PyValueError::new_err(e))?;
        Ok(())
    }

    /// Predict → 1-D float64 array.
    fn predict_reg<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let v = py.allow_threads(|| self.model.predict(x_slice, n, p));
        Ok(Array1::from(v).into_pyarray_bound(py))
    }

    #[getter]
    fn n_features(&self) -> usize { self.model.n_features }

    fn _to_json(&self) -> PyResult<String> {
        self.model.to_json().map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn _from_json(s: &str) -> PyResult<Self> {
        SvmRegressor::from_json(s)
            .map(|model| PySvmRegressor { model, gamma_auto: false })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
