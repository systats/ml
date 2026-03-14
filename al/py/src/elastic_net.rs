use ml::elastic_net::ElasticNetModel;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// ElasticNet
// ---------------------------------------------------------------------------

/// Elastic Net regression (coordinate descent, L1 + L2).
///
/// Parameters
/// ----------
/// alpha : float, default 1.0
///     Regularization strength.
/// l1_ratio : float, default 0.5
///     L1/L2 mix. 0 = Ridge, 1 = Lasso.
/// max_iter : int, default 1000
/// tol : float, default 1e-4
#[pyclass(name = "ElasticNet")]
pub(crate) struct PyElasticNet {
    model: ElasticNetModel,
}

#[pymethods]
impl PyElasticNet {
    #[new]
    #[pyo3(signature = (alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4))]
    fn new(alpha: f64, l1_ratio: f64, max_iter: usize, tol: f64) -> Self {
        PyElasticNet {
            model: ElasticNetModel::new(alpha, l1_ratio, max_iter, tol),
        }
    }

    /// Fit regression. y must be float64.
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
        let sw_vec: Option<Vec<f64>> = sample_weight
            .map(|s| s.as_slice().map(|sl| sl.to_vec()))
            .transpose()?;
        py.allow_threads(|| {
            self.model
                .fit(x_slice, y_slice, n, p, sw_vec.as_deref())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Ok(())
    }

    /// Predict -> 1-D float64 array.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let v = py.allow_threads(|| self.model.predict(x_slice, n, p));
        Ok(Array1::from(v).into_pyarray_bound(py))
    }

    /// Fitted coefficients, shape (n_features,).
    #[getter]
    fn coef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.model.coef.clone()).into_pyarray_bound(py)
    }

    /// Fitted intercept.
    #[getter]
    fn intercept(&self) -> f64 {
        self.model.intercept
    }

    /// Iterations performed.
    #[getter]
    fn n_iter(&self) -> usize {
        self.model.n_iter
    }

    /// Serialize fitted model to JSON string (internal).
    #[pyo3(name = "_to_json")]
    fn to_json(&self) -> PyResult<String> {
        self.model
            .to_json()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Reconstruct fitted model from JSON string (internal).
    #[staticmethod]
    #[pyo3(name = "_from_json")]
    fn from_json(json: &str) -> PyResult<Self> {
        let model = ElasticNetModel::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyElasticNet { model })
    }
}
