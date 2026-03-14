use ml::linear::LinearModel;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Linear Regression (Ridge)
// ---------------------------------------------------------------------------

/// Ridge regression.
///
/// Parameters
/// ----------
/// alpha : float, default 1.0
///     L2 regularization strength.
#[pyclass(name = "Linear")]
pub(crate) struct PyLinear {
    model: LinearModel,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (alpha=1.0))]
    fn new(alpha: f64) -> Self {
        PyLinear {
            model: LinearModel::new(alpha),
        }
    }

    /// Fit ridge regression.
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
                .fit_row_major(x_slice, y_slice, n, p, sw_vec.as_deref())
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
        let v = py.allow_threads(|| self.model.predict_row_major(x_slice, n, p));
        Ok(Array1::from(v).into_pyarray_bound(py))
    }

    /// Fitted coefficients, shape (n_features,).
    #[getter]
    fn coef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let v: Vec<f64> = self.model.coef.iter().copied().collect();
        Array1::from(v).into_pyarray_bound(py)
    }

    /// Fitted intercept (bias).
    #[getter]
    fn intercept(&self) -> f64 {
        self.model.intercept
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
        let model = LinearModel::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyLinear { model })
    }
}
