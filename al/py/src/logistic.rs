use ml::logistic::{LogisticModel, MultiClass};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Logistic Regression (OvR + L-BFGS)
// ---------------------------------------------------------------------------

/// Logistic regression (One-vs-Rest, L2 regularization).
///
/// Parameters
/// ----------
/// c : float, default 1.0
///     Inverse regularization strength (higher = less regularization).
/// max_iter : int, default 1000
///     Maximum L-BFGS iterations.
#[pyclass(name = "Logistic")]
pub(crate) struct PyLogistic {
    model: LogisticModel,
}

#[pymethods]
impl PyLogistic {
    #[new]
    #[pyo3(signature = (c=1.0, max_iter=1000, multi_class="ovr"))]
    fn new(c: f64, max_iter: usize, multi_class: &str) -> PyResult<Self> {
        let mc = match multi_class {
            "ovr" | "OvR" => MultiClass::OvR,
            "softmax" | "multinomial" => MultiClass::Softmax,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown multi_class '{}'. Valid values: ovr, softmax",
                    multi_class
                )))
            }
        };
        let mut model = LogisticModel::new(c, max_iter);
        model.multi_class = mc;
        Ok(PyLogistic { model })
    }

    /// Fit logistic regression. y must be 0-based integer labels (dtype int64).
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

    /// Predict class indices -> 1-D int64 array.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let v = py.allow_threads(|| self.model.predict_row_major(x_slice, n, p));
        Ok(Array1::from(v).into_pyarray_bound(py))
    }

    /// Predict probabilities -> 2-D float64 array, shape (n_samples, n_classes).
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let k = self.model.n_classes;
        let flat = py.allow_threads(|| self.model.predict_proba_row_major(x_slice, n, p));
        let arr = Array2::from_shape_vec((n, k), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    /// Number of classes.
    #[getter]
    fn n_classes(&self) -> usize {
        self.model.n_classes
    }

    /// OvR coefficient vectors. List of 1-D arrays, one per classifier.
    /// Each vector = [bias, w1, w2, ...] (bias at index 0).
    #[getter]
    fn coefs<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<f64>>> {
        self.model
            .coefs
            .iter()
            .map(|c| {
                let v: Vec<f64> = c.iter().copied().collect();
                Array1::from(v).into_pyarray_bound(py)
            })
            .collect()
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
        let model = LogisticModel::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyLogistic { model })
    }
}
