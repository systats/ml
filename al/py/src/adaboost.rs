use ml::adaboost::AdaBoostModel;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// AdaBoost
// ---------------------------------------------------------------------------

/// AdaBoost classifier (SAMME — discrete multi-class).
///
/// Parameters
/// ----------
/// n_estimators : int, default 50
///     Number of boosting rounds.
/// learning_rate : float, default 1.0
///     Shrinkage applied to each stump's alpha.
/// seed : int, default 42
///     Random seed for stump splitting.
#[pyclass(name = "AdaBoost")]
pub(crate) struct PyAdaBoost {
    model: AdaBoostModel,
}

#[pymethods]
impl PyAdaBoost {
    #[new]
    #[pyo3(signature = (n_estimators=50, learning_rate=1.0, seed=42))]
    fn new(n_estimators: usize, learning_rate: f64, seed: u64) -> Self {
        PyAdaBoost {
            model: AdaBoostModel::new(n_estimators, learning_rate, seed),
        }
    }

    /// Fit classifier. y must be 0-based integer labels (dtype int64).
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit_clf(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<i64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let y_raw = y.as_slice()?;
        // Convert i64 → usize (0-based labels)
        let y_usize: Vec<usize> = y_raw
            .iter()
            .map(|&v| {
                if v < 0 {
                    Err(PyValueError::new_err("AdaBoost labels must be non-negative"))
                } else {
                    Ok(v as usize)
                }
            })
            .collect::<PyResult<Vec<usize>>>()?;
        let sw_vec: Option<Vec<f64>> = sample_weight
            .map(|s| s.as_slice().map(|sl| sl.to_vec()))
            .transpose()?;
        py.allow_threads(|| {
            self.model
                .fit(x_slice, &y_usize, n, p, sw_vec.as_deref())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Ok(())
    }

    /// Predict class indices -> 1-D int64 array.
    fn predict_clf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let preds = py.allow_threads(|| self.model.predict(x_slice, n, p));
        let preds_i64: Vec<i64> = preds.iter().map(|&v| v as i64).collect();
        Ok(Array1::from(preds_i64).into_pyarray_bound(py))
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
        let flat = py.allow_threads(|| self.model.predict_proba(x_slice, n, p));
        let arr = Array2::from_shape_vec((n, k), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    /// Number of classes.
    #[getter]
    fn n_classes(&self) -> usize {
        self.model.n_classes
    }

    /// Number of features.
    #[getter]
    fn n_features(&self) -> usize {
        self.model.n_features
    }

    /// Feature importances (alpha-weighted MDI, normalized to sum=1).
    #[getter]
    fn feature_importances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.model.feature_importances.clone()).into_pyarray_bound(py)
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
        let model = AdaBoostModel::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyAdaBoost { model })
    }
}
