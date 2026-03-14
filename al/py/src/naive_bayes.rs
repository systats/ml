use ml::naive_bayes::NaiveBayesModel;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// NaiveBayes
// ---------------------------------------------------------------------------

/// Gaussian Naive Bayes classifier.
///
/// Parameters
/// ----------
/// var_smoothing : float, default 1e-9
///     Portion of the largest variance of all features added to variances.
#[pyclass(name = "NaiveBayes")]
pub(crate) struct PyNaiveBayes {
    model: NaiveBayesModel,
}

#[pymethods]
impl PyNaiveBayes {
    #[new]
    #[pyo3(signature = (var_smoothing=1e-9))]
    fn new(var_smoothing: f64) -> Self {
        PyNaiveBayes {
            model: NaiveBayesModel::new(var_smoothing),
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

    /// Predict class indices -> 1-D int64 array.
    fn predict_clf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let preds = py.allow_threads(|| self.model.predict_clf(x_slice, n, p));
        Ok(Array1::from(preds).into_pyarray_bound(py))
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
        let model = NaiveBayesModel::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyNaiveBayes { model })
    }
}
