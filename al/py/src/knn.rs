use ml::knn::{KnnModel, KnnWeights};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// KNN (KD-tree + brute-force hybrid)
// ---------------------------------------------------------------------------

/// Parse weights string to KnnWeights enum.
fn parse_weights(s: &str) -> Result<KnnWeights, PyErr> {
    match s {
        "uniform" => Ok(KnnWeights::Uniform),
        "distance" => Ok(KnnWeights::Distance),
        _ => Err(PyValueError::new_err(format!(
            "Invalid weights '{}'. Must be 'uniform' or 'distance'.", s
        ))),
    }
}

/// K-Nearest Neighbors (KD-tree for d<=20, brute-force otherwise).
///
/// Parameters
/// ----------
/// k : int, default 5
///     Number of neighbors.
/// weights : str, default "uniform"
///     Weight function: "uniform" (all neighbors equal) or
///     "distance" (closer neighbors have more influence).
#[pyclass(name = "KNN")]
pub(crate) struct PyKNN {
    model: KnnModel,
}

#[pymethods]
impl PyKNN {
    #[new]
    #[pyo3(signature = (k=5, weights="uniform"))]
    fn new(k: usize, weights: &str) -> PyResult<Self> {
        let w = parse_weights(weights)?;
        let mut model = KnnModel::new(k);
        model.weights = w;
        Ok(PyKNN { model })
    }

    /// Fit classification. y must be 0-based integer labels (dtype int64).
    fn fit_clf(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<i64>,
    ) -> PyResult<()> {
        let (n, d) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        py.allow_threads(|| {
            self.model
                .fit_clf(x_slice, n, d, y_slice)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Ok(())
    }

    /// Fit regression. y must be float64.
    fn fit_reg(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let (n, d) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        py.allow_threads(|| {
            self.model
                .fit_reg(x_slice, n, d, y_slice)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Ok(())
    }

    /// Classification predict -> 1-D int64 array.
    fn predict_clf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (n, d) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let preds = py.allow_threads(|| self.model.predict_clf(x_slice, n, d));
        Ok(Array1::from(preds).into_pyarray_bound(py))
    }

    /// Regression predict -> 1-D float64 array.
    fn predict_reg<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (n, d) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let preds = py.allow_threads(|| self.model.predict_reg(x_slice, n, d));
        Ok(Array1::from(preds).into_pyarray_bound(py))
    }

    /// Classification predict_proba -> 2-D float64 array, shape (n_samples, n_classes).
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, d) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let nc = self.model.n_classes;
        let flat = py.allow_threads(|| self.model.predict_proba(x_slice, n, d));
        let arr = Array2::from_shape_vec((n, nc), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
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
        let model = KnnModel::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyKNN { model })
    }

    #[getter]
    fn k(&self) -> usize {
        self.model.k
    }

    #[getter]
    fn n_classes(&self) -> usize {
        self.model.n_classes
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.model.n_samples
    }
}
