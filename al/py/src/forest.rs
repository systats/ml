use crate::{dmat_to_flat, parse_criterion, to_dvec};
use ml::cart::ColMajorMatrix;
use ml::forest::RandomForestModel;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Random Forest
// ---------------------------------------------------------------------------

/// Random forest ensemble (classification + regression).
///
/// Parameters
/// ----------
/// n_trees : int, default 100
/// max_depth : int, default 500
/// min_samples_split : int, default 2
/// min_samples_leaf : int, default 1
/// max_features : int or None, default None
///     Number of features per split. None = sqrt(p) for clf, p for reg.
/// histogram_threshold : int, default 4096
/// seed : int, default 42
#[pyclass(name = "RandomForest")]
pub(crate) struct PyRandomForest {
    model: RandomForestModel,
}

#[pymethods]
impl PyRandomForest {
    #[new]
    #[pyo3(signature = (n_trees=100, max_depth=500, min_samples_split=2, min_samples_leaf=1, max_features=None, histogram_threshold=4096, seed=42, compute_oob=true, criterion="gini", monotone_cst=None, min_impurity_decrease=0.0, ccp_alpha=0.0))]
    fn new(
        n_trees: usize,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        histogram_threshold: usize,
        seed: u64,
        compute_oob: bool,
        criterion: &str,
        monotone_cst: Option<Vec<i64>>,
        min_impurity_decrease: f64,
        ccp_alpha: f64,
    ) -> PyResult<Self> {
        let parsed_criterion = parse_criterion(criterion)?;
        let mut model = RandomForestModel::new(n_trees, max_depth, min_samples_split, min_samples_leaf, seed);
        model.max_features = max_features;
        model.histogram_threshold = histogram_threshold;
        model.compute_oob = compute_oob;
        model.criterion = parsed_criterion;
        model.min_impurity_decrease = min_impurity_decrease;
        // ccp_alpha accepted for API consistency but not used by RF (ignored).
        let _ = ccp_alpha;
        if let Some(cst) = monotone_cst {
            model.monotone_cst = Some(cst.iter().map(|&v| v.clamp(-1, 1) as i8).collect());
        }
        Ok(PyRandomForest { model })
    }

    /// Fit classification. y must be 0-based integer labels (dtype int64).
    /// Uses zero-copy path: numpy row-major → ColMajorMatrix directly (no DMatrix).
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
        let sw = to_dvec(sample_weight)?;
        py.allow_threads(|| {
            let cm = ColMajorMatrix::from_row_major_slice(x_slice, n, p);
            self.model
                .fit_clf_prepared(&cm, y_slice, sw.as_ref())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Ok(())
    }

    /// Fit regression. y must be float64.
    /// Uses zero-copy path: numpy row-major → ColMajorMatrix directly (no DMatrix).
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit_reg(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        let sw = to_dvec(sample_weight)?;
        py.allow_threads(|| {
            let cm = ColMajorMatrix::from_row_major_slice(x_slice, n, p);
            let y_vec = DVector::from_column_slice(y_slice);
            self.model
                .fit_reg_prepared(&cm, &y_vec, sw.as_ref())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Ok(())
    }

    /// Classification predict -> 1-D int64 array of class indices.
    fn predict_clf<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let result = py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            self.model.predict_clf(&x_mat)
        });
        Ok(Array1::from(result).into_pyarray_bound(py))
    }

    /// Regression predict -> 1-D float64 array.
    fn predict_reg<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let v: Vec<f64> = py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            self.model.predict_reg(&x_mat).data.into()
        });
        Ok(Array1::from(v).into_pyarray_bound(py))
    }

    /// Classification predict_proba -> 2-D float64 array, shape (n_samples, n_classes).
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let (flat, rows, cols) = py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            let proba = self.model.predict_proba(&x_mat);
            dmat_to_flat(&proba)
        });
        let arr = Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray_bound(py))
    }

    // ----- serialization (internal — not part of public Python API) -----

    /// Serialize fitted forest to JSON string.
    #[pyo3(name = "_to_json")]
    fn to_json(&self) -> PyResult<String> {
        self.model
            .to_json()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Reconstruct fitted forest from JSON string.
    #[staticmethod]
    #[pyo3(name = "_from_json")]
    fn from_json(json: &str) -> PyResult<Self> {
        let model = RandomForestModel::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyRandomForest { model })
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.model.n_features
    }

    #[getter]
    fn n_classes(&self) -> usize {
        self.model.n_classes
    }

    #[getter]
    fn feature_importances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.model.feature_importances.clone()).into_pyarray_bound(py)
    }

    #[getter]
    fn oob_score(&self) -> Option<f64> {
        self.model.oob_score
    }
}

// ---------------------------------------------------------------------------
// Extra Trees (Random Forest variant with random thresholds, no bootstrap)
// ---------------------------------------------------------------------------

/// Extra Trees ensemble (Geurts et al. 2006).
///
/// Same interface as RandomForest but uses random thresholds per feature
/// (no optimal split scan) and no bootstrap (full dataset per tree).
#[pyclass(name = "ExtraTrees")]
pub(crate) struct PyExtraTrees {
    model: RandomForestModel,
}

#[pymethods]
impl PyExtraTrees {
    #[new]
    #[pyo3(signature = (n_trees=100, max_depth=500, min_samples_split=2, min_samples_leaf=1, max_features=None, histogram_threshold=4096, seed=42, compute_oob=false, criterion="gini", monotone_cst=None, min_impurity_decrease=0.0, ccp_alpha=0.0))]
    fn new(
        n_trees: usize,
        max_depth: usize,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        histogram_threshold: usize,
        seed: u64,
        compute_oob: bool,
        criterion: &str,
        monotone_cst: Option<Vec<i64>>,
        min_impurity_decrease: f64,
        ccp_alpha: f64,
    ) -> PyResult<Self> {
        let parsed_criterion = parse_criterion(criterion)?;
        let mut model = RandomForestModel::new(n_trees, max_depth, min_samples_split, min_samples_leaf, seed);
        model.max_features = max_features;
        model.histogram_threshold = histogram_threshold;
        model.compute_oob = compute_oob;
        model.criterion = parsed_criterion;
        model.extra_trees = true;
        model.min_impurity_decrease = min_impurity_decrease;
        // ccp_alpha accepted for API consistency but not used by ExtraTrees (ignored).
        let _ = ccp_alpha;
        if let Some(cst) = monotone_cst {
            model.monotone_cst = Some(cst.iter().map(|&v| v.clamp(-1, 1) as i8).collect());
        }
        Ok(PyExtraTrees { model })
    }

    /// Fit classification. y must be 0-based integer labels (dtype int64).
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
        let sw = to_dvec(sample_weight)?;
        py.allow_threads(|| {
            let cm = ColMajorMatrix::from_row_major_slice(x_slice, n, p);
            self.model
                .fit_clf_prepared(&cm, y_slice, sw.as_ref())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
        Ok(())
    }

    /// Fit regression. y must be float64.
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit_reg(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        let sw = to_dvec(sample_weight)?;
        py.allow_threads(|| {
            let cm = ColMajorMatrix::from_row_major_slice(x_slice, n, p);
            let y_vec = DVector::from_column_slice(y_slice);
            self.model
                .fit_reg_prepared(&cm, &y_vec, sw.as_ref())
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
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let result = py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            self.model.predict_clf(&x_mat)
        });
        Ok(Array1::from(result).into_pyarray_bound(py))
    }

    /// Regression predict -> 1-D float64 array.
    fn predict_reg<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let v: Vec<f64> = py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            self.model.predict_reg(&x_mat).data.into()
        });
        Ok(Array1::from(v).into_pyarray_bound(py))
    }

    /// Classification predict_proba -> 2-D float64 array, shape (n_samples, n_classes).
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (n, p) = (x.shape()[0], x.shape()[1]);
        let x_slice = x.as_slice()?;
        let (flat, rows, cols) = py.allow_threads(|| {
            let x_mat = DMatrix::from_row_slice(n, p, x_slice);
            let proba = self.model.predict_proba(&x_mat);
            dmat_to_flat(&proba)
        });
        let arr = Array2::from_shape_vec((rows, cols), flat)
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
        let model = RandomForestModel::from_json(json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyExtraTrees { model })
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.model.n_features
    }

    #[getter]
    fn n_classes(&self) -> usize {
        self.model.n_classes
    }

    #[getter]
    fn feature_importances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.model.feature_importances.clone()).into_pyarray_bound(py)
    }

    #[getter]
    fn oob_score(&self) -> Option<f64> {
        self.model.oob_score
    }
}
