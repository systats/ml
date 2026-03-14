//! Logistic regression (One-vs-Rest) via L-BFGS.
//!
//! Port of `ml._logistic._LogisticModel` (Python reference implementation).
//!
//! Regularization uses the sklearn C convention:
//!   loss = -avg_log_loss + 0.5 / (C * n) * ||w||^2
//! where n = training set size and w excludes the bias term.
//!
//! Binary classification: 1 L-BFGS call.
//! Multiclass (K classes): K independent OvR L-BFGS calls.
//!
//! # v1 contract
//! `predict()` returns 0-based class indices, NOT original labels.
//! Label remapping (e.g. original labels {-1, 1} or {"cat", "dog"}) is the
//! responsibility of the Python/R bridge layer (v2/v3).

use crate::blas;
use crate::error::MlError;
use crate::lbfgs::lbfgs;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const LBFGS_M: usize = 10;
const LBFGS_TOL: f64 = 1e-4;

/// Multi-class strategy for logistic regression.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MultiClass {
    /// One-vs-Rest: k independent binary classifiers (default).
    #[default]
    OvR,
    /// Joint multinomial softmax via single L-BFGS call.
    Softmax,
}

/// Logistic regression model (OvR or Softmax, L2 regularization).
#[derive(Serialize, Deserialize)]
pub struct LogisticModel {
    /// Inverse regularization strength (higher = less regularization).
    pub c: f64,
    /// Maximum number of L-BFGS iterations.
    pub max_iter: usize,
    /// Fitted coefficients.  coefs[k] = DVector of length (n_features + 1):
    /// `coefs[k][0]` = bias, `coefs[k][1..]` = feature weights for class k.
    pub coefs: Vec<DVector<f64>>,
    /// Number of distinct classes seen during fit.
    pub n_classes: usize,
    /// Multi-class strategy. Old models without this field deserialize as OvR.
    #[serde(default)]
    pub multi_class: MultiClass,
}

impl LogisticModel {
    /// Create a new, unfitted `LogisticModel`.
    pub fn new(c: f64, max_iter: usize) -> Self {
        Self {
            c,
            max_iter,
            coefs: Vec::new(),
            n_classes: 0,
            multi_class: MultiClass::default(),
        }
    }

    /// Fit logistic regression on `(x, y)`.
    ///
    /// # Arguments
    /// - `x`: `(n_samples, n_features)` feature matrix.
    /// - `y`: integer class labels (values can be any i64).
    /// - `sample_weight`: optional per-sample weights (unnormalized).
    ///
    /// # Errors
    /// - `EmptyData` if `x.nrows() == 0`.
    /// - `DimensionMismatch` if `y.len() != x.nrows()` or fewer than 2 unique classes.
    pub fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &[i64],
        sample_weight: Option<&DVector<f64>>,
    ) -> Result<&mut Self, MlError> {
        let n = x.nrows();
        let p = x.ncols();

        if n == 0 {
            return Err(MlError::EmptyData);
        }
        if y.len() != n {
            return Err(MlError::DimensionMismatch {
                expected: n,
                got: y.len(),
            });
        }

        // Collect sorted unique classes
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let k = classes.len();
        if k < 2 {
            return Err(MlError::DimensionMismatch {
                expected: 2,
                got: k,
            });
        }

        self.n_classes = k;
        let c = self.c;
        let max_iter = self.max_iter;

        // Train OvR classifiers in parallel.
        // Binary (k=2): 1 classifier for the positive class (classes[1]).
        // Multiclass: k classifiers, one per class — parallel via rayon.
        let pos_classes: Vec<i64> = if k == 2 {
            vec![classes[1]]
        } else {
            classes.clone()
        };

        self.coefs = pos_classes
            .par_iter()
            .map(|&pos_class| {
                let y_bin: Vec<f64> = y
                    .iter()
                    .map(|&label| if label == pos_class { 1.0 } else { 0.0 })
                    .collect();
                let w0 = DVector::zeros(p + 1);
                lbfgs(
                    |w| binary_loss_grad(w, x, &y_bin, c, sample_weight),
                    w0,
                    LBFGS_M,
                    max_iter,
                    LBFGS_TOL,
                )
            })
            .collect();

        Ok(self)
    }

    /// Fit from row-major slice — avoids DMatrix construction for PyO3 path.
    pub fn fit_row_major(
        &mut self,
        x_rm: &[f64],
        y: &[i64],
        n: usize,
        p: usize,
        sw: Option<&[f64]>,
    ) -> Result<&mut Self, MlError> {
        if n == 0 {
            return Err(MlError::EmptyData);
        }
        if y.len() != n {
            return Err(MlError::DimensionMismatch {
                expected: n,
                got: y.len(),
            });
        }

        let mut classes: Vec<i64> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let k = classes.len();
        if k < 2 {
            return Err(MlError::DimensionMismatch {
                expected: 2,
                got: k,
            });
        }

        self.n_classes = k;
        let c = self.c;
        let max_iter = self.max_iter;

        // Precompute normalized weights once.
        let sw_norm: Option<Vec<f64>> = sw.map(|weights| {
            let sw_sum: f64 = weights.iter().sum();
            let scale = n as f64 / sw_sum;
            weights.iter().map(|v| v * scale).collect()
        });

        if self.multi_class == MultiClass::Softmax {
            // Joint multinomial softmax via single L-BFGS call over k*(p+1) weights.
            // y_enc[i] = 0-based class index of sample i
            let y_enc: Vec<usize> = y
                .iter()
                .map(|&label| classes.binary_search(&label).unwrap())
                .collect();
            let w0 = DVector::zeros(k * (p + 1));
            let coef_flat = lbfgs(
                |w| softmax_loss_grad_rm(w, x_rm, &y_enc, sw_norm.as_deref(), c, n, p, k),
                w0,
                LBFGS_M,
                max_iter,
                LBFGS_TOL,
            );
            // Split flat vector into k DVector entries (one per class, bias-first)
            self.coefs = (0..k)
                .map(|ki| {
                    let start = ki * (p + 1);
                    DVector::from_column_slice(&coef_flat.as_slice()[start..start + (p + 1)])
                })
                .collect();
        } else {
            // OvR: k independent binary L-BFGS calls — parallel via rayon.
            let pos_classes: Vec<i64> = if k == 2 {
                vec![classes[1]]
            } else {
                classes.clone()
            };

            // Each parallel task gets its own scratch buffers.
            // Buffers are reused across L-BFGS iterations within a single
            // OvR classifier, avoiding ~16MB allocation per gradient call.
            self.coefs = pos_classes
                .par_iter()
                .map(|&pos_class| {
                    let y_bin: Vec<f64> = y
                        .iter()
                        .map(|&label| if label == pos_class { 1.0 } else { 0.0 })
                        .collect();
                    let w0 = DVector::zeros(p + 1);
                    let mut z_buf = vec![0.0_f64; n];
                    let mut err_buf = vec![0.0_f64; n];
                    let mut gw_buf = vec![0.0_f64; p];
                    lbfgs(
                        |w| binary_loss_grad_blas_reuse(
                            w, x_rm, &y_bin, n, p, c, sw_norm.as_deref(),
                            &mut z_buf, &mut err_buf, &mut gw_buf,
                        ),
                        w0,
                        LBFGS_M,
                        max_iter,
                        LBFGS_TOL,
                    )
                })
                .collect();
        }

        Ok(self)
    }

    /// Predict from row-major slice — avoids DMatrix construction.
    pub fn predict_row_major(&self, x_rm: &[f64], n: usize, p: usize) -> Vec<i64> {
        let proba = self.predict_proba_row_major(x_rm, n, p);
        let k = self.n_classes;
        (0..n)
            .map(|i| {
                let row = &proba[i * k..(i + 1) * k];
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Predict probabilities from row-major slice, returns flat (n*k) row-major.
    ///
    /// All probabilities are clipped to `[EPS, 1-EPS]` where `EPS = 1e-15`
    /// to prevent exact 0.0 or 1.0 which cause `log(0) = -inf` in downstream
    /// metrics (log-loss, cross-entropy).
    pub fn predict_proba_row_major(&self, x_rm: &[f64], n: usize, p: usize) -> Vec<f64> {
        let k = self.n_classes;

        let mut out = if self.multi_class == MultiClass::Softmax {
            // Softmax: compute linear scores for all k classes, apply row softmax.
            // self.coefs has k entries (one per class), each length (p+1).
            let mut out = vec![0.0_f64; n * k];
            for (ki, w) in self.coefs.iter().enumerate() {
                let bias = w[0];
                let weights = w.rows(1, p);
                for i in 0..n {
                    let row = &x_rm[i * p..(i + 1) * p];
                    let mut z: f64 = bias;
                    for j in 0..p {
                        z += row[j] * weights[j];
                    }
                    out[i * k + ki] = z;
                }
            }
            // Row-max stabilized softmax
            for i in 0..n {
                let row = &mut out[i * k..(i + 1) * k];
                let max_z = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum_exp = 0.0_f64;
                for v in row.iter_mut() {
                    *v = (*v - max_z).exp();
                    sum_exp += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum_exp;
                }
            }
            out
        } else if k == 2 {
            // OvR binary: 1 coef, sigmoid. Use BLAS GEMV for large n.
            let w = &self.coefs[0];
            let bias = w[0];
            let weights = w.rows(1, p);
            let mut out = vec![0.0_f64; n * 2];
            if n >= 32 {
                // BLAS: z = X(n×p) * weights(p)
                let weights_slice: Vec<f64> = weights.iter().copied().collect();
                let mut z_buf = vec![0.0_f64; n];
                blas::gemv_rm(x_rm, &weights_slice, &mut z_buf, n, p);
                for i in 0..n {
                    let z = (z_buf[i] + bias).clamp(-500.0, 500.0);
                    let p1 = 1.0 / (1.0 + (-z).exp());
                    out[i * 2] = 1.0 - p1;
                    out[i * 2 + 1] = p1;
                }
            } else {
                for i in 0..n {
                    let row = &x_rm[i * p..(i + 1) * p];
                    let mut z: f64 = bias;
                    for j in 0..p {
                        z += row[j] * weights[j];
                    }
                    z = z.clamp(-500.0, 500.0);
                    let p1 = 1.0 / (1.0 + (-z).exp());
                    out[i * 2] = 1.0 - p1;
                    out[i * 2 + 1] = p1;
                }
            }
            out
        } else {
            // OvR multiclass: k coefs, sigmoid + row normalize
            let mut out = vec![0.0_f64; n * k];
            for (ki, w) in self.coefs.iter().enumerate() {
                let bias = w[0];
                let weights = w.rows(1, p);
                for i in 0..n {
                    let row = &x_rm[i * p..(i + 1) * p];
                    let mut z: f64 = bias;
                    for j in 0..p {
                        z += row[j] * weights[j];
                    }
                    z = z.clamp(-500.0, 500.0);
                    out[i * k + ki] = 1.0 / (1.0 + (-z).exp());
                }
            }
            // Row-normalize
            for i in 0..n {
                let row = &mut out[i * k..(i + 1) * k];
                let sum: f64 = row.iter().sum();
                let safe_sum = if sum == 0.0 { 1.0 } else { sum };
                for v in row.iter_mut() {
                    *v /= safe_sum;
                }
            }
            out
        };

        // Clip all probabilities to [EPS, 1-EPS] to prevent log(0) = -inf
        clip_proba(&mut out);
        out
    }

    /// Predict class indices (0-based) for each row of `x`.
    pub fn predict(&self, x: &DMatrix<f64>) -> Vec<i64> {
        let proba = self.predict_proba(x);
        (0..proba.nrows())
            .map(|i| {
                let row = proba.row(i);
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Predict class probabilities, shape `(n_samples, n_classes)`.
    ///
    /// Rows sum to 1.0.  Probabilities are based on sigmoid scores normalized
    /// across classes (OvR softmax-style normalization).
    ///
    /// All probabilities are clipped to `[EPS, 1-EPS]` where `EPS = 1e-15`
    /// to prevent exact 0.0 or 1.0 which cause `log(0) = -inf` in downstream
    /// metrics (log-loss, cross-entropy).
    pub fn predict_proba(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let n = x.nrows();
        let k = self.n_classes;

        let mut result = if k == 2 {
            // Binary: 2 columns [P(class 0), P(class 1)]
            let w = &self.coefs[0];
            let (p1, _) = sigmoid_scores(x, w);
            DMatrix::from_fn(n, 2, |i, j| if j == 0 { 1.0 - p1[i] } else { p1[i] })
        } else {
            // Multiclass: k columns, each column = P(class k) via OvR sigmoid
            let mut raw = DMatrix::zeros(n, k);
            for (ki, w) in self.coefs.iter().enumerate() {
                let (scores, _) = sigmoid_scores(x, w);
                for i in 0..n {
                    raw[(i, ki)] = scores[i];
                }
            }
            // Row-normalize: guard against all-zero rows (degenerate input)
            DMatrix::from_fn(n, k, |i, j| {
                let row_sum: f64 = (0..k).map(|kk| raw[(i, kk)]).sum();
                let safe_sum = if row_sum == 0.0 { 1.0 } else { row_sum };
                raw[(i, j)] / safe_sum
            })
        };

        // Clip all probabilities to [EPS, 1-EPS] to prevent log(0) = -inf
        for v in result.iter_mut() {
            *v = v.clamp(PROBA_EPS, 1.0 - PROBA_EPS);
        }
        result
    }

    /// Serialize to JSON for Python `__getstate__`.
    pub fn to_json(&self) -> Result<String, MlError> {
        serde_json::to_string(self).map_err(|_| MlError::EmptyData)
    }

    /// Deserialize from JSON for Python `__setstate__`.
    pub fn from_json(json: &str) -> Result<Self, MlError> {
        serde_json::from_str(json).map_err(|_| MlError::EmptyData)
    }
}

impl Default for LogisticModel {
    fn default() -> Self {
        Self::new(1.0, 1000)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Epsilon for probability clipping: prevents exact 0.0 or 1.0 in
/// predict_proba output, which would cause log(0) = -inf in downstream
/// metrics (log-loss, cross-entropy, KL divergence).
const PROBA_EPS: f64 = 1e-15;

/// Clip a flat probability slice to [EPS, 1-EPS] in place.
#[inline]
fn clip_proba(proba: &mut [f64]) {
    for v in proba.iter_mut() {
        *v = v.clamp(PROBA_EPS, 1.0 - PROBA_EPS);
    }
}

/// Compute sigmoid scores for a weight vector `w = [bias, w1..wp]`.
///
/// Returns `(p, z)` where:
/// - `p[i] = sigmoid(z[i])` (clipped to [1e-15, 1-1e-15] for numerical safety)
/// - `z[i] = x[i] · w[1:] + w[0]` (clipped to [-500, 500])
fn sigmoid_scores(x: &DMatrix<f64>, w: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
    let p = x.ncols();
    let bias = w[0];
    let weights = w.rows(1, p).into_owned();
    let z = (x * &weights).map(|v| v + bias);
    let z = z.map(|v| v.clamp(-500.0, 500.0));
    let p_raw = z.map(|v| 1.0 / (1.0 + (-v).exp()));
    (p_raw, z)
}

/// Multinomial (softmax) cross-entropy loss + gradient.
///
/// # Layout
/// - `w`: flat k*(p+1) weights, class-major:
///   `[bias_0, w0_1..w0_p, bias_1, w1_1..w1_p, ...]`
/// - `x_rm`: n*p row-major feature matrix (no intercept column)
/// - `y_enc`: 0-based class index per sample, length n
/// - `sw_norm`: optional pre-normalized sample weights (sum = n)
///
/// # Numerical safety
/// Row-max subtracted before `exp()` to prevent overflow in softmax.
fn softmax_loss_grad_rm(
    w: &DVector<f64>,
    x_rm: &[f64],
    y_enc: &[usize],
    sw_norm: Option<&[f64]>,
    c: f64,
    n: usize,
    p: usize,
    k: usize,
) -> (f64, DVector<f64>) {
    let n_f = n as f64;

    // Compute linear scores: z[i*k + ki] = bias_ki + dot(x_i, w_ki)
    let mut z = vec![0.0_f64; n * k];
    for ki in 0..k {
        let base = ki * (p + 1);
        let bias = w[base];
        for i in 0..n {
            let row = &x_rm[i * p..(i + 1) * p];
            let mut score = bias;
            for j in 0..p {
                score += row[j] * w[base + 1 + j];
            }
            z[i * k + ki] = score;
        }
    }

    // Row-max stabilized softmax in-place → z becomes s (softmax proba)
    for i in 0..n {
        let row = &mut z[i * k..(i + 1) * k];
        let max_z = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum_exp = 0.0_f64;
        for v in row.iter_mut() {
            *v = (*v - max_z).exp();
            sum_exp += *v;
        }
        for v in row.iter_mut() {
            *v /= sum_exp;
        }
    }
    let s = z; // renamed for clarity: s[i*k+ki] = softmax prob of class ki for sample i

    // Cross-entropy loss: -sum_i sw_i * log(s[i, y_enc[i]]) / n
    let ce_loss: f64 = if let Some(sw) = sw_norm {
        (0..n)
            .map(|i| {
                let p_yi = s[i * k + y_enc[i]].max(1e-15);
                sw[i] * (-p_yi.ln())
            })
            .sum::<f64>()
            / n_f
    } else {
        (0..n)
            .map(|i| {
                let p_yi = s[i * k + y_enc[i]].max(1e-15);
                -p_yi.ln()
            })
            .sum::<f64>()
            / n_f
    };

    // L2 regularization (bias excluded): (1/(2*C*n)) * sum_k ||w_k[1:]||^2
    let mut reg = 0.0_f64;
    for ki in 0..k {
        let base = ki * (p + 1);
        for j in 1..=p {
            reg += w[base + j] * w[base + j];
        }
    }
    reg *= 0.5 / (c * n_f);

    let total_loss = ce_loss + reg;

    // Gradient: for class ki, residual[i] = s[i,ki] - 1{y_enc[i]==ki}
    // g_bias_ki = (1/n) * sum_i sw_i * residual[i]
    // g_w_ki[j] = (1/n) * sum_i sw_i * residual[i] * x_ij + w_ki[j] / (C*n)
    let mut grad = DVector::zeros(k * (p + 1));
    for ki in 0..k {
        let base = ki * (p + 1);
        let mut g_bias = 0.0_f64;
        let mut gw = vec![0.0_f64; p];
        for i in 0..n {
            let sw_i = sw_norm.map_or(1.0, |sw| sw[i]);
            let residual = s[i * k + ki] - if y_enc[i] == ki { 1.0 } else { 0.0 };
            let r = sw_i * residual;
            g_bias += r;
            let row = &x_rm[i * p..(i + 1) * p];
            for j in 0..p {
                gw[j] += r * row[j];
            }
        }
        grad[base] = g_bias / n_f;
        for j in 0..p {
            grad[base + 1 + j] = gw[j] / n_f + w[base + 1 + j] / (c * n_f);
        }
    }

    (total_loss, grad)
}

/// Binary cross-entropy loss + gradient for a single OvR classifier.
///
/// # Layout
/// `w[0]` = bias, `w[1..]` = feature weights (bias-first, length `p+1`).
///
/// # Regularization
/// L2 penalty: `0.5 / (C * n) * ||w[1:]||^2` (sklearn C convention; bias excluded).
/// Normalization is per-sample (`/n`), matching sklearn's `C` interpretation exactly.
///
/// # Numerical safety
/// - `z` clipped to [-500, 500]: prevents `exp` overflow in sigmoid.
/// - `p_safe` clipped to [1e-15, 1-1e-15]: prevents `log(0)` in cross-entropy.
fn binary_loss_grad(
    w: &DVector<f64>,
    x: &DMatrix<f64>,
    y_bin: &[f64],
    c: f64,
    sample_weight: Option<&DVector<f64>>,
) -> (f64, DVector<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let bias = w[0];
    let weights = w.rows(1, p).into_owned();

    // z = X @ w[1:] + w[0], clipped to prevent exp overflow
    let z = (x * &weights).map(|v| v + bias);
    let z = z.map(|v| v.clamp(-500.0, 500.0));

    // p_raw used for gradient (unclipped for p), p_safe for log (clipped)
    let p_raw = z.map(|v| 1.0 / (1.0 + (-v).exp()));
    let p_safe = p_raw.map(|v| v.clamp(1e-15, 1.0 - 1e-15));

    // Loss and residuals
    let (ce_loss, err) = if let Some(sw) = sample_weight {
        // Normalize weights to sum = n (matches Python: sw / sw.sum() * n)
        let sw_sum: f64 = sw.iter().sum();
        let sw_norm = sw.map(|v| v * n as f64 / sw_sum);
        let loss = -(0..n)
            .map(|i| {
                sw_norm[i] * (y_bin[i] * p_safe[i].ln() + (1.0 - y_bin[i]) * (1.0 - p_safe[i]).ln())
            })
            .sum::<f64>()
            / n as f64;
        let err = DVector::from_fn(n, |i, _| sw_norm[i] * (p_raw[i] - y_bin[i]));
        (loss, err)
    } else {
        let loss = -(0..n)
            .map(|i| y_bin[i] * p_safe[i].ln() + (1.0 - y_bin[i]) * (1.0 - p_safe[i]).ln())
            .sum::<f64>()
            / n as f64;
        let err = DVector::from_fn(n, |i, _| p_raw[i] - y_bin[i]);
        (loss, err)
    };

    // L2 regularization: 0.5 / (C * n) * ||w[1:]||^2  (bias excluded)
    let reg = 0.5 / (c * n as f64) * weights.dot(&weights);
    let total_loss = ce_loss + reg;

    // Gradient
    let n_f = n as f64;
    let g_bias = err.sum() / n_f;
    let g_weights = x.tr_mul(&err) / n_f + weights.scale(1.0 / (c * n_f));

    let mut grad = DVector::zeros(p + 1);
    grad[0] = g_bias;
    for i in 0..p {
        grad[i + 1] = g_weights[i];
    }

    (total_loss, grad)
}

/// BLAS-accelerated binary loss + gradient with pre-allocated buffers.
///
/// Uses dgemv for X@w (forward) and X^T@err (gradient), with fused
/// scalar loop for sigmoid/loss/error in between.
/// Buffers `z_buf(n)`, `err_buf(n)`, `gw_buf(p)` are reused across L-BFGS iterations
/// to avoid 16MB+ allocation per call at large n.
fn binary_loss_grad_blas_reuse(
    w: &DVector<f64>,
    x_rm: &[f64],
    y_bin: &[f64],
    n: usize,
    p: usize,
    c: f64,
    sw_norm: Option<&[f64]>,
    z_buf: &mut [f64],
    err_buf: &mut [f64],
    gw_buf: &mut [f64],
) -> (f64, DVector<f64>) {
    let bias = w[0];
    let wt: Vec<f64> = (1..=p).map(|i| w[i]).collect();
    let n_f = n as f64;

    // Forward pass via BLAS dgemv: z = X @ wt
    blas::gemv_rm(x_rm, &wt, z_buf, n, p);

    // Sigmoid + loss + error: parallel chunks when n > 100K.
    let z_ref: &[f64] = z_buf; // immutable re-borrow for parallel read
    let (ce_loss, g_bias) = if n > 100_000 {
        const CHUNK: usize = 65536;
        let result: (f64, f64) = err_buf
            .par_chunks_mut(CHUNK)
            .enumerate()
            .map(|(ci, err_chunk)| {
                let start = ci * CHUNK;
                let mut loss_acc = 0.0_f64;
                let mut gbias_acc = 0.0_f64;
                for (li, err_val) in err_chunk.iter_mut().enumerate() {
                    let i = start + li;
                    if i >= n { break; }
                    let zi = (z_ref[i] + bias).clamp(-500.0, 500.0);
                    let p_raw = 1.0 / (1.0 + (-zi).exp());
                    let p_safe = p_raw.clamp(1e-15, 1.0 - 1e-15);
                    let yi = y_bin[i];
                    let (loss_i, err_i) = if let Some(sw) = sw_norm {
                        let wi = sw[i];
                        (
                            -wi * (yi * p_safe.ln() + (1.0 - yi) * (1.0 - p_safe).ln()),
                            wi * (p_raw - yi),
                        )
                    } else {
                        (
                            -(yi * p_safe.ln() + (1.0 - yi) * (1.0 - p_safe).ln()),
                            p_raw - yi,
                        )
                    };
                    loss_acc += loss_i;
                    gbias_acc += err_i;
                    *err_val = err_i;
                }
                (loss_acc, gbias_acc)
            })
            .reduce(|| (0.0, 0.0), |(l1, g1), (l2, g2)| (l1 + l2, g1 + g2));
        (result.0 / n_f, result.1)
    } else {
        let mut loss = 0.0_f64;
        let mut gbias = 0.0_f64;
        for i in 0..n {
            let zi = (z_ref[i] + bias).clamp(-500.0, 500.0);
            let p_raw = 1.0 / (1.0 + (-zi).exp());
            let p_safe = p_raw.clamp(1e-15, 1.0 - 1e-15);
            let yi = y_bin[i];
            let (loss_i, err_i) = if let Some(sw) = sw_norm {
                let wi = sw[i];
                (
                    -wi * (yi * p_safe.ln() + (1.0 - yi) * (1.0 - p_safe).ln()),
                    wi * (p_raw - yi),
                )
            } else {
                (
                    -(yi * p_safe.ln() + (1.0 - yi) * (1.0 - p_safe).ln()),
                    p_raw - yi,
                )
            };
            loss += loss_i;
            gbias += err_i;
            err_buf[i] = err_i;
        }
        (loss / n_f, gbias)
    };

    // L2 reg
    let w_sq: f64 = wt.iter().map(|v| v * v).sum();
    let reg = 0.5 / (c * n_f) * w_sq;
    let total_loss = ce_loss + reg;

    // Gradient via BLAS dgemv_t: g_weights = X^T @ err
    blas::gemv_rm_t(x_rm, err_buf, gw_buf, n, p);

    // Assemble gradient
    let reg_scale = 1.0 / (c * n_f);
    let mut grad = DVector::zeros(p + 1);
    grad[0] = g_bias / n_f;
    for j in 0..p {
        grad[j + 1] = gw_buf[j] / n_f + wt[j] * reg_scale;
    }

    (total_loss, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    /// Deterministic PRNG (xorshift64) — no external dependency needed.
    fn xorshift64(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state as f64) / (u64::MAX as f64)
    }

    /// Generate a synthetic multiclass dataset with `k` classes.
    fn synthetic_multiclass(n: usize, p: usize, k: usize, seed: u64) -> (Vec<f64>, Vec<i64>) {
        let mut state = seed;
        let x_rm: Vec<f64> = (0..n * p).map(|_| xorshift64(&mut state) * 2.0 - 1.0).collect();
        let y: Vec<i64> = (0..n).map(|i| (i % k) as i64).collect();
        (x_rm, y)
    }

    /// Sequential OvR baseline — identical logic but no rayon.
    fn fit_sequential_ovr(
        x_rm: &[f64],
        y: &[i64],
        n: usize,
        p: usize,
        c: f64,
        max_iter: usize,
    ) -> Vec<DVector<f64>> {
        let mut classes: Vec<i64> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let k = classes.len();
        let pos_classes: Vec<i64> = if k == 2 {
            vec![classes[1]]
        } else {
            classes.clone()
        };

        let mut coefs = Vec::with_capacity(pos_classes.len());
        for &pos_class in &pos_classes {
            let y_bin: Vec<f64> = y
                .iter()
                .map(|&label| if label == pos_class { 1.0 } else { 0.0 })
                .collect();
            let w0 = DVector::zeros(p + 1);
            let mut z_buf = vec![0.0_f64; n];
            let mut err_buf = vec![0.0_f64; n];
            let mut gw_buf = vec![0.0_f64; p];
            let coef = lbfgs(
                |w| binary_loss_grad_blas_reuse(
                    w, x_rm, &y_bin, n, p, c, None,
                    &mut z_buf, &mut err_buf, &mut gw_buf,
                ),
                w0,
                LBFGS_M,
                max_iter,
                LBFGS_TOL,
            );
            coefs.push(coef);
        }
        coefs
    }

    #[test]
    fn parallel_ovr_matches_sequential_5class() {
        let k = 5;
        let n = 200;
        let p = 10;
        let c = 1.0;
        let max_iter = 500;
        let (x_rm, y) = synthetic_multiclass(n, p, k, 42);

        // Sequential baseline
        let seq_coefs = fit_sequential_ovr(&x_rm, &y, n, p, c, max_iter);

        // Parallel (via fit_row_major)
        let mut model = LogisticModel::new(c, max_iter);
        model.fit_row_major(&x_rm, &y, n, p, None).unwrap();

        assert_eq!(model.coefs.len(), seq_coefs.len(), "number of classifiers must match");
        for (ci, (par, seq)) in model.coefs.iter().zip(seq_coefs.iter()).enumerate() {
            assert_eq!(par.len(), seq.len(), "coef length mismatch for class {ci}");
            for j in 0..par.len() {
                let diff = (par[j] - seq[j]).abs();
                assert!(
                    diff < 1e-10,
                    "class {ci} coef[{j}]: parallel={} sequential={} diff={diff}",
                    par[j], seq[j],
                );
            }
        }
    }

    #[test]
    fn parallel_ovr_matches_sequential_10class() {
        // Larger k to ensure parallelism is exercised across many tasks.
        let k = 10;
        let n = 300;
        let p = 8;
        let c = 0.5;
        let max_iter = 500;
        let (x_rm, y) = synthetic_multiclass(n, p, k, 7777);

        let seq_coefs = fit_sequential_ovr(&x_rm, &y, n, p, c, max_iter);

        let mut model = LogisticModel::new(c, max_iter);
        model.fit_row_major(&x_rm, &y, n, p, None).unwrap();

        assert_eq!(model.coefs.len(), seq_coefs.len());
        for (ci, (par, seq)) in model.coefs.iter().zip(seq_coefs.iter()).enumerate() {
            for j in 0..par.len() {
                let diff = (par[j] - seq[j]).abs();
                assert!(
                    diff < 1e-10,
                    "class {ci} coef[{j}]: parallel={} sequential={} diff={diff}",
                    par[j], seq[j],
                );
            }
        }
    }

    #[test]
    fn parallel_ovr_binary_unchanged() {
        // Binary (k=2) uses only 1 classifier — parallelism is trivial but must not break.
        let n = 100;
        let p = 5;
        let c = 1.0;
        let max_iter = 500;
        let (x_rm, y_full) = synthetic_multiclass(n, p, 2, 99);

        let seq_coefs = fit_sequential_ovr(&x_rm, &y_full, n, p, c, max_iter);

        let mut model = LogisticModel::new(c, max_iter);
        model.fit_row_major(&x_rm, &y_full, n, p, None).unwrap();

        assert_eq!(model.coefs.len(), 1);
        assert_eq!(seq_coefs.len(), 1);
        for j in 0..model.coefs[0].len() {
            let diff = (model.coefs[0][j] - seq_coefs[0][j]).abs();
            assert!(diff < 1e-10, "binary coef[{j}] diff={diff}");
        }
    }

    #[test]
    fn parallel_ovr_dmatrix_path() {
        // Verify the DMatrix-based fit() path also produces correct results.
        let k = 5;
        let n = 150;
        let p = 6;
        let c = 1.0;
        let max_iter = 500;
        let (x_rm, y) = synthetic_multiclass(n, p, k, 2025);

        // Fit via DMatrix path
        let x = DMatrix::from_row_slice(n, p, &x_rm);
        let mut model = LogisticModel::new(c, max_iter);
        model.fit(&x, &y, None).unwrap();

        // Fit via row-major path (already verified against sequential)
        let mut model_rm = LogisticModel::new(c, max_iter);
        model_rm.fit_row_major(&x_rm, &y, n, p, None).unwrap();

        assert_eq!(model.coefs.len(), model_rm.coefs.len());
        // DMatrix and row-major paths use different loss/grad functions, so
        // coefficients won't be bitwise identical. But they should converge to
        // the same optimum within L-BFGS tolerance.
        for (ci, (dm, rm)) in model.coefs.iter().zip(model_rm.coefs.iter()).enumerate() {
            for j in 0..dm.len() {
                let diff = (dm[j] - rm[j]).abs();
                assert!(
                    diff < 1e-4,
                    "DMatrix vs row-major class {ci} coef[{j}]: dm={} rm={} diff={diff}",
                    dm[j], rm[j],
                );
            }
        }
    }
}
