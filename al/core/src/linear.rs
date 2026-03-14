//! Ridge regression via augmented normal equations.
//!
//! Port of `ml._linear._LinearModel` (Python reference implementation).
//!
//! Solves: `min ||y - (X w + b)||^2 + alpha * ||w||^2`
//! via:    `(X_aug^T X_aug + Lambda) w_aug = X_aug^T y`
//! where   `X_aug = [1 | X]` and `Lambda = diag(0, alpha, ..., alpha)`.
//!
//! Solver strategy:
//! - **Dual path** (p > n, alpha > 0): solve the n×n dual system
//!   `(X X^T + alpha*I_n) a = y`, then `w = X^T a`.  O(n³) vs O(p³).
//! - **Primal path** (p <= n or alpha = 0): augmented normal equations.
//!   When alpha > 0 the system matrix is positive definite → Cholesky (fast).
//! - When alpha = 0 or features are collinear the matrix is PSD, not PD.
//!   Cholesky::new() returns None → fall back to try_inverse() (LU-based).

use crate::blas;
use crate::error::MlError;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Ridge regression model.
#[derive(Serialize, Deserialize)]
pub struct LinearModel {
    /// L2 regularization strength (0.0 = OLS, no regularization).
    pub alpha: f64,
    /// Fitted bias (intercept) term.  Set by `fit`.
    pub intercept: f64,
    /// Fitted feature weights, shape `(n_features,)`.  Set by `fit`.
    pub coef: DVector<f64>,
}

impl LinearModel {
    /// Create a new, unfitted `LinearModel`.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            intercept: 0.0,
            coef: DVector::zeros(0),
        }
    }

    /// Fit Ridge regression on `(x, y)`.
    ///
    /// # Arguments
    /// - `x`: `(n_samples, n_features)` design matrix.
    /// - `y`: `(n_samples,)` target vector.
    /// - `sample_weight`: optional per-sample weights (unnormalized).
    ///   Normalized internally to sum = n (matches Python/R convention).
    ///
    /// # Errors
    /// - `EmptyData` if `x.nrows() == 0`.
    /// - `DimensionMismatch` if `y.len() != x.nrows()`.
    /// - `SingularMatrix` if the system is singular (alpha=0 with collinear features).
    pub fn fit(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
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

        // Dual path: when p > n and alpha > 0, solve the n×n dual system
        // instead of the (p+1)×(p+1) primal system. O(n³) vs O(p³).
        if p > n && self.alpha > 0.0 {
            return self.fit_dual(x, y, sample_weight);
        }

        // Primal path: build normal equations via BLAS GEMM on column-major data.
        //
        // nalgebra DMatrix stores data column-major. We compute X^T X
        // directly on this layout via cblas_dgemm(ColMajor, Trans, NoTrans),
        // then assemble the (p+1)×(p+1) system with intercept terms.
        // Zero copies from nalgebra.

        let (a, rhs) = if let Some(sw) = sample_weight {
            if sw.len() != n {
                return Err(MlError::DimensionMismatch {
                    expected: n,
                    got: sw.len(),
                });
            }
            // Normalize weights to sum = n, apply sqrt to rows.
            let sw_sum: f64 = sw.iter().sum();
            let scale = n as f64 / sw_sum;
            let sw_sqrt: Vec<f64> = sw.iter().map(|v| (v * scale).sqrt()).collect();

            // Build sqrt-weighted X in column-major layout.
            let mut xw = vec![0.0_f64; n * p];
            for j in 0..p {
                let col_start = j * n;
                let x_col = x.column(j);
                for i in 0..n {
                    xw[col_start + i] = x_col[i] * sw_sqrt[i];
                }
            }
            let yw: Vec<f64> = (0..n).map(|i| y[i] * sw_sqrt[i]).collect();
            build_normal_eqs_col(&xw, &yw, &sw_sqrt, n, p, self.alpha)
        } else {
            let x_data = x.as_slice();
            let y_data = y.as_slice();
            build_normal_eqs_col(x_data, y_data, &[], n, p, self.alpha)
        };

        // Solve the (p+1)×(p+1) system.
        // Try Cholesky first (positive definite when alpha > 0).
        // Fall back to try_inverse() (LU-based) for alpha=0 or collinear features.
        let w: DVector<f64> = match a.clone().cholesky() {
            Some(chol) => chol.solve(&rhs),
            None => {
                let a_inv = a.try_inverse().ok_or(MlError::SingularMatrix)?;
                a_inv * &rhs
            }
        };

        self.intercept = w[0];
        self.coef = w.rows(1, p).into_owned();
        Ok(self)
    }

    /// Dual solver for Ridge regression when p > n.
    ///
    /// Instead of solving the (p+1)×(p+1) primal system, centers X and y,
    /// then solves the n×n dual system: (X_c X_c^T + alpha*I_n) a = y_c,
    /// and recovers w = X_c^T a.  Complexity: O(n³) instead of O(p³).
    ///
    /// Requires alpha > 0 (dual system needs regularization to be PD).
    fn fit_dual(
        &mut self,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
        sample_weight: Option<&DVector<f64>>,
    ) -> Result<&mut Self, MlError> {
        let n = x.nrows();
        let p = x.ncols();

        // Compute column means and y mean, applying sample weights if present.
        let (x_mean, y_mean, x_c, y_c) = if let Some(sw) = sample_weight {
            // Normalize weights to sum = n.
            let sw_sum: f64 = sw.iter().sum();
            let scale = n as f64 / sw_sum;
            let w_norm: Vec<f64> = sw.iter().map(|v| v * scale).collect();

            // Weighted means.
            let mut xm = DVector::zeros(p);
            let mut ym = 0.0_f64;
            for i in 0..n {
                ym += w_norm[i] * y[i];
                for j in 0..p {
                    xm[j] += w_norm[i] * x[(i, j)];
                }
            }
            let wsum: f64 = w_norm.iter().sum(); // = n after normalization
            xm /= wsum;
            ym /= wsum;

            // Center and apply sqrt-weights.
            let mut xc = DMatrix::zeros(n, p);
            let mut yc = DVector::zeros(n);
            for i in 0..n {
                let sw_i = w_norm[i].sqrt();
                yc[i] = (y[i] - ym) * sw_i;
                for j in 0..p {
                    xc[(i, j)] = (x[(i, j)] - xm[j]) * sw_i;
                }
            }
            (xm, ym, xc, yc)
        } else {
            // Unweighted means.
            let mut xm = DVector::zeros(p);
            for j in 0..p {
                let mut s = 0.0_f64;
                for i in 0..n {
                    s += x[(i, j)];
                }
                xm[j] = s / n as f64;
            }
            let ym: f64 = y.iter().sum::<f64>() / n as f64;

            // Center X and y.
            let mut xc = DMatrix::zeros(n, p);
            let mut yc = DVector::zeros(n);
            for i in 0..n {
                yc[i] = y[i] - ym;
                for j in 0..p {
                    xc[(i, j)] = x[(i, j)] - xm[j];
                }
            }
            (xm, ym, xc, yc)
        };

        // K = X_c * X_c^T  (n × n)
        let k = &x_c * x_c.transpose();

        // K + alpha * I_n
        let mut k_reg = k;
        for i in 0..n {
            k_reg[(i, i)] += self.alpha;
        }

        // Solve (K + alpha*I) a = y_c
        let a_dual: DVector<f64> = match k_reg.clone().cholesky() {
            Some(chol) => chol.solve(&y_c),
            None => {
                let inv = k_reg.try_inverse().ok_or(MlError::SingularMatrix)?;
                inv * &y_c
            }
        };

        // w = X_c^T a
        self.coef = x_c.transpose() * &a_dual;

        // intercept = y_mean - x_mean^T w
        self.intercept = y_mean - x_mean.dot(&self.coef);

        Ok(self)
    }

    /// Fit from row-major slices — avoids DMatrix construction for PyO3 path.
    ///
    /// `x_rm` is `(n, p)` row-major. `y` is `(n,)`.
    /// `sw` is optional `(n,)` sample weights.
    pub fn fit_row_major(
        &mut self,
        x_rm: &[f64],
        y: &[f64],
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

        // Dual path: when p > n and alpha > 0, solve n×n system instead.
        if p > n && self.alpha > 0.0 {
            // Convert row-major to DMatrix for the dual solver.
            let x_mat = DMatrix::from_row_slice(n, p, x_rm);
            let y_vec = DVector::from_column_slice(y);
            let sw_vec = sw.map(|s| DVector::from_column_slice(s));
            return self.fit_dual(&x_mat, &y_vec, sw_vec.as_ref());
        }

        let (a, rhs) = if let Some(weights) = sw {
            if weights.len() != n {
                return Err(MlError::DimensionMismatch {
                    expected: n,
                    got: weights.len(),
                });
            }
            let sw_sum: f64 = weights.iter().sum();
            let scale = n as f64 / sw_sum;
            let mut xw = vec![0.0_f64; n * p];
            let mut yw = vec![0.0_f64; n];
            for i in 0..n {
                let w = (weights[i] * scale).sqrt();
                let row = i * p;
                for j in 0..p {
                    xw[row + j] = x_rm[row + j] * w;
                }
                yw[i] = y[i] * w;
            }
            build_normal_eqs_row(&xw, &yw, n, p, self.alpha)
        } else {
            build_normal_eqs_row(x_rm, y, n, p, self.alpha)
        };

        let w: DVector<f64> = match a.clone().cholesky() {
            Some(chol) => chol.solve(&rhs),
            None => {
                let a_inv = a.try_inverse().ok_or(MlError::SingularMatrix)?;
                a_inv * &rhs
            }
        };

        self.intercept = w[0];
        self.coef = w.rows(1, p).into_owned();
        Ok(self)
    }

    /// Predict target values for `x`.
    pub fn predict(&self, x: &DMatrix<f64>) -> DVector<f64> {
        let intercept = self.intercept;
        (x * &self.coef).map(|v| v + intercept)
    }

    /// Predict from row-major slice — uses BLAS GEMV for large inputs.
    pub fn predict_row_major(&self, x_rm: &[f64], n: usize, p: usize) -> Vec<f64> {
        let intercept = self.intercept;
        let coef = self.coef.as_slice();
        if n >= 32 {
            // BLAS: out = X(n×p) * coef(p), then add intercept
            let mut out = vec![0.0_f64; n];
            blas::gemv_rm(x_rm, coef, &mut out, n, p);
            for v in out.iter_mut() {
                *v += intercept;
            }
            out
        } else {
            (0..n)
                .map(|i| {
                    let row = &x_rm[i * p..(i + 1) * p];
                    let mut v = intercept;
                    for j in 0..p {
                        v += row[j] * coef[j];
                    }
                    v
                })
                .collect()
        }
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

impl Default for LinearModel {
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// Build (p+1)×(p+1) normal equations from row-major X and y via BLAS.
///
/// Used by the PyO3 path (numpy is row-major).
fn build_normal_eqs_row(
    x_rm: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    alpha: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    // X^T X (p×p) via BLAS: gemm_at_b on row-major data.
    let mut xtx = vec![0.0_f64; p * p];
    blas::gemm_at_b(x_rm, x_rm, &mut xtx, p, p, n);

    // X^T y and column sums in a single pass over rows.
    let mut xty = vec![0.0_f64; p];
    let mut col_sums = vec![0.0_f64; p];
    let mut sum_y = 0.0_f64;
    for i in 0..n {
        let yi = y[i];
        sum_y += yi;
        let row = &x_rm[i * p..(i + 1) * p];
        for j in 0..p {
            xty[j] += row[j] * yi;
            col_sums[j] += row[j];
        }
    }

    // Assemble (p+1)×(p+1) system.
    let d = p + 1;
    let mut a = DMatrix::zeros(d, d);
    a[(0, 0)] = n as f64;
    for j in 0..p {
        a[(0, j + 1)] = col_sums[j];
        a[(j + 1, 0)] = col_sums[j];
    }
    // xtx is row-major (p×p) from gemm_at_b
    for i in 0..p {
        for j in 0..p {
            a[(i + 1, j + 1)] = xtx[i * p + j];
        }
        a[(i + 1, i + 1)] += alpha;
    }

    let mut rhs = DVector::zeros(d);
    rhs[0] = sum_y;
    for j in 0..p {
        rhs[j + 1] = xty[j];
    }

    (a, rhs)
}

/// Build (p+1)×(p+1) normal equations from column-major X and y via BLAS.
///
/// `x_cm` is (n × p) column-major. `y` is (n,).
/// `sw_sqrt` is empty for unweighted, or (n,) sqrt-weights for weighted.
/// Returns (A, rhs) for the augmented system including intercept.
fn build_normal_eqs_col(
    x_cm: &[f64],
    y: &[f64],
    sw_sqrt: &[f64],
    n: usize,
    p: usize,
    alpha: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    // X^T X (p×p) via BLAS — zero-copy on nalgebra's column-major storage.
    let mut xtx_cm = vec![0.0_f64; p * p];
    blas::gram_col_major(x_cm, &mut xtx_cm, n, p);

    // X^T y (p,)
    let mut xty = vec![0.0_f64; p];
    blas::gemv_col_major_t(x_cm, y, &mut xty, n, p);

    // Column sums of X (for intercept terms).
    // For weighted case, sum of sqrt-weighted ones = sum(sw_sqrt).
    let mut col_sums = vec![0.0_f64; p];
    for j in 0..p {
        let col = &x_cm[j * n..(j + 1) * n];
        let mut s = 0.0_f64;
        for i in 0..n {
            s += col[i];
        }
        col_sums[j] = s;
    }

    // Sum of y (or weighted y). Sum of "ones" for intercept diagonal.
    let sum_y: f64 = y.iter().sum();
    let sum_ones: f64 = if sw_sqrt.is_empty() {
        n as f64
    } else {
        // Weighted: sum of (sw_sqrt_i)^2 = sum of normalized weights = n
        sw_sqrt.iter().map(|w| w * w).sum()
    };

    // Assemble (p+1)×(p+1) system:
    //   A = [ sum_ones,   col_sums^T       ]  +  diag(0, alpha, ..., alpha)
    //       [ col_sums,   X^T X + alpha*I  ]
    let d = p + 1;
    let mut a = DMatrix::zeros(d, d);
    a[(0, 0)] = sum_ones;
    for j in 0..p {
        a[(0, j + 1)] = col_sums[j];
        a[(j + 1, 0)] = col_sums[j];
    }
    // xtx_cm is column-major (p×p)
    for j in 0..p {
        for i in 0..p {
            a[(i + 1, j + 1)] = xtx_cm[j * p + i];
        }
        a[(j + 1, j + 1)] += alpha;
    }

    let mut rhs = DVector::zeros(d);
    rhs[0] = sum_y;
    for j in 0..p {
        rhs[j + 1] = xty[j];
    }

    (a, rhs)
}
