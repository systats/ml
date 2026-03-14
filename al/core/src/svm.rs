//! Support Vector Machine (SMO, Platt 1998).
//!
//! # Design
//! - **Linear kernel**: Weight vector `w` is maintained explicitly
//!   so f(x) = w·x + b is O(p), not O(n_sv * p).
//! - **RBF/Polynomial kernel**: Kernel matrix cached for n <= 3000,
//!   computed on-the-fly for 3000 < n <= 5000, hard cap at n > 5000.
//!   Prediction uses support vectors: f(x) = sum(alpha_i * y_i * K(x_i, x)) + b.
//! - **Classifier**: OvR multiclass (k binary SVMs).  Predict by max decision
//!   score; predict_proba via Platt scaling per binary SVM.
//! - **Regressor**: epsilon-insensitive SVR via modified SMO (2n dual variables).
//!
//! # Contract
//! `predict()` returns 0-based class indices. Label remapping is done by the
//! Python/R bridge layer (`_rust.py`).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Kernel types
// ---------------------------------------------------------------------------

/// Kernel function for SVM.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum KernelType {
    Linear,
    RBF { gamma: f64 },
    Polynomial { gamma: f64, coef0: f64, degree: u32 },
}

impl Default for KernelType {
    fn default() -> Self { KernelType::Linear }
}

/// Maximum n for precomputed kernel matrix (72MB at n=3000: 3000^2 * 8 bytes).
const KERNEL_CACHE_MAX: usize = 3000;
/// Maximum n for on-the-fly kernel evaluation (no cache).
const KERNEL_OTF_MAX: usize = 5000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Squared Euclidean distance ||a - b||^2.
#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| { let d = x - y; d * d }).sum()
}

/// Evaluate kernel between two samples.
#[inline]
fn kernel_eval(kt: &KernelType, a: &[f64], b: &[f64]) -> f64 {
    match kt {
        KernelType::Linear => dot(a, b),
        KernelType::RBF { gamma } => (-gamma * sq_dist(a, b)).exp(),
        KernelType::Polynomial { gamma, coef0, degree } => {
            (gamma * dot(a, b) + coef0).powi(*degree as i32)
        }
    }
}

/// Compute default gamma ("scale" mode): 1 / (n_features * X_variance).
pub fn default_gamma_scale(x: &[f64], n: usize, p: usize) -> f64 {
    if n == 0 || p == 0 { return 1.0; }
    let total = (n * p) as f64;
    let mean_all: f64 = x.iter().sum::<f64>() / total;
    let var_all: f64 = x.iter().map(|&v| (v - mean_all).powi(2)).sum::<f64>() / total;
    if var_all < 1e-15 { 1.0 } else { 1.0 / (p as f64 * var_all) }
}

// ---------------------------------------------------------------------------
// Kernel access — cached or on-the-fly
// ---------------------------------------------------------------------------

/// Access pattern for kernel values during SMO training.
enum KernelAccess {
    /// Full precomputed matrix (n <= KERNEL_CACHE_MAX).
    Cached(Vec<f64>),  // [n * n], row-major
    /// Compute on-the-fly (KERNEL_CACHE_MAX < n <= KERNEL_OTF_MAX).
    OnTheFly,
}

impl KernelAccess {
    fn build(kt: &KernelType, x: &[f64], n: usize, p: usize) -> Self {
        if n <= KERNEL_CACHE_MAX {
            let xi = |i: usize| &x[i * p..(i + 1) * p];
            let mut mat = vec![0.0f64; n * n];
            for i in 0..n {
                mat[i * n + i] = kernel_eval(kt, xi(i), xi(i));
                for j in (i + 1)..n {
                    let v = kernel_eval(kt, xi(i), xi(j));
                    mat[i * n + j] = v;
                    mat[j * n + i] = v;
                }
            }
            KernelAccess::Cached(mat)
        } else {
            KernelAccess::OnTheFly
        }
    }

    /// Get K(i, j). For OnTheFly, caller must pass x, p, kt.
    #[inline]
    fn get(&self, i: usize, j: usize, n: usize, x: &[f64], p: usize, kt: &KernelType) -> f64 {
        match self {
            KernelAccess::Cached(mat) => mat[i * n + j],
            KernelAccess::OnTheFly => {
                kernel_eval(kt, &x[i * p..(i + 1) * p], &x[j * p..(j + 1) * p])
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BinarySvm — one SMO-trained binary SVM
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone)]
struct BinarySvm {
    w: Vec<f64>,   // linear weights [p] (empty for kernel SVM)
    b: f64,
    platt_a: f64,  // Platt sigmoid param
    platt_b: f64,
    /// Support vectors for kernel SVM: (indices, alpha*y values, data [n_sv * p]).
    #[serde(default)]
    sv_indices: Vec<usize>,
    #[serde(default)]
    sv_alpha_y: Vec<f64>,   // alpha[i] * y[i] for each support vector
    #[serde(default)]
    sv_data: Vec<f64>,      // row-major support vector features [n_sv * p]
    #[serde(default)]
    n_features: usize,
    #[serde(default)]
    kernel: KernelType,
}

impl BinarySvm {
    #[inline]
    fn decision(&self, x: &[f64]) -> f64 {
        match self.kernel {
            KernelType::Linear => dot(&self.w, x) + self.b,
            _ => {
                // Kernel SVM: f(x) = sum(alpha_i * y_i * K(sv_i, x)) + b
                let p = self.n_features;
                let n_sv = self.sv_indices.len();
                let mut sum = self.b;
                for s in 0..n_sv {
                    let sv = &self.sv_data[s * p..(s + 1) * p];
                    sum += self.sv_alpha_y[s] * kernel_eval(&self.kernel, sv, x);
                }
                sum
            }
        }
    }

    #[inline]
    fn proba_pos(&self, x: &[f64]) -> f64 {
        sigmoid(self.platt_a * self.decision(x) + self.platt_b)
    }
}

// ---------------------------------------------------------------------------
// SMO binary classifier — WSS3 (Fan, Chen, Lin 2005) + gradient cache
// ---------------------------------------------------------------------------
//
// The gradient vector G[i] = sum_j(alpha_j * y_j * K(x_i, x_j)) - 1
// is maintained incrementally.  For linear kernel with explicit w:
//   f(x_i) = w . x_i + b,  so  G[i] = y_i * f(x_i) - 1.
//
// When alpha_i changes by delta_i and alpha_j by delta_j:
//   delta_w = delta_i * y_i * x_i + delta_j * y_j * x_j
//   G[k] += y_k * (dot(delta_w, x_k) + delta_b)  for all k
//
// WSS3 pair selection:
//   i = argmax { -y_t * G[t] : t in I_up }  where I_up = upper violators
//   j = argmin { -b_tj^2 / a_tj : t in I_low, -y_t * G[t] < -y_i * G[i] }
//   where b_tj = G_i - G_j (using LIBSVM convention), a_tj = K_ii + K_jj - 2*K_ij
//
// This gives O(n*p) per iteration instead of O(n^2*p).

// ---------------------------------------------------------------------------
// Primal coordinate descent for linear SVM (liblinear-style)
// ---------------------------------------------------------------------------
//
// L2-regularized L2-loss SVM (dual coordinate descent, Fan et al. 2008).
// Much faster convergence than SMO for linear SVM:
//   - Each iteration visits ALL n samples (full sweep)
//   - Updates alpha[i] + weight vector w in O(p) per sample
//   - Total per-iteration: O(n*p) — same as SMO but converges in ~20 iters vs ~10000
//
// Equivalent to liblinear solver type 1 (L2-SVM, L2-loss, dual).

fn linear_svm_dual_cd(
    x: &[f64],  // row-major [n, p]
    y: &[f64],  // +1 / -1 labels
    n: usize,
    p: usize,
    c_vec: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, f64, bool) {
    // Upper bound on alpha: 2*C for L2-loss SVM (Keerthi & DeCoste 2005)
    let upper: Vec<f64> = c_vec.iter().map(|_| f64::INFINITY).collect();
    let diag: Vec<f64> = c_vec.iter().map(|&c| 0.5 / c).collect();

    let mut alpha = vec![0.0f64; n];
    let mut w = vec![0.0f64; p];
    let xi = |i: usize| &x[i * p..(i + 1) * p];

    // Precompute QD[i] = ||x_i||^2 + diag[i]
    let qd: Vec<f64> = (0..n).map(|i| dot(xi(i), xi(i)) + diag[i]).collect();

    let mut converged = false;

    for _iter in 0..max_iter {
        let mut max_pg = f64::NEG_INFINITY;
        let mut min_pg = f64::INFINITY;

        for i in 0..n {
            // G = y_i * (w · x_i) - 1 + diag[i] * alpha[i]
            let g = y[i] * dot(&w, xi(i)) - 1.0 + diag[i] * alpha[i];

            // Projected gradient
            let pg = if alpha[i] <= 0.0 {
                g.min(0.0)
            } else if alpha[i] >= upper[i] {
                g.max(0.0)
            } else {
                g
            };

            if pg > max_pg { max_pg = pg; }
            if pg < min_pg { min_pg = pg; }

            if pg.abs() < 1e-12 { continue; }

            let old_alpha = alpha[i];
            alpha[i] = (alpha[i] - g / qd[i]).clamp(0.0, upper[i]);
            let d = (alpha[i] - old_alpha) * y[i];

            // Update w
            if d.abs() > 1e-15 {
                let xr = xi(i);
                for k in 0..p {
                    w[k] += d * xr[k];
                }
            }
        }

        if max_pg - min_pg < tol {
            converged = true;
            break;
        }
    }

    // Compute b from free SVs (same as SMO)
    let mut b_sum = 0.0f64;
    let mut b_count = 0usize;
    let mut ub = f64::INFINITY;
    let mut lb = f64::NEG_INFINITY;
    for i in 0..n {
        let fi = dot(&w, xi(i));
        let yi_minus_fi = y[i] - fi;
        if alpha[i] > 1e-8 && alpha[i] < upper[i] - 1e-8 {
            b_sum += yi_minus_fi;
            b_count += 1;
        } else if alpha[i] >= upper[i] - 1e-8 {
            if y[i] > 0.0 { lb = lb.max(yi_minus_fi); } else { ub = ub.min(yi_minus_fi); }
        } else if alpha[i] > 1e-8 {
            if y[i] > 0.0 { ub = ub.min(yi_minus_fi); } else { lb = lb.max(yi_minus_fi); }
        }
    }
    let b = if b_count > 0 {
        b_sum / b_count as f64
    } else if lb > f64::NEG_INFINITY && ub < f64::INFINITY {
        (ub + lb) * 0.5
    } else {
        0.0
    };

    (w, b, converged)
}

#[allow(dead_code)]
fn smo_train(
    x: &[f64],  // row-major [n, p]
    y: &[f64],  // +1 / -1 labels
    n: usize,
    p: usize,
    c_vec: &[f64],  // per-sample C values (supports sample_weight)
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, f64, bool) {
    let mut alpha = vec![0.0f64; n];
    let mut w = vec![0.0f64; p]; // maintained explicitly: w = sum(alpha_i * y_i * x_i)

    // Honor user-provided max_iter. Shrinking + gradient reconstruction handle
    // accuracy; inflating to n*100 caused 60× slowdowns on typical datasets.
    let effective_max_iter = max_iter.min(10_000_000);

    // Precompute row slices once
    let xi = |i: usize| &x[i * p..(i + 1) * p];

    // Precompute diagonal of kernel matrix: K_diag[i] = ||x_i||^2
    let k_diag: Vec<f64> = (0..n).map(|i| dot(xi(i), xi(i))).collect();

    // Gradient cache: G[i] = sum_j(alpha_j * Q_ij) - 1
    // where Q_ij = y_i * y_j * K(x_i, x_j).
    // Initially all alpha=0, so G[i] = -1 for all i.
    // NOTE: b is NOT tracked during iteration (libSVM approach).
    // Alpha updates use (y_i*G_i - y_j*G_j) where b cancels.
    let mut grad = vec![-1.0f64; n];

    // --- Shrinking state (libSVM-style working set reduction) ---
    let mut active = vec![true; n];
    let mut n_active = n;
    let shrink_interval = n.min(1000);
    let mut shrink_counter = 0usize;
    let mut unshrunk = false;
    let mut converged = false;

    // Periodic gradient reconstruction interval (combats FP drift).
    // For linear SVM this is O(n*p) — cheap enough every ~1000 iters.
    let reconstruct_interval = n.min(1000);
    let mut reconstruct_counter = 0usize;

    for _iter in 0..effective_max_iter {
        // --- WSS3 step 1: select i from I_upper (active set only) ---
        let mut best_i = None::<usize>;
        let mut best_gi = f64::NEG_INFINITY;

        for t in 0..n {
            if !active[t] { continue; }
            let in_upper = (y[t] > 0.0 && alpha[t] < c_vec[t])
                        || (y[t] < 0.0 && alpha[t] > 0.0);
            if !in_upper { continue; }
            let neg_yg = -y[t] * grad[t];
            if neg_yg > best_gi {
                best_gi = neg_yg;
                best_i = Some(t);
            }
        }

        let i = match best_i {
            Some(i) => i,
            None => { converged = true; break; }
        };

        // --- WSS3 step 2: select j from I_lower (active set only) ---
        let mut best_j = None::<usize>;
        let mut best_obj = f64::INFINITY;

        for t in 0..n {
            if !active[t] { continue; }
            if t == i { continue; }
            let in_lower = (y[t] > 0.0 && alpha[t] > 0.0)
                        || (y[t] < 0.0 && alpha[t] < c_vec[t]);
            if !in_lower { continue; }
            let neg_yg_t = -y[t] * grad[t];
            if neg_yg_t >= best_gi { continue; }

            let b_val = best_gi - neg_yg_t;
            let kij = dot(xi(i), xi(t));
            let a_val = k_diag[i] + k_diag[t] - 2.0 * y[i] * y[t] * kij;
            let a_val = if a_val > 0.0 { a_val } else { 1e-12 };

            let obj = -(b_val * b_val) / a_val;
            if obj < best_obj {
                best_obj = obj;
                best_j = Some(t);
            }
        }

        // Fallback j: max gradient gap (gradient-based, no f_cache needed)
        let j = match best_j {
            Some(j) => j,
            None => {
                let gi = -y[i] * grad[i];
                let mut fallback_j = (i + 1) % n;
                let mut best_delta = 0.0f64;
                for t in 0..n {
                    if !active[t] { continue; }
                    if t == i { continue; }
                    let gt = -y[t] * grad[t];
                    let d = (gi - gt).abs();
                    if d > best_delta {
                        best_delta = d;
                        fallback_j = t;
                    }
                }
                fallback_j
            }
        };

        // --- Convergence check (on active set) ---
        let mut min_neg_yg_low = f64::INFINITY;
        for t in 0..n {
            if !active[t] { continue; }
            let in_lower = (y[t] > 0.0 && alpha[t] > 0.0)
                        || (y[t] < 0.0 && alpha[t] < c_vec[t]);
            if !in_lower { continue; }
            let neg_yg = -y[t] * grad[t];
            if neg_yg < min_neg_yg_low { min_neg_yg_low = neg_yg; }
        }
        let gap = best_gi - min_neg_yg_low;
        if gap < tol {
            if n_active < n {
                smo_linear_reconstruct_all(&mut active, &mut grad, &w, x, y, n, p);
                n_active = n;
                continue;
            }
            converged = true;
            break;
        }

        // --- Shrinking (every min(n,1000) iterations) ---
        shrink_counter += 1;
        if shrink_counter >= shrink_interval {
            shrink_counter = 0;

            if !unshrunk && gap < 10.0 * tol {
                unshrunk = true;
                smo_linear_reconstruct_all(&mut active, &mut grad, &w, x, y, n, p);
                n_active = n;
                continue;
            }

            for t in 0..n {
                if !active[t] { continue; }
                let is_free = alpha[t] > 1e-10 && alpha[t] < c_vec[t] - 1e-10;
                if is_free { continue; }

                let neg_yg = -y[t] * grad[t];
                let only_upper = (y[t] > 0.0 && alpha[t] < 1e-10)
                              || (y[t] < 0.0 && (c_vec[t] - alpha[t]) < 1e-10);

                let shrink = if only_upper {
                    neg_yg < min_neg_yg_low
                } else {
                    neg_yg > best_gi
                };
                if shrink {
                    active[t] = false;
                    n_active -= 1;
                }
            }
        }

        // --- Solve the (i, j) pair ---
        // Alpha update uses gradient directly (libSVM approach).
        // ei - ej = y_i*G_i - y_j*G_j (b cancels in error difference).
        let ei_ej = y[i] * grad[i] - y[j] * grad[j];

        let alpha_i_old = alpha[i];
        let alpha_j_old = alpha[j];

        let (l, h) = if (y[i] - y[j]).abs() < 1e-9 {
            let s = alpha[i] + alpha[j];
            ((s - c_vec[i]).max(0.0), s.min(c_vec[j]))
        } else {
            let d = alpha[j] - alpha[i];
            (d.max(0.0), (d + c_vec[i]).min(c_vec[j]))
        };
        if (h - l).abs() < 1e-9 { continue; }

        let kij = dot(xi(i), xi(j));
        let eta = 2.0 * kij - k_diag[i] - k_diag[j];
        const TAU: f64 = 1e-12;
        let eta_safe = if eta >= -TAU { -TAU } else { eta };

        let new_aj = (alpha[j] - y[j] * ei_ej / eta_safe).clamp(l, h);
        if (new_aj - alpha[j]).abs() < 1e-8 { continue; }

        let new_ai = alpha[i] + y[i] * y[j] * (alpha[j] - new_aj);
        let new_ai = new_ai.clamp(0.0, c_vec[i]);

        let d_ai = new_ai - alpha_i_old;
        let d_aj = new_aj - alpha_j_old;

        // Update w (maintained for gradient reconstruction)
        let xi_sl = xi(i);
        let xj_sl = xi(j);
        for k in 0..p {
            w[k] += d_ai * y[i] * xi_sl[k] + d_aj * y[j] * xj_sl[k];
        }

        alpha[i] = new_ai;
        alpha[j] = new_aj;

        // --- Incremental gradient update (active set only) ---
        for k in 0..n {
            if !active[k] { continue; }
            let xk = xi(k);
            let dot_ik = dot(xi_sl, xk);
            let dot_jk = dot(xj_sl, xk);
            grad[k] += d_ai * y[k] * y[i] * dot_ik
                      + d_aj * y[k] * y[j] * dot_jk;
        }

        // --- Periodic gradient reconstruction (combats FP drift) ---
        reconstruct_counter += 1;
        if reconstruct_counter >= reconstruct_interval {
            reconstruct_counter = 0;
            // Recompute w from alphas for numerical stability
            w.fill(0.0);
            for t in 0..n {
                if alpha[t].abs() > 1e-15 {
                    let xt = xi(t);
                    let ay = alpha[t] * y[t];
                    for k in 0..p {
                        w[k] += ay * xt[k];
                    }
                }
            }
            // Reconstruct gradients from fresh w for ALL active samples
            for t in 0..n {
                if active[t] {
                    grad[t] = y[t] * dot(&w, xi(t)) - 1.0;
                }
            }
        }
    }

    // Final w recomputed from alphas for numerical stability
    let mut w_final = vec![0.0f64; p];
    for i in 0..n {
        if alpha[i].abs() > 1e-10 {
            for k in 0..p {
                w_final[k] += alpha[i] * y[i] * xi(i)[k];
            }
        }
    }

    // Compute b as average over free SVs; fallback to midpoint of
    // KKT bounds from bounded variables (libSVM's calculate_rho).
    let mut b_sum = 0.0f64;
    let mut b_count = 0usize;
    let mut ub = f64::INFINITY;   // upper bound from bounded SVs
    let mut lb = f64::NEG_INFINITY; // lower bound from bounded SVs
    for i in 0..n {
        let fi = dot(&w_final, xi(i));
        let yi_minus_fi = y[i] - fi;
        if alpha[i] > 1e-8 && alpha[i] < c_vec[i] - 1e-8 {
            // Free SV: exact b = y_i - w·x_i
            b_sum += yi_minus_fi;
            b_count += 1;
        } else if alpha[i] >= c_vec[i] - 1e-8 {
            // At upper bound
            if y[i] > 0.0 {
                lb = lb.max(yi_minus_fi);
            } else {
                ub = ub.min(yi_minus_fi);
            }
        } else if alpha[i] > 1e-8 {
            // Between 0 and C but not free (shouldn't happen, but handle)
            if y[i] > 0.0 {
                ub = ub.min(yi_minus_fi);
            } else {
                lb = lb.max(yi_minus_fi);
            }
        }
    }
    let b_final = if b_count > 0 {
        b_sum / b_count as f64
    } else if lb > f64::NEG_INFINITY && ub < f64::INFINITY {
        (ub + lb) * 0.5  // libSVM midpoint fallback
    } else {
        0.0
    };

    (w_final, b_final, converged)
}

/// Reconstruct ALL gradients from weight vector for active+inactive samples.
/// Used during unshrinking and periodic reconstruction. O(n * p).
#[allow(dead_code)]
fn smo_linear_reconstruct_all(
    active: &mut [bool],
    grad: &mut [f64],
    w: &[f64],
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
) {
    let xi = |i: usize| &x[i * p..(i + 1) * p];
    for t in 0..n {
        if !active[t] {
            active[t] = true;
        }
        grad[t] = y[t] * dot(w, xi(t)) - 1.0;
    }
}

// ---------------------------------------------------------------------------
// SMO for kernel SVM — uses kernel cache or on-the-fly evaluation
// ---------------------------------------------------------------------------
//
// Unlike the linear SMO above that maintains w explicitly, this version
// works with kernel values K(x_i, x_j). Returns (alpha, b) where alpha[i]
// are the dual variables (in [0, C_i]).

fn smo_train_kernel(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    c_vec: &[f64],
    tol: f64,
    max_iter: usize,
    kt: &KernelType,
) -> (Vec<f64>, f64, bool) {
    let mut alpha = vec![0.0f64; n];

    let ka = KernelAccess::build(kt, x, n, p);

    // K_diag[i] = K(x_i, x_i)
    let k_diag: Vec<f64> = (0..n).map(|i| ka.get(i, i, n, x, p, kt)).collect();

    // Gradient cache: G[i] = sum_j(alpha_j * Q_ij) - 1.
    // NOTE: b is NOT tracked during iteration (libSVM approach).
    let mut grad = vec![-1.0f64; n];

    let effective_max_iter = max_iter.min(10_000_000);

    // --- Shrinking state ---
    let mut active = vec![true; n];
    let mut n_active = n;
    let shrink_interval = n.min(1000);
    let mut shrink_counter = 0usize;
    let mut unshrunk = false;
    let mut converged = false;

    for _iter in 0..effective_max_iter {
        // --- WSS3 step 1: select i from I_upper (active set only) ---
        let mut best_i = None::<usize>;
        let mut best_gi = f64::NEG_INFINITY;
        for t in 0..n {
            if !active[t] { continue; }
            let in_upper = (y[t] > 0.0 && alpha[t] < c_vec[t])
                        || (y[t] < 0.0 && alpha[t] > 0.0);
            if !in_upper { continue; }
            let neg_yg = -y[t] * grad[t];
            if neg_yg > best_gi {
                best_gi = neg_yg;
                best_i = Some(t);
            }
        }

        let i = match best_i {
            Some(i) => i,
            None => { converged = true; break; }
        };

        // --- WSS3 step 2: select j from I_lower (active set only) ---
        let mut best_j = None::<usize>;
        let mut best_obj = f64::INFINITY;
        for t in 0..n {
            if !active[t] { continue; }
            if t == i { continue; }
            let in_lower = (y[t] > 0.0 && alpha[t] > 0.0)
                        || (y[t] < 0.0 && alpha[t] < c_vec[t]);
            if !in_lower { continue; }
            let neg_yg_t = -y[t] * grad[t];
            if neg_yg_t >= best_gi { continue; }
            let b_val = best_gi - neg_yg_t;
            let kij = ka.get(i, t, n, x, p, kt);
            let a_val = k_diag[i] + k_diag[t] - 2.0 * y[i] * y[t] * kij;
            let a_val = if a_val > 0.0 { a_val } else { 1e-12 };
            let obj = -(b_val * b_val) / a_val;
            if obj < best_obj {
                best_obj = obj;
                best_j = Some(t);
            }
        }

        let j = match best_j {
            Some(j) => j,
            None => {
                let gi = -y[i] * grad[i];
                let mut fallback_j = (i + 1) % n;
                let mut best_delta = 0.0f64;
                for t in 0..n {
                    if !active[t] { continue; }
                    if t == i { continue; }
                    let gt = -y[t] * grad[t];
                    let d = (gi - gt).abs();
                    if d > best_delta {
                        best_delta = d;
                        fallback_j = t;
                    }
                }
                fallback_j
            }
        };

        // --- Convergence check (on active set) ---
        let mut min_neg_yg_low = f64::INFINITY;
        for t in 0..n {
            if !active[t] { continue; }
            let in_lower = (y[t] > 0.0 && alpha[t] > 0.0)
                        || (y[t] < 0.0 && alpha[t] < c_vec[t]);
            if !in_lower { continue; }
            let neg_yg = -y[t] * grad[t];
            if neg_yg < min_neg_yg_low { min_neg_yg_low = neg_yg; }
        }
        let gap = best_gi - min_neg_yg_low;
        if gap < tol {
            if n_active < n {
                smo_kernel_reconstruct_all(&mut active, &mut grad,
                    &alpha, x, y, n, p, kt, &ka);
                n_active = n;
                continue;
            }
            converged = true;
            break;
        }

        // --- Shrinking ---
        shrink_counter += 1;
        if shrink_counter >= shrink_interval {
            shrink_counter = 0;

            if !unshrunk && gap < 10.0 * tol {
                unshrunk = true;
                smo_kernel_reconstruct_all(&mut active, &mut grad,
                    &alpha, x, y, n, p, kt, &ka);
                n_active = n;
                continue;
            }

            for t in 0..n {
                if !active[t] { continue; }
                let is_free = alpha[t] > 1e-10 && alpha[t] < c_vec[t] - 1e-10;
                if is_free { continue; }

                let neg_yg = -y[t] * grad[t];
                let only_upper = (y[t] > 0.0 && alpha[t] < 1e-10)
                              || (y[t] < 0.0 && (c_vec[t] - alpha[t]) < 1e-10);

                let shrink = if only_upper {
                    neg_yg < min_neg_yg_low
                } else {
                    neg_yg > best_gi
                };
                if shrink {
                    active[t] = false;
                    n_active -= 1;
                }
            }
        }

        // --- Solve the (i, j) pair ---
        // Use gradients directly: ei - ej = y_i*G_i - y_j*G_j (b cancels)
        let ei_ej = y[i] * grad[i] - y[j] * grad[j];
        let alpha_i_old = alpha[i];
        let alpha_j_old = alpha[j];

        let (l, h) = if (y[i] - y[j]).abs() < 1e-9 {
            let s = alpha[i] + alpha[j];
            ((s - c_vec[i]).max(0.0), s.min(c_vec[j]))
        } else {
            let d = alpha[j] - alpha[i];
            (d.max(0.0), (d + c_vec[i]).min(c_vec[j]))
        };
        if (h - l).abs() < 1e-9 { continue; }

        let kij = ka.get(i, j, n, x, p, kt);
        let eta = 2.0 * kij - k_diag[i] - k_diag[j];
        const TAU_K: f64 = 1e-12;
        let eta_safe = if eta >= -TAU_K { -TAU_K } else { eta };

        let new_aj = (alpha[j] - y[j] * ei_ej / eta_safe).clamp(l, h);
        if (new_aj - alpha[j]).abs() < 1e-8 { continue; }

        let new_ai = alpha[i] + y[i] * y[j] * (alpha[j] - new_aj);
        let new_ai = new_ai.clamp(0.0, c_vec[i]);

        let d_ai = new_ai - alpha_i_old;
        let d_aj = new_aj - alpha_j_old;

        alpha[i] = new_ai;
        alpha[j] = new_aj;

        // --- Incremental gradient update (active set only) ---
        for k in 0..n {
            if !active[k] { continue; }
            let k_ik = ka.get(i, k, n, x, p, kt);
            let k_jk = ka.get(j, k, n, x, p, kt);
            grad[k] += d_ai * y[k] * y[i] * k_ik
                      + d_aj * y[k] * y[j] * k_jk;
        }
    }

    // Compute b from free SVs; fallback to midpoint of KKT bounds
    let mut b_sum = 0.0f64;
    let mut b_count = 0usize;
    let mut ub = f64::INFINITY;
    let mut lb = f64::NEG_INFINITY;
    for i in 0..n {
        if alpha[i] > 1e-8 && alpha[i] < c_vec[i] - 1e-8 {
            // Free SV: compute f(x_i) from alphas
            let mut fi = 0.0f64;
            for j in 0..n {
                if alpha[j].abs() > 1e-10 {
                    fi += alpha[j] * y[j] * ka.get(i, j, n, x, p, kt);
                }
            }
            b_sum += y[i] - fi;
            b_count += 1;
        } else if alpha[i] >= c_vec[i] - 1e-8 {
            // rho from gradient: G[i] + 1 = y_i * sum_j(alpha_j * y_j * K_ij)
            // b should satisfy y_i * (f_i + b) >= 1 (active constraint)
            // Use gradient: -y_i * G_i gives the violation measure
            let neg_yg = -y[i] * grad[i];
            if y[i] > 0.0 { lb = lb.max(neg_yg); }
            else { ub = ub.min(neg_yg); }
        }
    }
    let b_final = if b_count > 0 {
        b_sum / b_count as f64
    } else if lb > f64::NEG_INFINITY && ub < f64::INFINITY {
        // libSVM midpoint fallback using gradient-based bounds
        // rho = (ub + lb) / 2, b = -rho
        // But our neg_yg = -y*G correlates with -rho, so:
        (ub + lb) * 0.5
    } else {
        0.0
    };

    (alpha, b_final, converged)
}

/// Reconstruct ALL gradients for active+inactive samples from alpha values.
/// O(n * n_sv). Used during unshrinking.
fn smo_kernel_reconstruct_all(
    active: &mut [bool],
    grad: &mut [f64],
    alpha: &[f64],
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    kt: &KernelType,
    ka: &KernelAccess,
) {
    for t in 0..n {
        if !active[t] {
            active[t] = true;
        }
        let mut g = -1.0f64;
        for j in 0..n {
            if alpha[j].abs() < 1e-15 { continue; }
            let k_tj = ka.get(t, j, n, x, p, kt);
            g += alpha[j] * y[t] * y[j] * k_tj;
        }
        grad[t] = g;
    }
}

// ---------------------------------------------------------------------------
// Platt scaling  (Lin et al. 2007 improved version)
// ---------------------------------------------------------------------------

fn platt_calibrate(decisions: &[f64], labels: &[f64]) -> (f64, f64) {
    let n = decisions.len() as f64;
    let n_pos = labels.iter().filter(|&&y| y > 0.0).count() as f64;
    let n_neg = n - n_pos;

    // Modified targets
    let t: Vec<f64> = labels
        .iter()
        .map(|&y| {
            if y > 0.0 {
                (n_pos + 1.0) / (n_pos + 2.0)
            } else {
                1.0 / (n_neg + 2.0)
            }
        })
        .collect();

    let mut a = 0.0f64;
    let mut b = f64::ln((n_neg + 1.0) / (n_pos + 1.0));

    let max_iter = 100;
    let sigma = 1e-12;

    for _ in 0..max_iter {
        let mut h11 = sigma;
        let mut h22 = sigma;
        let mut h21 = 0.0f64;
        let mut g1 = 0.0f64;
        let mut g2 = 0.0f64;

        for (i, &f) in decisions.iter().enumerate() {
            let fapb = a * f + b;
            let p = if fapb >= 0.0 {
                let ef = (-fapb).exp();
                ef / (1.0 + ef)
            } else {
                let ef = fapb.exp();
                1.0 / (1.0 + ef)
            };
            let q = 1.0 - p;
            let d2 = p * q;
            h11 += f * f * d2;
            h22 += d2;
            h21 += f * d2;
            let d1 = t[i] - p;
            g1 += f * d1;
            g2 += d1;
        }

        if g1.abs() < 1e-5 && g2.abs() < 1e-5 { break; }

        let det = h11 * h22 - h21 * h21;
        if det.abs() < 1e-15 { break; }
        a += (h22 * g1 - h21 * g2) / det;
        b += (h11 * g2 - h21 * g1) / det;
    }

    (a, b)
}

// ---------------------------------------------------------------------------
// SvmClassifier (public)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct SvmClassifier {
    pub c: f64,
    pub tol: f64,
    pub max_iter: usize,
    pub n_classes: usize,
    pub n_features: usize,
    #[serde(default)]
    pub kernel: KernelType,
    classifiers: Vec<BinarySvm>,
    /// Per-feature mean for built-in standardization (empty = no scaling).
    #[serde(default)]
    feat_mean: Vec<f64>,
    /// Per-feature std for built-in standardization (empty = no scaling).
    #[serde(default)]
    feat_std: Vec<f64>,
}

impl SvmClassifier {
    pub fn new(c: f64, tol: f64, max_iter: usize) -> Self {
        Self { c, tol, max_iter, n_classes: 0, n_features: 0, kernel: KernelType::Linear, classifiers: vec![], feat_mean: vec![], feat_std: vec![] }
    }

    pub fn with_kernel(c: f64, tol: f64, max_iter: usize, kernel: KernelType) -> Self {
        Self { c, tol, max_iter, n_classes: 0, n_features: 0, kernel, classifiers: vec![], feat_mean: vec![], feat_std: vec![] }
    }

    /// Standardize features in-place: (x - mean) / std per column.
    /// Stores mean/std for use at predict time.
    fn standardize_fit(&mut self, x: &[f64], n: usize, p: usize) -> Vec<f64> {
        let mut mean = vec![0.0f64; p];
        let mut std = vec![0.0f64; p];
        for j in 0..p {
            let s: f64 = (0..n).map(|i| x[i * p + j]).sum();
            mean[j] = s / n as f64;
        }
        for j in 0..p {
            let s: f64 = (0..n).map(|i| { let d = x[i * p + j] - mean[j]; d * d }).sum();
            let v = (s / n as f64).sqrt();
            std[j] = if v > 1e-12 { v } else { 1.0 }; // avoid div by zero
        }
        let mut xs = vec![0.0f64; n * p];
        for i in 0..n {
            for j in 0..p {
                xs[i * p + j] = (x[i * p + j] - mean[j]) / std[j];
            }
        }
        self.feat_mean = mean;
        self.feat_std = std;
        xs
    }

    /// Standardize features using stored mean/std from fit.
    fn standardize_transform(&self, x: &[f64], n: usize, p: usize) -> Vec<f64> {
        if self.feat_mean.is_empty() { return x.to_vec(); }
        let mut xs = vec![0.0f64; n * p];
        for i in 0..n {
            for j in 0..p {
                xs[i * p + j] = (x[i * p + j] - self.feat_mean[j]) / self.feat_std[j];
            }
        }
        xs
    }

    /// Fit OvR SVM. `y` is 0-based integer class index.
    ///
    /// `sample_weight`: optional per-sample weights. When provided, each
    /// sample's box constraint C_i is scaled proportionally:
    ///   C_i = C * w_i * (n / sum(w))
    /// so that the total regularization budget is preserved.
    ///
    /// Returns `Err` if n > 5000 for non-linear kernel.
    pub fn fit(&mut self, x: &[f64], y: &[i64], n: usize, p: usize, sample_weight: Option<&[f64]>) -> Result<(), String> {
        // Hard cap: n > 5000 for non-linear kernel
        if !matches!(self.kernel, KernelType::Linear) && n > KERNEL_OTF_MAX {
            return Err(format!(
                "Kernel SVM (non-linear) limited to n <= {}. Got n = {}. \
                 Use kernel='linear' for larger datasets, or subsample.",
                KERNEL_OTF_MAX, n
            ));
        }

        // Built-in feature standardization — critical for SVM convergence
        let xs = self.standardize_fit(x, n, p);
        let x = &xs;

        let classes: Vec<i64> = {
            let mut v: Vec<i64> = y.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        };
        let k = classes.len();
        self.n_classes = k;
        self.n_features = p;

        // Compute per-sample C values from sample_weight
        let c_vec: Vec<f64> = match sample_weight {
            Some(sw) => {
                let s: f64 = sw.iter().sum();
                let scale = n as f64 / s;
                sw.iter().map(|&w| self.c * w * scale).collect()
            }
            None => vec![self.c; n],
        };

        let is_linear = matches!(self.kernel, KernelType::Linear);

        // Recompute kernel gamma on standardized data (variance ≈ 1.0 per feature)
        if !is_linear {
            let new_gamma = default_gamma_scale(x, n, p);
            match &mut self.kernel {
                KernelType::RBF { gamma } => { *gamma = new_gamma; }
                KernelType::Polynomial { gamma, .. } => { *gamma = new_gamma; }
                _ => {}
            }
        }
        let kernel = self.kernel.clone();

        let n_clf = if k == 2 { 1 } else { k };
        let pos_classes: Vec<i64> = if k == 2 { vec![classes[1]] } else { classes.clone() };

        self.classifiers = if is_linear {
            // Fast linear path — primal coordinate descent (liblinear-style)
            pos_classes
                .par_iter()
                .map(|&pos_class| {
                    let y_bin: Vec<f64> = y.iter().map(|&yi| if yi == pos_class { 1.0 } else { -1.0 }).collect();
                    let (w, b, _converged) = linear_svm_dual_cd(x, &y_bin, n, p, &c_vec, self.tol, self.max_iter);
                    let decisions: Vec<f64> = (0..n).map(|i| dot(&w, &x[i*p..(i+1)*p]) + b).collect();
                    let (pa, pb) = platt_calibrate(&decisions, &y_bin);
                    BinarySvm {
                        w, b, platt_a: pa, platt_b: pb,
                        sv_indices: vec![], sv_alpha_y: vec![], sv_data: vec![],
                        n_features: p, kernel: KernelType::Linear,
                    }
                })
                .collect()
        } else {
            // Kernel SVM path — use smo_train_kernel, store support vectors
            // Note: kernel SVM uses sequential OvR (no par_iter) because
            // the kernel cache is large and we don't want multiple copies
            pos_classes
                .iter()
                .map(|&pos_class| {
                    let y_bin: Vec<f64> = y.iter().map(|&yi| if yi == pos_class { 1.0 } else { -1.0 }).collect();
                    let (alpha, b, _converged) = smo_train_kernel(x, &y_bin, n, p, &c_vec, self.tol, self.max_iter, &kernel);

                    // Extract support vectors (alpha > 0)
                    let mut sv_indices = Vec::new();
                    let mut sv_alpha_y = Vec::new();
                    let mut sv_data = Vec::new();
                    for i in 0..n {
                        if alpha[i] > 1e-10 {
                            sv_indices.push(i);
                            sv_alpha_y.push(alpha[i] * y_bin[i]);
                            sv_data.extend_from_slice(&x[i * p..(i + 1) * p]);
                        }
                    }

                    // Compute decisions for Platt calibration
                    let decisions: Vec<f64> = (0..n).map(|idx| {
                        let xi = &x[idx * p..(idx + 1) * p];
                        let mut f = b;
                        for (s, &ay) in sv_alpha_y.iter().enumerate() {
                            let sv = &sv_data[s * p..(s + 1) * p];
                            f += ay * kernel_eval(&kernel, sv, xi);
                        }
                        f
                    }).collect();
                    let (pa, pb) = platt_calibrate(&decisions, &y_bin);

                    BinarySvm {
                        w: vec![], b, platt_a: pa, platt_b: pb,
                        sv_indices, sv_alpha_y, sv_data,
                        n_features: p, kernel: kernel.clone(),
                    }
                })
                .collect()
        };

        // For binary, duplicate so classes[0] and classes[1] both have a classifier.
        if k == 2 && n_clf == 1 {
            let orig = &self.classifiers[0];
            if is_linear {
                let neg = BinarySvm {
                    w: orig.w.iter().map(|&v| -v).collect(),
                    b: -orig.b,
                    platt_a: orig.platt_a,
                    platt_b: -orig.platt_b,
                    sv_indices: vec![], sv_alpha_y: vec![], sv_data: vec![],
                    n_features: p, kernel: KernelType::Linear,
                };
                self.classifiers.insert(0, neg);
            } else {
                // For kernel SVM, negate sv_alpha_y and b to flip decision
                let neg = BinarySvm {
                    w: vec![],
                    b: -orig.b,
                    platt_a: orig.platt_a,
                    platt_b: -orig.platt_b,
                    sv_indices: orig.sv_indices.clone(),
                    sv_alpha_y: orig.sv_alpha_y.iter().map(|&v| -v).collect(),
                    sv_data: orig.sv_data.clone(),
                    n_features: p,
                    kernel: kernel.clone(),
                };
                self.classifiers.insert(0, neg);
            }
        }
        Ok(())
    }

    pub fn predict(&self, x: &[f64], n: usize, p: usize) -> Vec<i64> {
        // Standardize input using stored mean/std
        let xs = self.standardize_transform(x, n, p);
        let x = &xs;

        (0..n)
            .map(|i| {
                let xi = &x[i * p..(i + 1) * p];
                (0..self.n_classes)
                    .map(|k| (k, self.classifiers[k].decision(xi)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(k, _)| k as i64)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Returns flat [n, n_classes] row-major proba array.
    pub fn predict_proba(&self, x: &[f64], n: usize, p: usize) -> Vec<f64> {
        // Standardize input using stored mean/std
        let xs = self.standardize_transform(x, n, p);
        let x = &xs;

        let k = self.n_classes;
        let mut out = vec![0.0f64; n * k];
        for i in 0..n {
            let xi = &x[i * p..(i + 1) * p];
            let mut sum = 0.0f64;
            for c in 0..k {
                let pr = self.classifiers[c].proba_pos(xi).clamp(1e-9, 1.0 - 1e-9);
                out[i * k + c] = pr;
                sum += pr;
            }
            // Normalize row
            for c in 0..k { out[i * k + c] /= sum; }
        }
        out
    }

    /// Total number of support vectors across all binary classifiers.
    pub fn n_support_vectors(&self) -> usize {
        self.classifiers.iter().map(|c| c.sv_indices.len()).sum()
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> { serde_json::to_string(self) }
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> { serde_json::from_str(s) }
}

// ---------------------------------------------------------------------------
// SvmRegressor (epsilon-SVR)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
pub struct SvmRegressor {
    pub c: f64,
    pub epsilon: f64,
    pub tol: f64,
    pub max_iter: usize,
    pub n_features: usize,
    #[serde(default)]
    pub kernel: KernelType,
    w: Vec<f64>,
    b: f64,
    #[serde(default)]
    sv_beta: Vec<f64>,     // beta[i] = alpha_p[i] - alpha_n[i] for each support vector
    #[serde(default)]
    sv_data: Vec<f64>,     // row-major support vector features
    #[serde(default)]
    feat_mean: Vec<f64>,
    #[serde(default)]
    feat_std: Vec<f64>,
    #[serde(default)]
    target_mean: f64,
    #[serde(default)]
    target_std: f64,
}

impl SvmRegressor {
    pub fn new(c: f64, epsilon: f64, tol: f64, max_iter: usize) -> Self {
        Self {
            c, epsilon, tol, max_iter, n_features: 0,
            kernel: KernelType::Linear,
            w: vec![], b: 0.0, sv_beta: vec![], sv_data: vec![],
            feat_mean: vec![], feat_std: vec![], target_mean: 0.0, target_std: 1.0,
        }
    }

    pub fn with_kernel(c: f64, epsilon: f64, tol: f64, max_iter: usize, kernel: KernelType) -> Self {
        Self {
            c, epsilon, tol, max_iter, n_features: 0,
            kernel,
            w: vec![], b: 0.0, sv_beta: vec![], sv_data: vec![],
            feat_mean: vec![], feat_std: vec![], target_mean: 0.0, target_std: 1.0,
        }
    }

    fn standardize_fit_x(&mut self, x: &[f64], n: usize, p: usize) -> Vec<f64> {
        let mut mean = vec![0.0f64; p];
        let mut std = vec![0.0f64; p];
        for j in 0..p {
            let s: f64 = (0..n).map(|i| x[i * p + j]).sum();
            mean[j] = s / n as f64;
        }
        for j in 0..p {
            let s: f64 = (0..n).map(|i| { let d = x[i * p + j] - mean[j]; d * d }).sum();
            let v = (s / n as f64).sqrt();
            std[j] = if v > 1e-12 { v } else { 1.0 };
        }
        let mut xs = vec![0.0f64; n * p];
        for i in 0..n {
            for j in 0..p {
                xs[i * p + j] = (x[i * p + j] - mean[j]) / std[j];
            }
        }
        self.feat_mean = mean;
        self.feat_std = std;
        xs
    }

    fn standardize_fit_y(&mut self, y: &[f64], n: usize) -> Vec<f64> {
        let s: f64 = y.iter().sum();
        let mean = s / n as f64;
        let var: f64 = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std = var.sqrt();
        let std = if std > 1e-12 { std } else { 1.0 };
        self.target_mean = mean;
        self.target_std = std;
        y.iter().map(|&v| (v - mean) / std).collect()
    }

    fn standardize_transform_x(&self, x: &[f64], n: usize, p: usize) -> Vec<f64> {
        if self.feat_mean.is_empty() { return x.to_vec(); }
        let mut xs = vec![0.0f64; n * p];
        for i in 0..n {
            for j in 0..p {
                xs[i * p + j] = (x[i * p + j] - self.feat_mean[j]) / self.feat_std[j];
            }
        }
        xs
    }

    /// Fit epsilon-SVR. Coordinate descent on dual (Joachims 1998 simplified).
    ///
    /// For linear kernel, maintains w = sum_i (alpha_p[i] - alpha_n[i]) * x_i
    /// explicitly. For kernel SVR, stores support vectors.
    ///
    /// `sample_weight`: optional per-sample weights. When provided, each
    /// sample's box constraint C_i is scaled proportionally:
    ///   C_i = C * w_i * (n / sum(w))
    ///
    /// Returns `Err` if n > 5000 for non-linear kernel.
    pub fn fit(&mut self, x: &[f64], y: &[f64], n: usize, p: usize, sample_weight: Option<&[f64]>) -> Result<(), String> {
        // Hard cap for non-linear kernel
        if !matches!(self.kernel, KernelType::Linear) && n > KERNEL_OTF_MAX {
            return Err(format!(
                "Kernel SVR (non-linear) limited to n <= {}. Got n = {}. \
                 Use kernel='linear' for larger datasets, or subsample.",
                KERNEL_OTF_MAX, n
            ));
        }

        self.n_features = p;
        let tol = self.tol;

        // Built-in feature + target standardization
        let xs = self.standardize_fit_x(x, n, p);
        let x = &xs;
        let ys = self.standardize_fit_y(y, n);
        let y = &ys;
        // Scale epsilon to standardized target units
        let eps = self.epsilon / self.target_std;

        // Compute per-sample C values from sample_weight
        let c_vec: Vec<f64> = match sample_weight {
            Some(sw) => {
                let s: f64 = sw.iter().sum();
                let scale = n as f64 / s;
                sw.iter().map(|&w| self.c * w * scale).collect()
            }
            None => vec![self.c; n],
        };

        let is_linear = matches!(self.kernel, KernelType::Linear);

        // Recompute kernel gamma on standardized data (variance ≈ 1.0 per feature)
        if !is_linear {
            let new_gamma = default_gamma_scale(x, n, p);
            match &mut self.kernel {
                KernelType::RBF { gamma } => { *gamma = new_gamma; }
                KernelType::Polynomial { gamma, .. } => { *gamma = new_gamma; }
                _ => {}
            }
        }

        if is_linear {
            // Fast linear path — identical to original implementation
            let mut beta = vec![0.0f64; n];
            let mut w = vec![0.0f64; p];

            for _iter in 0..self.max_iter {
                let mut max_change = 0.0f64;
                for i in 0..n {
                    let fi: f64 = dot(&w, &x[i * p..(i + 1) * p]);
                    let ri = fi - y[i];
                    let kii = dot(&x[i * p..(i + 1) * p], &x[i * p..(i + 1) * p]);
                    if kii < 1e-15 { continue; }

                    let g = if ri > eps { -ri + eps } else if ri < -eps { -ri - eps } else { 0.0 };
                    if g.abs() < tol { continue; }

                    let old_beta = beta[i];
                    let new_beta = (beta[i] + g / kii).clamp(-c_vec[i], c_vec[i]);
                    let delta = new_beta - old_beta;
                    if delta.abs() < 1e-10 { continue; }

                    beta[i] = new_beta;
                    for k in 0..p {
                        w[k] += delta * x[i * p + k];
                    }
                    if delta.abs() > max_change { max_change = delta.abs(); }
                }
                if max_change < tol { break; }
            }

            // Recompute w from beta for numerical stability
            let mut w_final = vec![0.0f64; p];
            for i in 0..n {
                if beta[i].abs() > 1e-10 {
                    for k in 0..p {
                        w_final[k] += beta[i] * x[i * p + k];
                    }
                }
            }

            let mut b_sum = 0.0f64;
            let mut b_count = 0usize;
            for i in 0..n {
                let ab = beta[i].abs();
                if ab > 1e-8 && ab < c_vec[i] - 1e-8 {
                    let fi: f64 = dot(&w_final, &x[i * p..(i + 1) * p]);
                    let b_i = y[i] - fi - eps * beta[i].signum();
                    b_sum += b_i;
                    b_count += 1;
                }
            }
            self.b = if b_count > 0 { b_sum / b_count as f64 } else { 0.0 };
            self.w = w_final;
            self.sv_beta = vec![];
            self.sv_data = vec![];
        } else {
            // Kernel SVR path — coordinate descent with kernel evaluations
            let ka = KernelAccess::build(&self.kernel, x, n, p);
            let mut beta = vec![0.0f64; n];
            // f_cache[i] = sum_j beta[j] * K(x_i, x_j)
            let mut f_cache = vec![0.0f64; n];

            for _iter in 0..self.max_iter {
                let mut max_change = 0.0f64;
                for i in 0..n {
                    let fi = f_cache[i];
                    let ri = fi - y[i];
                    let kii = ka.get(i, i, n, x, p, &self.kernel);
                    if kii < 1e-15 { continue; }

                    let g = if ri > eps { -ri + eps } else if ri < -eps { -ri - eps } else { 0.0 };
                    if g.abs() < tol { continue; }

                    let old_beta = beta[i];
                    let new_beta = (beta[i] + g / kii).clamp(-c_vec[i], c_vec[i]);
                    let delta = new_beta - old_beta;
                    if delta.abs() < 1e-10 { continue; }

                    beta[i] = new_beta;
                    // Update f_cache incrementally
                    for k_idx in 0..n {
                        f_cache[k_idx] += delta * ka.get(i, k_idx, n, x, p, &self.kernel);
                    }
                    if delta.abs() > max_change { max_change = delta.abs(); }
                }
                if max_change < tol { break; }
            }

            // Extract support vectors
            let mut sv_beta_out = Vec::new();
            let mut sv_data_out = Vec::new();
            for i in 0..n {
                if beta[i].abs() > 1e-10 {
                    sv_beta_out.push(beta[i]);
                    sv_data_out.extend_from_slice(&x[i * p..(i + 1) * p]);
                }
            }

            // Compute b from non-bound support vectors
            let mut b_sum = 0.0f64;
            let mut b_count = 0usize;
            for i in 0..n {
                let ab = beta[i].abs();
                if ab > 1e-8 && ab < c_vec[i] - 1e-8 {
                    let mut fi = 0.0f64;
                    for (s, &bv) in sv_beta_out.iter().enumerate() {
                        let sv = &sv_data_out[s * p..(s + 1) * p];
                        fi += bv * kernel_eval(&self.kernel, &x[i * p..(i + 1) * p], sv);
                    }
                    let b_i = y[i] - fi - eps * beta[i].signum();
                    b_sum += b_i;
                    b_count += 1;
                }
            }
            self.b = if b_count > 0 { b_sum / b_count as f64 } else { 0.0 };
            self.w = vec![];
            self.sv_beta = sv_beta_out;
            self.sv_data = sv_data_out;
        }
        Ok(())
    }

    pub fn predict(&self, x: &[f64], n: usize, p: usize) -> Vec<f64> {
        // Standardize input using stored mean/std
        let xs = self.standardize_transform_x(x, n, p);
        let x = &xs;

        let raw: Vec<f64> = if matches!(self.kernel, KernelType::Linear) {
            (0..n).map(|i| dot(&self.w, &x[i * p..(i + 1) * p]) + self.b).collect()
        } else {
            let n_sv = self.sv_beta.len();
            (0..n).map(|i| {
                let xi = &x[i * p..(i + 1) * p];
                let mut f = self.b;
                for s in 0..n_sv {
                    let sv = &self.sv_data[s * p..(s + 1) * p];
                    f += self.sv_beta[s] * kernel_eval(&self.kernel, sv, xi);
                }
                f
            }).collect()
        };
        // Unstandardize predictions back to original target scale
        raw.iter().map(|&v| v * self.target_std + self.target_mean).collect()
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> { serde_json::to_string(self) }
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> { serde_json::from_str(s) }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
    }

    #[test]
    fn test_svm_binary_separable() {
        // Trivially separable: x < 0 -> class 0, x > 0 -> class 1
        let n = 40;
        let x: Vec<f64> = (0..n).map(|i| if i < 20 { -1.0 - i as f64 * 0.1 } else { 1.0 + (i - 20) as f64 * 0.1 }).collect();
        let y: Vec<i64> = (0..n).map(|i| if i < 20 { 0 } else { 1 }).collect();
        let x2d: Vec<f64> = x.iter().map(|&v| v).collect();

        let mut clf = SvmClassifier::new(1.0, 1e-3, 200);
        clf.fit(&x2d, &y, n, 1, None).unwrap();

        let preds = clf.predict(&x2d, n, 1);
        let acc = preds.iter().zip(y.iter()).filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        assert!(acc > 0.95, "accuracy {acc} < 0.95");
    }

    #[test]
    fn test_svm_proba_sums_to_one() {
        let n = 40;
        let x: Vec<f64> = (0..n).map(|i| if i < 20 { -1.5 - i as f64 * 0.05 } else { 1.5 + (i - 20) as f64 * 0.05 }).collect();
        let y: Vec<i64> = (0..n).map(|i| if i < 20 { 0 } else { 1 }).collect();

        let mut clf = SvmClassifier::new(1.0, 1e-3, 200);
        clf.fit(&x, &y, n, 1, None).unwrap();
        let proba = clf.predict_proba(&x, n, 1);

        for i in 0..n {
            let s = proba[i * 2] + proba[i * 2 + 1];
            assert!((s - 1.0).abs() < 1e-9, "row {i} sums to {s}");
        }
    }

    #[test]
    fn test_svm_regressor_linear_signal() {
        let n = 50;
        let xs = linspace(-2.0, 2.0, n);
        let y: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 0.1).collect();

        let mut reg = SvmRegressor::new(10.0, 0.1, 1e-3, 500);
        reg.fit(&xs, &y, n, 1, None).unwrap();
        let preds = reg.predict(&xs, n, 1);

        let rmse: f64 = (preds.iter().zip(y.iter()).map(|(p, y)| (p - y).powi(2)).sum::<f64>() / n as f64).sqrt();
        assert!(rmse < 0.5, "SVR RMSE {rmse} >= 0.5");
    }

    #[test]
    fn test_svm_json_roundtrip() {
        let n = 20;
        let x: Vec<f64> = (0..n).map(|i| if i < 10 { -1.0 } else { 1.0 }).collect();
        let y: Vec<i64> = (0..n).map(|i| if i < 10 { 0 } else { 1 }).collect();

        let mut clf = SvmClassifier::new(1.0, 1e-3, 100);
        clf.fit(&x, &y, n, 1, None).unwrap();
        let json = clf.to_json().unwrap();
        let clf2 = SvmClassifier::from_json(&json).unwrap();
        assert_eq!(clf.predict(&x, n, 1), clf2.predict(&x, n, 1));
    }

    #[test]
    fn test_svm_sample_weight_uniform() {
        // Uniform weights should produce same results as no weights
        let n = 40;
        let x: Vec<f64> = (0..n).map(|i| if i < 20 { -1.0 - i as f64 * 0.1 } else { 1.0 + (i - 20) as f64 * 0.1 }).collect();
        let y: Vec<i64> = (0..n).map(|i| if i < 20 { 0 } else { 1 }).collect();

        let mut clf1 = SvmClassifier::new(1.0, 1e-3, 200);
        clf1.fit(&x, &y, n, 1, None).unwrap();

        let sw = vec![1.0; n];
        let mut clf2 = SvmClassifier::new(1.0, 1e-3, 200);
        clf2.fit(&x, &y, n, 1, Some(&sw)).unwrap();

        let p1 = clf1.predict(&x, n, 1);
        let p2 = clf2.predict(&x, n, 1);
        assert_eq!(p1, p2, "uniform weights should match no-weight predictions");
    }

    #[test]
    fn test_svm_sample_weight_runs() {
        // Smoke test: non-uniform weights should not panic
        let n = 40;
        let x: Vec<f64> = (0..n).map(|i| if i < 20 { -1.0 - i as f64 * 0.1 } else { 1.0 + (i - 20) as f64 * 0.1 }).collect();
        let y: Vec<i64> = (0..n).map(|i| if i < 20 { 0 } else { 1 }).collect();
        let sw: Vec<f64> = (0..n).map(|i| if i < 20 { 5.0 } else { 1.0 }).collect();

        let mut clf = SvmClassifier::new(1.0, 1e-3, 200);
        clf.fit(&x, &y, n, 1, Some(&sw)).unwrap();

        let preds = clf.predict(&x, n, 1);
        assert_eq!(preds.len(), n);
    }

    #[test]
    fn test_svr_sample_weight_runs() {
        // Smoke test: SVR with non-uniform weights should not panic
        let n = 50;
        let xs = linspace(-2.0, 2.0, n);
        let y: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 0.1).collect();
        let sw: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.1).collect();

        let mut reg = SvmRegressor::new(10.0, 0.1, 1e-3, 500);
        reg.fit(&xs, &y, n, 1, Some(&sw)).unwrap();
        let preds = reg.predict(&xs, n, 1);
        assert_eq!(preds.len(), n);
    }

    // ------------------------------------------------------------------
    // WSS3 gradient cache tests
    // ------------------------------------------------------------------

    /// Deterministic LCG for test data generation (no rand dependency)
    fn lcg_f64(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 33) as f64 / (1u64 << 31) as f64
    }

    fn make_clf_data(n: usize, p: usize, seed: u64) -> (Vec<f64>, Vec<i64>) {
        let mut state = seed;
        // Features: random in [0, 1]
        let x: Vec<f64> = (0..n * p).map(|_| lcg_f64(&mut state)).collect();
        // Label: based on first feature > 0.5 (linearly separable on dim 0)
        let y: Vec<i64> = (0..n).map(|i| if x[i * p] > 0.5 { 1 } else { 0 }).collect();
        (x, y)
    }

    #[test]
    fn test_wss3_binary_accuracy_medium() {
        // n=200, p=10: WSS3 should achieve >= 90% accuracy on linearly separable data
        let n = 200;
        let p = 10;
        let (x, y) = make_clf_data(n, p, 42);

        let mut clf = SvmClassifier::new(1.0, 1e-3, 500);
        clf.fit(&x, &y, n, p, None).unwrap();

        let preds = clf.predict(&x, n, p);
        let acc = preds.iter().zip(y.iter()).filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        assert!(acc > 0.90, "WSS3 accuracy on n=200,p=10: {acc} < 0.90");
    }

    #[test]
    fn test_wss3_multiclass_accuracy() {
        // 3-class OvR with WSS3 backend
        let n = 150;
        let p = 5;
        let mut state = 99u64;
        let x: Vec<f64> = (0..n * p).map(|_| lcg_f64(&mut state)).collect();
        // 3 classes based on feature 0 terciles
        let y: Vec<i64> = (0..n).map(|i| {
            let v = x[i * p];
            if v < 0.33 { 0 } else if v < 0.66 { 1 } else { 2 }
        }).collect();

        let mut clf = SvmClassifier::new(1.0, 1e-3, 500);
        clf.fit(&x, &y, n, p, None).unwrap();

        let preds = clf.predict(&x, n, p);
        let acc = preds.iter().zip(y.iter()).filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        // 3-class linear: 50% is reasonable floor for non-trivially separable data
        // (tercile labels on random features with standardization)
        assert!(acc > 0.50, "WSS3 3-class accuracy: {acc} < 0.50");
    }

    #[test]
    fn test_wss3_with_sample_weights() {
        // WSS3 must handle per-sample C correctly
        let n = 100;
        let p = 5;
        let (x, y) = make_clf_data(n, p, 77);

        // Heavy weight on class 0 samples
        let sw: Vec<f64> = y.iter().map(|&yi| if yi == 0 { 5.0 } else { 1.0 }).collect();

        let mut clf = SvmClassifier::new(1.0, 1e-3, 500);
        clf.fit(&x, &y, n, p, Some(&sw)).unwrap();

        let preds = clf.predict(&x, n, p);
        // Should not panic and should have reasonable predictions
        assert_eq!(preds.len(), n);

        // Class 0 recall should be higher with heavy weights
        let class0_correct = preds.iter().zip(y.iter())
            .filter(|&(p, y)| *y == 0 && *p == 0).count();
        let class0_total = y.iter().filter(|&&yi| yi == 0).count();
        let class0_recall = class0_correct as f64 / class0_total as f64;
        assert!(class0_recall > 0.5, "weighted class 0 recall {class0_recall} too low");
    }

    #[test]
    fn test_wss3_n500_convergence() {
        // WSS3 should converge within max_iter on n=500
        let n = 500;
        let p = 10;
        let (x, y) = make_clf_data(n, p, 123);

        let mut clf = SvmClassifier::new(1.0, 1e-3, 1000);
        clf.fit(&x, &y, n, p, None).unwrap();

        let preds = clf.predict(&x, n, p);
        let acc = preds.iter().zip(y.iter()).filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        assert!(acc > 0.85, "WSS3 n=500 accuracy: {acc} < 0.85");
    }

    #[test]
    fn test_wss3_proba_sums_medium() {
        // Probabilities must still sum to 1.0 after WSS3 changes
        let n = 100;
        let p = 5;
        let (x, y) = make_clf_data(n, p, 55);

        let mut clf = SvmClassifier::new(1.0, 1e-3, 300);
        clf.fit(&x, &y, n, p, None).unwrap();
        let proba = clf.predict_proba(&x, n, p);

        let k = clf.n_classes;
        for i in 0..n {
            let s: f64 = (0..k).map(|c| proba[i * k + c]).sum();
            assert!((s - 1.0).abs() < 1e-6, "row {i} proba sums to {s}");
        }
    }

    #[test]
    fn test_wss3_speed_advantage() {
        // WSS3 with gradient cache should handle n=1000 without timing out.
        // The old O(n^2) inner loop would be ~250x slower on this size.
        // This test just verifies it completes and is accurate.
        let n = 1000;
        let p = 20;
        let (x, y) = make_clf_data(n, p, 42);

        let mut clf = SvmClassifier::new(1.0, 1e-3, 500);
        clf.fit(&x, &y, n, p, None).unwrap();

        let preds = clf.predict(&x, n, p);
        let acc = preds.iter().zip(y.iter()).filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        assert!(acc > 0.85, "WSS3 n=1000 accuracy: {acc} < 0.85");
    }

    #[test]
    fn test_wss3_json_roundtrip_medium() {
        // Serialization still works after WSS3 changes
        let n = 100;
        let p = 5;
        let (x, y) = make_clf_data(n, p, 88);

        let mut clf = SvmClassifier::new(1.0, 1e-3, 300);
        clf.fit(&x, &y, n, p, None).unwrap();
        let json = clf.to_json().unwrap();
        let clf2 = SvmClassifier::from_json(&json).unwrap();

        let p1 = clf.predict(&x, n, p);
        let p2 = clf2.predict(&x, n, p);
        assert_eq!(p1, p2, "JSON roundtrip predictions differ after WSS3");
    }

    #[test]
    fn test_wss3_identical_features_no_panic() {
        // Edge case: all samples identical features (K_diag = same, eta = 0 for all pairs)
        let n = 20;
        let p = 3;
        let x: Vec<f64> = (0..n).flat_map(|_| vec![1.0, 2.0, 3.0]).collect();
        let y: Vec<i64> = (0..n).map(|i| if i < 10 { 0 } else { 1 }).collect();

        let mut clf = SvmClassifier::new(1.0, 1e-3, 100);
        clf.fit(&x, &y, n, p, None).unwrap();
        let preds = clf.predict(&x, n, p);
        assert_eq!(preds.len(), n);
    }

    #[test]
    fn test_wss3_single_feature() {
        // Minimal dimensionality: p=1, well-separated
        let n = 60;
        let x: Vec<f64> = (0..n).map(|i| if i < 30 { -(i as f64) - 1.0 } else { (i - 30) as f64 + 1.0 }).collect();
        let y: Vec<i64> = (0..n).map(|i| if i < 30 { 0 } else { 1 }).collect();

        let mut clf = SvmClassifier::new(1.0, 1e-3, 200);
        clf.fit(&x, &y, n, 1, None).unwrap();

        let preds = clf.predict(&x, n, 1);
        let acc = preds.iter().zip(y.iter()).filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        assert!(acc > 0.95, "WSS3 p=1 accuracy: {acc}");
    }

    // ------------------------------------------------------------------
    // Kernel SVM tests (RBF + Polynomial)
    // ------------------------------------------------------------------

    /// Generate XOR-like non-linearly-separable data.
    /// Quadrants: (+,+)->1, (-,-)->1, (+,-)->0, (-,+)->0
    fn make_xor_data(n_per_quadrant: usize, seed: u64) -> (Vec<f64>, Vec<i64>) {
        let mut state = seed;
        let mut x = Vec::new();
        let mut y = Vec::new();
        for q in 0..4 {
            let (sx, sy, label) = match q {
                0 => (1.0, 1.0, 1i64),    // (+,+) -> 1
                1 => (-1.0, -1.0, 1),     // (-,-) -> 1
                2 => (1.0, -1.0, 0),      // (+,-) -> 0
                3 => (-1.0, 1.0, 0),      // (-,+) -> 0
                _ => unreachable!(),
            };
            for _ in 0..n_per_quadrant {
                let r1 = lcg_f64(&mut state) * 0.8; // noise in [0, 0.8]
                let r2 = lcg_f64(&mut state) * 0.8;
                x.push(sx * (0.5 + r1));
                x.push(sy * (0.5 + r2));
                y.push(label);
            }
        }
        (x, y)
    }

    #[test]
    fn test_svm_rbf_kernel_separable() {
        // XOR data is NOT linearly separable. RBF SVM should handle it.
        let (x, y) = make_xor_data(50, 42);
        let n = y.len();
        let p = 2;

        let gamma = default_gamma_scale(&x, n, p);
        let kernel = KernelType::RBF { gamma };
        let mut clf = SvmClassifier::with_kernel(10.0, 1e-3, 2000, kernel);
        clf.fit(&x, &y, n, p, None).unwrap();

        let preds = clf.predict(&x, n, p);
        let acc = preds.iter().zip(y.iter()).filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        assert!(acc >= 0.85, "RBF SVM XOR accuracy {acc} < 0.85 (kill gate)");
    }

    #[test]
    fn test_svm_rbf_n_over_5000_errors() {
        // n=6000 with RBF kernel must return clear error
        let n = 6000;
        let p = 2;
        let x = vec![0.0f64; n * p];
        let y: Vec<i64> = (0..n).map(|i| (i % 2) as i64).collect();

        let kernel = KernelType::RBF { gamma: 1.0 };
        let mut clf = SvmClassifier::with_kernel(1.0, 1e-3, 100, kernel);
        let result = clf.fit(&x, &y, n, p, None);
        assert!(result.is_err(), "expected error for n=6000 RBF SVM");
        let msg = result.unwrap_err();
        assert!(msg.contains("5000"), "error message should mention limit: {msg}");
    }

    #[test]
    fn test_svm_rbf_gamma_effect() {
        // Smaller gamma = smoother boundary = fewer support vectors for well-separated data.
        // Larger gamma = more complex boundary.
        let (x, y) = make_xor_data(30, 77);
        let n = y.len();
        let p = 2;

        // Small gamma: very smooth
        let kernel_small = KernelType::RBF { gamma: 0.01 };
        let mut clf_small = SvmClassifier::with_kernel(10.0, 1e-3, 1000, kernel_small);
        clf_small.fit(&x, &y, n, p, None).unwrap();

        // Large gamma: very peaky
        let kernel_large = KernelType::RBF { gamma: 100.0 };
        let mut clf_large = SvmClassifier::with_kernel(10.0, 1e-3, 1000, kernel_large);
        clf_large.fit(&x, &y, n, p, None).unwrap();

        // Both should produce valid predictions
        let preds_small = clf_small.predict(&x, n, p);
        let preds_large = clf_large.predict(&x, n, p);
        assert_eq!(preds_small.len(), n);
        assert_eq!(preds_large.len(), n);

        // The large-gamma model overfits → higher train accuracy
        let acc_large = preds_large.iter().zip(y.iter())
            .filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        let acc_small = preds_small.iter().zip(y.iter())
            .filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        // Large gamma on XOR should fit the training data better than tiny gamma
        assert!(acc_large >= acc_small,
            "large gamma acc {acc_large} should be >= small gamma acc {acc_small} on train data");
    }

    #[test]
    fn test_svm_poly_kernel_basic() {
        // Polynomial kernel should produce valid output on XOR data
        let (x, y) = make_xor_data(30, 99);
        let n = y.len();
        let p = 2;

        let kernel = KernelType::Polynomial { gamma: 1.0, coef0: 1.0, degree: 3 };
        let mut clf = SvmClassifier::with_kernel(10.0, 1e-3, 2000, kernel);
        clf.fit(&x, &y, n, p, None).unwrap();

        let preds = clf.predict(&x, n, p);
        assert_eq!(preds.len(), n);
        // Poly degree 3 should handle XOR reasonably
        let acc = preds.iter().zip(y.iter()).filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        assert!(acc > 0.60, "Poly SVM accuracy {acc} too low on XOR");
    }

    #[test]
    fn test_svm_rbf_support_vectors_stored() {
        // Kernel SVM must store support vector data for prediction
        let (x, y) = make_xor_data(25, 55);
        let n = y.len();
        let p = 2;

        let gamma = default_gamma_scale(&x, n, p);
        let kernel = KernelType::RBF { gamma };
        let mut clf = SvmClassifier::with_kernel(10.0, 1e-3, 2000, kernel);
        clf.fit(&x, &y, n, p, None).unwrap();

        // Must have at least some support vectors
        let n_sv = clf.n_support_vectors();
        assert!(n_sv > 0, "RBF SVM should have support vectors, got 0");
        assert!(n_sv <= n * clf.n_classes,
            "too many SVs: {n_sv} > n*k = {}", n * clf.n_classes);
    }

    #[test]
    fn test_svm_rbf_serialization_roundtrip() {
        // JSON serialize/deserialize must preserve kernel config and predictions
        let (x, y) = make_xor_data(25, 88);
        let n = y.len();
        let p = 2;

        let gamma = default_gamma_scale(&x, n, p);
        let kernel = KernelType::RBF { gamma };
        let mut clf = SvmClassifier::with_kernel(10.0, 1e-3, 2000, kernel.clone());
        clf.fit(&x, &y, n, p, None).unwrap();

        let json = clf.to_json().unwrap();
        let clf2 = SvmClassifier::from_json(&json).unwrap();

        // Kernel type must survive roundtrip (gamma may be recomputed during fit)
        assert_eq!(clf2.kernel, clf.kernel);

        // Predictions must match exactly
        let p1 = clf.predict(&x, n, p);
        let p2 = clf2.predict(&x, n, p);
        assert_eq!(p1, p2, "JSON roundtrip predictions differ for RBF SVM");

        // Probabilities must also match
        let pr1 = clf.predict_proba(&x, n, p);
        let pr2 = clf2.predict_proba(&x, n, p);
        for (a, b) in pr1.iter().zip(pr2.iter()) {
            assert!((a - b).abs() < 1e-12, "proba mismatch after roundtrip");
        }
    }

    #[test]
    fn test_svm_linear_no_cap() {
        // Linear SVM should work fine at n=6000 (no cap)
        let n = 6000;
        let p = 2;
        let mut state = 42u64;
        let x: Vec<f64> = (0..n * p).map(|_| lcg_f64(&mut state)).collect();
        let y: Vec<i64> = (0..n).map(|i| if x[i * p] > 0.5 { 1 } else { 0 }).collect();

        let mut clf = SvmClassifier::new(1.0, 1e-3, 500);
        let result = clf.fit(&x, &y, n, p, None);
        assert!(result.is_ok(), "linear SVM at n=6000 should work");

        let preds = clf.predict(&x, n, p);
        let acc = preds.iter().zip(y.iter()).filter(|(p, y)| *p == *y).count() as f64 / n as f64;
        assert!(acc > 0.80, "linear SVM n=6000 accuracy: {acc}");
    }

    #[test]
    fn test_svm_kernel_default_is_linear() {
        // Default kernel should be Linear
        let clf = SvmClassifier::new(1.0, 1e-3, 100);
        assert_eq!(clf.kernel, KernelType::Linear);

        let reg = SvmRegressor::new(1.0, 0.1, 1e-3, 100);
        assert_eq!(reg.kernel, KernelType::Linear);
    }

    #[test]
    fn test_svm_rbf_regressor() {
        // RBF kernel SVR on sinusoidal data
        let n = 100;
        let p = 1;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&v| v.sin()).collect();

        let gamma = default_gamma_scale(&x, n, p);
        let kernel = KernelType::RBF { gamma };
        let mut reg = SvmRegressor::with_kernel(10.0, 0.01, 1e-3, 1000, kernel);
        reg.fit(&x, &y, n, p, None).unwrap();

        let preds = reg.predict(&x, n, p);
        let rmse: f64 = (preds.iter().zip(y.iter())
            .map(|(p, y)| (p - y).powi(2)).sum::<f64>() / n as f64).sqrt();
        // RBF SVR should fit sin() reasonably well
        assert!(rmse < 0.5, "RBF SVR RMSE {rmse} on sin() too high");
    }

    #[test]
    fn test_svm_rbf_n5000_otf_no_error() {
        // n=5000 with RBF should work (on-the-fly, no cache)
        // Keep data small to avoid slow test
        let n = 5000;
        let p = 1;
        let x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let y: Vec<i64> = (0..n).map(|i| if x[i] > 0.5 { 1 } else { 0 }).collect();

        let kernel = KernelType::RBF { gamma: 1.0 };
        let mut clf = SvmClassifier::with_kernel(1.0, 1e-2, 50, kernel);
        // Should not error at n=5000 (exactly at boundary)
        let result = clf.fit(&x, &y, n, p, None);
        assert!(result.is_ok(), "RBF SVM at n=5000 should succeed");
    }

    // ------------------------------------------------------------------
    // Shrinking tests — verify large-n convergence and correctness
    // ------------------------------------------------------------------

    #[test]
    fn test_shrinking_large_n_convergence() {
        // n=5000 binary clf: without shrinking this would hit the iteration cap
        // and produce ~50% accuracy. With shrinking, solver converges properly.
        // Uses train/test split: train on first 4000, test on last 1000.
        let n_total = 5000;
        let p = 10;
        let (x, y) = make_clf_data(n_total, p, 42);

        let n_train = 4000;
        let n_test = n_total - n_train;
        let x_train = &x[..n_train * p];
        let y_train: Vec<i64> = y[..n_train].to_vec();
        let x_test = &x[n_train * p..];
        let y_test = &y[n_train..];

        let mut clf = SvmClassifier::new(1.0, 1e-3, 500);
        clf.fit(x_train, &y_train, n_train, p, None).unwrap();

        let preds = clf.predict(x_test, n_test, p);
        let acc = preds.iter().zip(y_test.iter())
            .filter(|(p, y)| *p == *y).count() as f64 / n_test as f64;
        // With shrinking, solver converges → test accuracy > 0.85
        assert!(acc > 0.85, "shrinking n=5000 test accuracy: {acc} < 0.85");
    }

    #[test]
    fn test_shrinking_small_n_correctness() {
        // n=100: shrinking must be transparent — no accuracy regression vs
        // the pre-shrinking implementation. Small n means shrinking barely
        // activates, verifying the baseline path is unbroken.
        let n = 100;
        let p = 5;
        let (x, y) = make_clf_data(n, p, 77);

        let n_train = 80;
        let n_test = n - n_train;
        let x_train = &x[..n_train * p];
        let y_train: Vec<i64> = y[..n_train].to_vec();
        let x_test = &x[n_train * p..];
        let y_test = &y[n_train..];

        let mut clf = SvmClassifier::new(1.0, 1e-3, 200);
        clf.fit(x_train, &y_train, n_train, p, None).unwrap();

        let preds = clf.predict(x_test, n_test, p);
        let acc = preds.iter().zip(y_test.iter())
            .filter(|(p, y)| *p == *y).count() as f64 / n_test as f64;
        // Small n should still achieve reasonable accuracy
        assert!(acc > 0.70, "shrinking n=100 test accuracy: {acc} < 0.70");
    }

    #[test]
    fn test_shrinking_kernel_rbf() {
        // RBF kernel + shrinking on XOR-like data (non-linearly separable).
        // Verifies kernel reconstruction path works correctly.
        let (x, y) = make_xor_data(75, 42);  // 300 samples total
        let n = y.len();
        let p = 2;

        // Split: first 240 train, last 60 test
        let n_train = 240;
        let n_test = n - n_train;
        let x_train = &x[..n_train * p];
        let y_train: Vec<i64> = y[..n_train].to_vec();
        let x_test = &x[n_train * p..];
        let y_test = &y[n_train..];

        let gamma = default_gamma_scale(x_train, n_train, p);
        let kernel = KernelType::RBF { gamma };
        let mut clf = SvmClassifier::with_kernel(10.0, 1e-3, 2000, kernel);
        clf.fit(x_train, &y_train, n_train, p, None).unwrap();

        let preds = clf.predict(x_test, n_test, p);
        let acc = preds.iter().zip(y_test.iter())
            .filter(|(p, y)| *p == *y).count() as f64 / n_test as f64;
        assert!(acc > 0.75, "shrinking RBF test accuracy: {acc} < 0.75");
    }
}
