//! L-BFGS optimizer (Nocedal-Wright Algorithm 7.4).
//!
//! Port of `ml._logistic._lbfgs` (Python reference implementation).
//! Internal kernel — not part of the public ml API (`pub(crate)` only).
//!
//! # Algorithm parameters (exact match to Python reference)
//! - m = 10        — history window size
//! - tol = 1e-4   — convergence: L2 norm of gradient < tol
//! - max_iter = 1000
//! - c1 = 1e-4    — Armijo sufficient-decrease constant
//! - c2 = 0.9     — weak Wolfe curvature constant (NOT strong Wolfe)
//! - max_ls = 20  — max line-search steps (step halving)

use nalgebra::DVector;
use std::collections::VecDeque;

/// Run L-BFGS minimization.
///
/// Returns the parameter vector at the minimum found within `max_iter` iterations.
/// Does not signal convergence failure — returns best iterate found (mirrors Python).
///
/// # Arguments
/// - `f_grad`: closure returning `(loss, gradient)` at a given parameter vector
/// - `x0`: initial parameter vector
/// - `m`: history window size (typically 10)
/// - `max_iter`: maximum number of iterations
/// - `tol`: convergence tolerance on gradient L2 norm
pub(crate) fn lbfgs<F>(
    mut f_grad: F,
    x0: DVector<f64>,
    m: usize,
    max_iter: usize,
    tol: f64,
) -> DVector<f64>
where
    F: FnMut(&DVector<f64>) -> (f64, DVector<f64>),
{
    let mut x = x0;
    let (_f_val, mut g) = f_grad(&x);

    // History: VecDeque of (s, y, rho) triples.
    // push_back → back = newest, front = oldest.
    let mut history: VecDeque<(DVector<f64>, DVector<f64>, f64)> = VecDeque::with_capacity(m);

    for _ in 0..max_iter {
        if g.norm() < tol {
            break;
        }

        // Compute L-BFGS search direction
        let p = two_loop_direction(&g, &history);

        // Weak Wolfe line search
        let (alpha, g_new, f_new) = wolfe_line_search(&mut f_grad, &x, &p, _f_val, &g);
        let _ = f_new; // suppress warning — used only via g_new

        // Compute step and gradient difference
        let s = p.scale(alpha);
        let y = &g_new - &g;
        let sy = s.dot(&y);

        // Update position and gradient
        x += &s;
        g = g_new;

        // Add to history only when curvature condition holds (sy > 0 required for PD update)
        // Threshold 1e-10 exactly matches Python reference.  Do NOT use sy.abs() — negative
        // sy means the curvature condition is violated and the pair must be skipped.
        if sy > 1e-10 {
            if history.len() == m {
                history.pop_front();
            }
            history.push_back((s, y, 1.0 / sy));
        }
    }

    x
}

/// Two-loop L-BFGS recursion (Nocedal-Wright Algorithm 7.4).
///
/// # CRITICAL: iteration directions are OPPOSITE in the two loops.
/// - First loop : `history.iter().rev()` (newest → oldest)
/// - Second loop: `history.iter()`       (oldest → newest)
///
/// Swapping either direction produces a wrong search direction — silently wrong,
/// not a crash.  The `alphas` vec is built newest-first in loop 1 and consumed
/// oldest-first (via `.rev()`) in loop 2.
fn two_loop_direction(
    g: &DVector<f64>,
    history: &VecDeque<(DVector<f64>, DVector<f64>, f64)>,
) -> DVector<f64> {
    if history.is_empty() {
        return -g;
    }

    let n = history.len();
    let mut q = g.clone();
    // alphas[0] = alpha for newest entry; alphas[n-1] = alpha for oldest entry
    let mut alphas = vec![0.0_f64; n];

    // First loop: newest → oldest
    for (i, (s, y, rho)) in history.iter().rev().enumerate() {
        let a = rho * s.dot(&q);
        alphas[i] = a; // newest-first order
        q -= y.scale(a);
    }

    // Initial Hessian scaling: gamma = (s_last · y_last) / (y_last · y_last)
    // s_last/y_last = most recently accepted pair (back of deque)
    let (s_last, y_last, _) = history.back().unwrap();
    let yy = y_last.dot(y_last);
    let gamma = if yy > 0.0 {
        s_last.dot(y_last) / yy
    } else {
        1.0 // fallback: identity scaling (mathematically unreachable since sy > 1e-10 guard)
    };

    let mut r = q.scale(gamma);

    // Second loop: oldest → newest; alphas consumed in reversed order (oldest-first)
    for (i, (s, y, rho)) in history.iter().enumerate() {
        // alphas built newest-first → alphas[n-1-i] = alpha for history[i] (oldest-first)
        let a = alphas[n - 1 - i];
        let beta = rho * y.dot(&r);
        r += s.scale(a - beta);
    }

    -r // descent direction
}

/// Backtracking weak Wolfe line search.
///
/// Checks Armijo sufficient decrease + weak Wolfe curvature condition.
/// This is weak Wolfe (NOT strong): curvature check is `g_new · p >= c2 * slope0`,
/// not `|g_new · p| <= c2 * |slope0|`.  Matches Python reference exactly.
///
/// Fallback after max_ls: takes a tiny step alpha=1e-6 with a fresh f_grad call
/// (NOT stale g0 — stale gradient would corrupt the BFGS history update).
fn wolfe_line_search<F>(
    f_grad: &mut F,
    x: &DVector<f64>,
    p: &DVector<f64>,
    f0: f64,
    g0: &DVector<f64>,
) -> (f64, DVector<f64>, f64)
where
    F: FnMut(&DVector<f64>) -> (f64, DVector<f64>),
{
    const C1: f64 = 1e-4; // Armijo constant
    const C2: f64 = 0.9; // weak Wolfe curvature constant
    const MAX_LS: usize = 20;

    let slope0 = g0.dot(p); // negative (p is descent direction)
    let mut alpha = 1.0_f64;

    for _ in 0..MAX_LS {
        let x_new = x + p.scale(alpha);
        let (f_new, g_new) = f_grad(&x_new);

        if f_new <= f0 + C1 * alpha * slope0 {
            // Armijo satisfied; check weak Wolfe curvature
            if g_new.dot(p) >= C2 * slope0 {
                return (alpha, g_new, f_new);
            }
        }

        alpha *= 0.5;
    }

    // Fallback: tiny step — call f_grad for a FRESH gradient.
    // Do NOT reuse g0: stale gradient → y_k = g_new - g_old = 0 → history entry skipped.
    let alpha = 1e-6_f64;
    let x_new = x + p.scale(alpha);
    let (f_new, g_new) = f_grad(&x_new);
    (alpha, g_new, f_new)
}

// ---------------------------------------------------------------------------
// Unit tests (access pub(crate) and private items)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_lbfgs_quadratic() {
        // Minimize f(x) = 0.5 * ||x - target||^2.  Minimum at x = target.
        let target = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let t = target.clone();
        let f_grad = move |x: &DVector<f64>| {
            let diff = x - &t;
            let f = 0.5 * diff.dot(&diff);
            (f, diff)
        };
        let x0 = DVector::zeros(3);
        let result = lbfgs(f_grad, x0, 10, 1000, 1e-6);
        let error = (&result - &target).norm();
        assert!(error < 1e-5, "quadratic: error {error:.2e} >= 1e-5");
    }

    #[test]
    fn test_lbfgs_rosenbrock() {
        // Rosenbrock: f(x, y) = (1-x)^2 + 100*(y - x^2)^2.  Minimum at (1, 1).
        let f_grad = |x: &DVector<f64>| {
            let a = x[0];
            let b = x[1];
            let f = (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2);
            let gx = -2.0 * (1.0 - a) - 400.0 * a * (b - a * a);
            let gy = 200.0 * (b - a * a);
            (f, DVector::from_vec(vec![gx, gy]))
        };
        let x0 = DVector::from_vec(vec![-1.0, 1.0]);
        let result = lbfgs(f_grad, x0, 10, 1000, 1e-4);
        assert!(
            (result[0] - 1.0).abs() < 1e-3,
            "Rosenbrock x: {}",
            result[0]
        );
        assert!(
            (result[1] - 1.0).abs() < 1e-3,
            "Rosenbrock y: {}",
            result[1]
        );
    }

    #[test]
    fn test_two_loop_empty_history_returns_neg_gradient() {
        let g = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let history = VecDeque::new();
        let p = two_loop_direction(&g, &history);
        assert_eq!(p, -&g);
    }

    #[test]
    fn test_lbfgs_descent_property() {
        // After one L-BFGS step, loss should decrease (or stay the same).
        let target = DVector::from_vec(vec![5.0, -3.0]);
        let t = target.clone();
        let f_grad = move |x: &DVector<f64>| {
            let diff = x - &t;
            (0.5 * diff.dot(&diff), diff)
        };
        let x0 = DVector::zeros(2);
        let (f0, _) = f_grad(&x0);
        let x_opt = lbfgs(f_grad, x0, 10, 5, 1e-10); // 5 iterations
        let t2 = target.clone();
        let f_grad2 = move |x: &DVector<f64>| {
            let diff = x - &t2;
            (0.5 * diff.dot(&diff), diff)
        };
        let (f5, _) = f_grad2(&x_opt);
        assert!(f5 < f0, "loss did not decrease: {f0:.4} -> {f5:.4}");
    }
}
