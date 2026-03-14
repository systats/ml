//! Shared utilities for the AL engine.

use std::collections::HashMap;

/// Compute per-sample weights from class labels using "balanced" mode.
///
/// Formula: weight_c = n_total / (n_classes * n_c)
/// where n_c = number of samples in class c.
///
/// If `existing_sw` is provided, returns element-wise product.
/// Otherwise returns the class weights directly.
pub fn balanced_class_weights(y: &[i64], existing_sw: Option<&[f64]>) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }

    // Count per-class frequencies
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &label in y {
        *counts.entry(label).or_insert(0) += 1;
    }
    let k = counts.len() as f64;
    let n_f = n as f64;

    // Compute per-sample weights
    let class_w: Vec<f64> = y
        .iter()
        .map(|&label| {
            let n_c = counts[&label] as f64;
            n_f / (k * n_c)
        })
        .collect();

    match existing_sw {
        Some(sw) => class_w
            .iter()
            .zip(sw.iter())
            .map(|(&cw, &sw)| cw * sw)
            .collect(),
        None => class_w,
    }
}

/// Compute per-sample weights from custom per-class weight map.
///
/// Classes not in the map get weight 1.0.
pub fn custom_class_weights(
    y: &[i64],
    weights: &HashMap<i64, f64>,
    existing_sw: Option<&[f64]>,
) -> Vec<f64> {
    let class_w: Vec<f64> = y
        .iter()
        .map(|&label| *weights.get(&label).unwrap_or(&1.0))
        .collect();

    match existing_sw {
        Some(sw) => class_w
            .iter()
            .zip(sw.iter())
            .map(|(&cw, &sw)| cw * sw)
            .collect(),
        None => class_w,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balanced_weights_equal_classes() {
        // 3 samples per class, 2 classes → weight = 6 / (2 * 3) = 1.0
        let y = vec![0i64, 0, 0, 1, 1, 1];
        let w = balanced_class_weights(&y, None);
        assert_eq!(w.len(), 6);
        for &v in &w {
            assert!((v - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_balanced_weights_imbalanced() {
        // 1 sample class 0, 3 samples class 1 → n=4, k=2
        // class 0: 4 / (2*1) = 2.0
        // class 1: 4 / (2*3) = 0.667
        let y = vec![0i64, 1, 1, 1];
        let w = balanced_class_weights(&y, None);
        assert!((w[0] - 2.0).abs() < 1e-9);
        assert!((w[1] - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_balanced_with_existing_sw() {
        let y = vec![0i64, 1, 1, 1];
        let sw = vec![1.0, 2.0, 0.5, 1.0];
        let w = balanced_class_weights(&y, Some(&sw));
        // class 0 weight=2.0, existing=1.0 → 2.0
        assert!((w[0] - 2.0).abs() < 1e-9);
        // class 1 weight=2/3, existing=2.0 → 4/3
        assert!((w[1] - 4.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_balanced_empty() {
        let y: Vec<i64> = vec![];
        let w = balanced_class_weights(&y, None);
        assert!(w.is_empty());
    }

    #[test]
    fn test_custom_weights() {
        let y = vec![0i64, 1, 2, 1];
        let mut map = HashMap::new();
        map.insert(0, 10.0);
        map.insert(1, 0.5);
        // class 2 not in map → default 1.0
        let w = custom_class_weights(&y, &map, None);
        assert!((w[0] - 10.0).abs() < 1e-9);
        assert!((w[1] - 0.5).abs() < 1e-9);
        assert!((w[2] - 1.0).abs() < 1e-9);
        assert!((w[3] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_custom_with_existing_sw() {
        let y = vec![0i64, 1];
        let mut map = HashMap::new();
        map.insert(0, 2.0);
        map.insert(1, 3.0);
        let sw = vec![0.5, 0.5];
        let w = custom_class_weights(&y, &map, Some(&sw));
        assert!((w[0] - 1.0).abs() < 1e-9);
        assert!((w[1] - 1.5).abs() < 1e-9);
    }
}
