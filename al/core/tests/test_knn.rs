use ml::knn::KnnModel;

/// Simple 2D classification: 3 classes in distinct clusters.
#[test]
fn test_knn_clf_basic() {
    // Class 0: around (0, 0)
    // Class 1: around (10, 10)
    // Class 2: around (20, 0)
    let x: Vec<f64> = vec![
        0.0, 0.0,
        0.1, 0.1,
        0.2, -0.1,
        10.0, 10.0,
        10.1, 10.1,
        9.9, 10.2,
        20.0, 0.0,
        20.1, 0.1,
        19.9, -0.1,
    ];
    let y: Vec<i64> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

    let mut model = KnnModel::new(3);
    model.fit_clf(&x, 9, 2, &y).unwrap();

    // Query points near each cluster center
    let query = vec![
        0.05, 0.05,   // should be class 0
        10.05, 10.05, // should be class 1
        20.05, 0.05,  // should be class 2
    ];
    let preds = model.predict_clf(&query, 3, 2);
    assert_eq!(preds, vec![0, 1, 2]);
}

#[test]
fn test_knn_clf_proba() {
    let x: Vec<f64> = vec![
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        3.0, 0.0,
        4.0, 0.0,
    ];
    let y: Vec<i64> = vec![0, 0, 0, 1, 1];

    let mut model = KnnModel::new(3);
    model.fit_clf(&x, 5, 2, &y).unwrap();

    // Query at x=0.5: 3 nearest are indices 0,1,2 → all class 0
    let query = vec![0.5, 0.0];
    let proba = model.predict_proba(&query, 1, 2);
    assert_eq!(proba.len(), 2); // 2 classes
    assert!((proba[0] - 1.0).abs() < 1e-10); // class 0 = 100%
    assert!((proba[1] - 0.0).abs() < 1e-10); // class 1 = 0%
}

#[test]
fn test_knn_reg_basic() {
    // Simple: y = x
    let x: Vec<f64> = vec![
        1.0, 0.0,
        2.0, 0.0,
        3.0, 0.0,
        4.0, 0.0,
        5.0, 0.0,
    ];
    let y: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let mut model = KnnModel::new(3);
    model.fit_reg(&x, 5, 2, &y).unwrap();

    // Query at x=3.0: 3 nearest are 2,3,4 → mean(3,2,4) = 3.0
    let query = vec![3.0, 0.0];
    let preds = model.predict_reg(&query, 1, 2);
    assert!((preds[0] - 3.0).abs() < 1e-10);
}

#[test]
fn test_knn_empty_data() {
    let mut model = KnnModel::new(3);
    let result = model.fit_clf(&[], 0, 2, &[]);
    assert!(result.is_err());
}

#[test]
fn test_knn_dimension_mismatch() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![0i64, 1, 0]; // 3 labels but 2 rows
    let mut model = KnnModel::new(3);
    let result = model.fit_clf(&x, 2, 2, &y);
    assert!(result.is_err());
}

#[test]
fn test_knn_serde_roundtrip() {
    let x: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    let y: Vec<i64> = vec![0, 1, 0];

    let mut model = KnnModel::new(1);
    model.fit_clf(&x, 3, 2, &y).unwrap();

    let json = model.to_json().unwrap();
    let loaded = KnnModel::from_json(&json).unwrap();

    let query = vec![0.1, 0.1];
    let orig_pred = model.predict_clf(&query, 1, 2);
    let loaded_pred = loaded.predict_clf(&query, 1, 2);
    assert_eq!(orig_pred, loaded_pred);
}

/// Test that brute-force mode kicks in for high-dimensional data (d > 20).
#[test]
fn test_knn_high_dim_brute_force() {
    let d = 25;
    let n = 10;
    let mut x = vec![0.0f64; n * d];
    let mut y = vec![0i64; n];

    // Create two clusters: first 5 at origin, last 5 at 10.0
    for i in 5..n {
        for j in 0..d {
            x[i * d + j] = 10.0;
        }
        y[i] = 1;
    }

    let mut model = KnnModel::new(3);
    model.fit_clf(&x, n, d, &y).unwrap();

    // Query near origin → class 0
    let query = vec![0.1; d];
    let preds = model.predict_clf(&query, 1, d);
    assert_eq!(preds[0], 0);

    // Query near 10 → class 1
    let query2 = vec![9.9; d];
    let preds2 = model.predict_clf(&query2, 1, d);
    assert_eq!(preds2[0], 1);
}
