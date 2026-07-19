import numpy as np

from farm_us.training.metrics import TargetScaler, regression_metrics


def test_regression_metrics_perfect():
    y = np.linspace(0, 10, 50)
    m = regression_metrics(y, y, np.ones_like(y))
    assert abs(m["mae"]) < 1e-9
    assert abs(m["rmse"]) < 1e-9
    assert abs(m["r2"] - 1.0) < 1e-9
    assert abs(m["pearson_r"] - 1.0) < 1e-9


def test_regression_metrics_matches_sklearn():
    rng = np.random.default_rng(0)
    t = rng.normal(size=200)
    p = t + rng.normal(scale=0.3, size=200)
    m = regression_metrics(p, t, np.ones_like(t))
    try:
        from sklearn.metrics import mean_absolute_error, r2_score

        assert abs(m["mae"] - mean_absolute_error(t, p)) < 1e-8
        assert abs(m["r2"] - r2_score(t, p)) < 1e-8
    except ImportError:
        pass


def test_mask_excludes_pixels():
    p = np.array([0.0, 0.0, 100.0])
    t = np.array([0.0, 0.0, 0.0])
    mask = np.array([1, 1, 0])
    m = regression_metrics(p, t, mask)
    assert m["n"] == 2
    assert m["rmse"] == 0.0


def test_target_scaler_roundtrip():
    s = TargetScaler(mode="zscore", center=3000.0, scale=400.0)
    y = np.array([2600.0, 3000.0, 3400.0])
    z = s.transform(y)
    assert np.allclose(z, [-1, 0, 1])
    assert np.allclose(s.inverse(z), y)


def test_target_scaler_none():
    s = TargetScaler(mode="none")
    y = np.array([1.0, 2.0])
    assert np.allclose(s.transform(y), y)
    assert np.allclose(s.inverse(y), y)


def test_bias_sign():
    p = np.array([5.0, 5.0])
    t = np.array([3.0, 3.0])
    m = regression_metrics(p, t, np.ones(2))
    assert m["bias"] == 2.0
