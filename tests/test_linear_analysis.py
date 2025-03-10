from linear_analysis.correlation import spatial_correlation, lagged_correlation
from linear_analysis.ols import OLS
import numpy as np

random = np.random.RandomState(0)


def test_spatial_correlation():
    X = random.rand(100, 100)

    sc = spatial_correlation(X)
    sc1 = spatial_correlation(X, lag=1)
    sc2 = spatial_correlation(X, lag=2)
    for i, j in random.randint(0, 99, size=(30, 2)).tolist():
        X[i, j] = np.nan
    sc3 = spatial_correlation(X, lag=2)
    assert sc == sc1
    assert np.isclose(sc, 0, atol=0.01)
    assert np.isclose(sc2, 0, atol=0.01)
    assert np.isclose(sc3, 0, atol=0.01)


def test_lagged_correlation():
    t = np.linspace(0, 100 * np.pi, 10000)
    x = np.sin(t)
    y = np.cos(t)
    lags, corr = lagged_correlation(x, y)
    corr = corr[lags >= 0]
    lags = lags[lags >= 0]
    max_lag = np.argmax(corr)
    min_lag = np.argmin(corr)
    assert np.isclose(t[lags[max_lag]], np.pi / 2, rtol=0.01)
    assert np.isclose(t[lags[min_lag]], 3 * np.pi / 2, rtol=0.01)


def test_ols():
    x = np.arange(1, 100)
    y = 3 * x + 5
    ols = OLS().fit(x, y)
    assert tuple(ols._coeffs) == (5, 3)
    assert ols.get_fit_result()
    y = y + random.normal(size=y.size)
    ols.fit(x, y)
    assert np.all(ols._pvals < 0.05)
    assert 0 < ols._r2 < 1
    ols = OLS(intercept=False).fit(x, y)
    assert ols._pvals.size == 1
    ols.fit(x, random.normal(size=x.size))
    assert ols._pvals[0] > 0.4
    assert 0 < ols._r2 < 1
