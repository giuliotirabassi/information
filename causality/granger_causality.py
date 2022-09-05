import numpy as np
from linear_analysis.ols import OLS
from scipy import stats


def granger_causality(x, y, dim=1):
    """Granger causality test between `x`and `y` (`y`-->`x`).
    Inputer have to be numpy arrays.
    `dim`represents the order of the AR process modelling `x`.
    If `None`, it will be determined using Schwartz criterios (BIC).

    Args:
        x (array): slave series
        y (_type_): possibly forcing series
        dim (int, optional): AR dimansion of `x`. Defaults to 1.

    Returns:
        tuple: F statistics of the test and its p-value
    """
    if not dim:
        dim = schwartz_criterion(x)

    X = np.stack([x[i : i + dim] for i in range(x.size - dim)])
    Y = np.stack([y[i : i + dim] for i in range(x.size - dim)])
    assert X.shape == (x.size - dim, dim)

    ols = OLS()
    ols.fit(X, x[dim:])
    n, p = X.shape
    rss1 = ols.get_fit_result()["RSS"]
    ols.fit(np.hstack((X, Y)), x[dim:])
    rss2 = ols.get_fit_result()["RSS"]
    f = (rss1 - rss2) * (n - 2 * p) / (p * rss2)
    p = 1 - stats.f.cdf(f, p, n - 2 * p)
    return f, p


def schwartz_criterion(x, maxdim=20):
    """Bayesian information criterion for the dimension of AR modelling
    of time series `x`."""
    if x.size <= maxdim:
        raise ValueError("Decrease maxdim below x.size")
    ols = OLS(intercept=False)
    s = []
    for dim in range(1, maxdim):
        X = np.stack([x[i : i + dim] for i in range(x.size - dim)])
        ols.fit(X, x[dim:])
        r2 = ols.get_fit_result()["R2"]
        s.append(-x.size * np.log(r2) + dim * np.log(x.size))
    s = (
        np.round(np.array(s) / 10) * 10
    )  # BIC should not consider significant log-likelyhood variations above 10
    return np.argmin(s) + 1
