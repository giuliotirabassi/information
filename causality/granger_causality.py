import numpy as np
from linear_analysis.ols import OLS
from scipy import stats
from embedding import embedding as emb


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
    x_emb = emb.time_delay_embedding(x, dim=dim + 1, lag=1)
    X = x_emb[:, :-1]
    x_future = x_emb[:, -1]
    Y = emb.time_delay_embedding(y, dim=dim, lag=1)
    Y = Y[: x_future.size]
    assert X.shape == (x.size - dim, dim)

    ols = OLS()
    ols.fit(X, x_future)
    n, p = X.shape
    rss1 = ols.get_fit_result()["RSS"]
    ols.fit(np.hstack((X, Y)), x[dim:])
    rss2 = ols.get_fit_result()["RSS"]
    f = (rss1 - rss2) * (n - 2 * p) / (p * rss2)
    p = 1 - stats.f.cdf(f, p, n - 2 * p)
    return f, p


def schwartz_criterion(x, maxdim=20) -> int:
    """Bayesian information criterion for the dimension of AR modelling
    of time series `x`."""
    if x.size <= maxdim:
        raise ValueError("Decrease maxdim below x.size")
    ols = OLS(intercept=False)
    s = []
    for dim in range(1, maxdim + 1):
        x_emb = emb.time_delay_embedding(x, dim=dim + 1, lag=1)
        X = x_emb[:, :-1]
        x_future = x_emb[:, -1]
        ols.fit(X, x_future)
        rss = ols.get_fit_result()["RSS"]
        n = X.shape[0]
        s.append(n * np.log(rss / n) + dim * np.log(n))
    s = (
        np.round(np.array(s) / 10) * 10
    )  # BIC should not consider significant log-likelyhood variations above 10
    return np.argmin(s) + 1
