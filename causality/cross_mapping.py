from embedding.embedding import time_delay_embedding
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import pearsonr


def convergent_cross_mapping(y, x, dim=2, lag=1):
    """Convergent Cross Mapping testing x --> y.
    The first argument is the slave variable. The second argument
    is the master variable. Based on
    Sugihara, G. et al., 2012. Science
    and
    https://phdinds-aim.github.io/time_series_handbook/06_ConvergentCrossMappingandSugiharaCausality/ccm_sugihara.html

    Args:
        y (array like): slave variable
        x (array like): master variable
        dim (int, optional): embedding dimension of x. Defaults to 2.
        lag (int, optional): embedding lag of x. Defaults to 1.

    Returns:
        tuple: causality pearson coefficient and relative p-value.
    """
    x = np.array(x)[::-1]  # future in the beginning past in the end
    y = np.array(y)[::-1]
    assert x.size == y.size
    y_hat = _compute_cross_mapping(x, y, dim, lag)
    return pearsonr(y[: y_hat.size], y_hat)


def _compute_cross_mapping(x, y, dim, lag):
    X = time_delay_embedding(x, dim=dim, lag=lag)
    neigh = NearestNeighbors(n_neighbors=dim + 1, radius=np.inf).fit(X)
    distances, neigh_index = neigh.kneighbors(return_distance=True)
    u = np.exp(-distances.T / distances[:, 0]).T
    w = (u.T / np.sum(u, axis=1)).T
    y_hat = (w * np.take(y, neigh_index)).sum(axis=1)
    return y_hat


def partial_cross_mapping(x, y, z, dim, maxlag):
    zhat = _find_most_correlated_mapping(x, z, dim, maxlag)
    yhathat = _find_most_correlated_mapping(zhat, y, dim, maxlag)
    yhat = _find_most_correlated_mapping(x, y, dim, maxlag)
    return _partial_cross_correlation(y, yhat, yhathat)


def _corr(x, y):
    length = min(x.size, y.size)
    return pearsonr(x[:length], y[:length])[0]


def _partial_cross_correlation(x, y, z):
    return (_corr(x, y) - _corr(x, z) * _corr(y, z)) / np.sqrt(
        (1 - _corr(x, z) ** 2) * (1 - _corr(y, z) ** 2)
    )


def _find_most_correlated_mapping(x, z, dim, maxlag):
    """Most correlated mapping Z --> X"""
    zhat = None
    corr_zhat = 0
    for lag in range(1, maxlag):
        zhatdummy = _compute_cross_mapping(x, z, dim=dim, lag=lag)
        corr, _ = pearsonr(z[: zhatdummy.size], zhatdummy)
        if abs(corr) > corr_zhat:
            corr_zhat = abs(corr)
            zhat = zhatdummy
    return zhat


if __name__ == "__main__":
    rx = 3.6
    ry = 3.72
    rz = 3.68
    betayz = 0.35
    betazx = 0.35
    maxL = 1000
    X = np.zeros(maxL)
    Y = np.zeros(maxL)
    Z = np.zeros(maxL)
    X[0] = 0.4
    Y[0] = 0.4
    Z[0] = 0.4
    for j in range(1, maxL):
        X[j] = X[j - 1] * (rx - rx * X[j - 1]) + np.random.normal(0, 0.005)
        Y[j] = Y[j - 1] * (ry - ry * Y[j - 1] - betayz * Z[j - 1]) + np.random.normal(
            0, 0.005
        )
        Z[j] = Z[j - 1] * (rz - rz * Z[j - 1] - betazx * X[j - 1]) + np.random.normal(
            0, 0.005
        )
    E = 4
    tau = 1
    print(partial_cross_mapping(Y, X, Z, dim=E, maxlag=10))
    print(convergent_cross_mapping(Y, X, dim=E, lag=1))
