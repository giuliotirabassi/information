from embedding.embedding import time_delay_embedding
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import pearsonr


def convergent_cross_mapping(x, y, dim=2, lag=1):
    """Convergent Cross Mapping testing x --> y.
    The first argument is the slave variable. The second argument
    is the master variable. Based on
    Sugihara, G. et al., 2012. Science
    and
    https://phdinds-aim.github.io/time_series_handbook/06_ConvergentCrossMappingandSugiharaCausality/ccm_sugihara.html
    It is called convergent cross mapping because its output
    should converge to 1 as the length of the time series in
    increased if there is a real causation `x` --> `y`.
    The idea is that if `x` causes `y`, the we can use the shadow
    manifold of `y` to predict `x`.

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
    """Use the embedding of `x` to predict `y`."""
    X = time_delay_embedding(x, dim=dim, lag=lag)
    neigh = NearestNeighbors(n_neighbors=dim + 1, radius=np.inf).fit(X)
    distances, neigh_index = neigh.kneighbors(return_distance=True)
    u = np.exp(-distances.T / distances[:, 0]).T
    w = (u.T / np.sum(u, axis=1)).T
    y_hat = (w * np.take(y, neigh_index)).sum(axis=1)
    return y_hat


def partial_cross_mapping(x, y, z, dim, maxlag):
    """Partial CrossMapping From Leng et al. 2020 Nat. Comm.
    Cross mapping Y -> X conditioning out the Y -> Z -> X case.
    Unlike convergent_cross_mapping has no p-value.

    Args:
        x (array): Slave series
        y (array): Master series
        z (array): Intermediary series
        dim (int): Embedding dimension of `x`
        maxlag (int): Maximum lag to span in the intermediate mappings

    Returns:
        float: partial cross mapping of Y onto X
    """
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
    print("#### INTEGRATION TEST FOR PARTIAL CROSS MAPPING")
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
    print("First value should be way lower than second")
    print(partial_cross_mapping(Y, X, Z, dim=E, maxlag=10))
    print(convergent_cross_mapping(Y, X, dim=E, lag=1))

    print("#### INTEGRATION TEST FOR CONVERGENT CROSS MAPPING")

    def func_1(A, B, r, beta):
        return A * (r - r * A - beta * B)

    # Initialize test dataset
    # params
    r_x = 3.7
    r_y = 3.7
    B_xy = 0  # effect on x given y (effect of y on x)
    B_yx = 0.32  # effect on y given x (effect of x on y)

    X0 = 0.2  # initial val following Sugihara et al
    Y0 = 0.4  # initial val following Sugihara et al
    t = 5000  # time steps

    X = [X0]
    Y = [Y0]
    for i in range(t):
        X_ = func_1(X[-1], Y[-1], r_x, B_xy)
        Y_ = func_1(Y[-1], X[-1], r_y, B_yx)
        X.append(X_)
        Y.append(Y_)
    print("Left column should converge to 1, right column should go to 0")
    for i in [50, 100, 200, 300, 500, 1000, 3000, 5000]:
        print(
            convergent_cross_mapping(Y[:i], X[:i], dim=E, lag=1)[0],
            convergent_cross_mapping(X[:i], Y[:i], dim=E, lag=1)[0],
        )
