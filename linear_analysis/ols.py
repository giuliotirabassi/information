from scipy.linalg import solve, inv
from scipy import stats
import numpy as np


class OLS(object):
    """Ordinary least squares y ~ intercept + coeffs * X
    the intercept coefficient (and p-value) is the first,
    the rest are for the columns of X"""

    def __init__(self, normalize_inputs=False, intercept=True):
        self._coeffs = None
        self._pvals = None
        self._normalize = normalize_inputs
        self._intercept = intercept

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape((X.size, 1))
        if self._intercept:
            X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        assert X.shape[0] == y.size
        n, p = X.shape
        if p >= n:
            raise ValueError(f"You need more than {X.shape[1]} datapoints")
        ym = y.mean()
        if self._normalize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            y = (y - ym) / y.std()
        XX = X.T.dot(X)
        self._coeffs = solve(XX, X.T.dot(y))
        y_pred = X.dot(self._coeffs)
        self._rss = ((y_pred - y) ** 2).sum()
        self._r2 = 1 - (self._rss / n) / np.var(y)
        self._s2 = self._rss / (n - p)
        se_coeffs = np.sqrt(self._s2 * np.diagonal(inv(XX)))
        t_coeffs = self._coeffs / se_coeffs
        self._pvals = (1 - stats.t.cdf(abs(t_coeffs), n - p)) * 2
        return self

    def get_fit_result(self):
        if self._coeffs is None:
            raise RuntimeError("Call fit first")
        return {
            "coeff": self._coeffs.copy(),
            "p-values": self._pvals,
            "RSS": self._rss,
            "R2": self._r2,
        }
