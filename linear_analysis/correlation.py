import numpy as np


def lagged_correlation(x, y, normalize=True):
    """numpy.correlate, non partial. Implicitly `x.size == y.size`.
    lag is relative to `y`, so if lag is positive, `y` is shifted to the right,
    so the past of `y` is correlated to `x` (Y --> X),
    if lag is negative `y` is shifted to the left, so the past of
    `x` is correlated to y (X --> Y)"""
    if normalize:
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
    corr = np.correlate(x, y, "full") / x.size
    lags = np.arange(corr.size) - x.size + 1
    return lags, corr


def spatial_correlation(x, lag=1):
    """Compute the spatial correlation of the 2D array `x` as the Moran's
    I coefficient between Manahattan neighbours at distance `lag`.
    NaN values are discarted from the calculation.

    Args:
        x (array-like): 2D spatial field
        lag (int, optional): Interval between neihgbors. Defaults to 1.

    Returns:
        float: Spatial correlation of `x`
    """
    xmean = np.nanmean(x)
    xvar = np.nanvar(x)
    if not xvar:
        return 1
    dxx = x - xmean
    corr = dxx * (np.roll(x, lag, axis=0) - xmean)
    corr += dxx * (np.roll(x, -lag, axis=0) - xmean)
    corr += dxx * (np.roll(x, lag, axis=1) - xmean)
    corr += dxx * (np.roll(x, -lag, axis=1) - xmean)
    corr /= 4
    return np.nanmean(corr) / xvar
