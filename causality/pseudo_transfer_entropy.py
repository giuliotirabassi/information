import numpy as np
from scipy.signal import detrend


def pseudo_transfer_entropy(x, y, dim=1, tau=1, normalize=False):
    """Preudo-transfer Entropy from
    R Silini, C Masoller - Scientific reports, 2021

    This causality metric assumes that both `x` and `y` are gaussian
    processes.

    `x`is the slave variable, `y` the master.
    `dim` is the amount of points in the past of `x`and `y`that contain information
    about the future of f
    `tau` represents the timesteps in the future to assess the influence of `y`.
    If `normalize=True` the inputs are linearly detrended and normalized at 0 mean
    and unitary variance.

    Args:
        x (array): slave variable
        y (array): possibly forcing variable
        dim (int, optional): embedding of the past. Defaults to 1.
        normalize (bool, optional): whether to normalize inputs. Defaults to False

    Returns:
        float: pseudo Transfer Entropy between `x` and `y` (`y` --> `x`)
    """
    if normalize:
        x = detrend(x)
        y = detrend(y)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
    offset = dim + tau - 1
    n = x.size - offset
    X = np.stack([x[i : i + dim] for i in range(n)]).T  # noqa: E203
    assert X.shape == (dim, n)
    Y = np.stack([y[i : i + dim] for i in range(n)]).T  # noqa: E203
    cov1 = np.cov(np.vstack((X, Y)))
    assert cov1.shape == (2 * dim, 2 * dim)
    cov2 = np.cov(np.vstack((x[offset:], X)))
    cov3 = np.cov(np.vstack((x[offset:], X, Y)))
    cov4 = np.cov(X)
    return 0.5 * (
        np.log(np.linalg.det(cov1))
        + np.log(np.linalg.det(cov2))
        - np.log(np.linalg.det(cov3))
        - np.log(np.linalg.det(cov4) if cov4.size > 1 else cov4)
    )
