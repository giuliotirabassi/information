import numpy as np


def pseudo_transfer_entropy(x, y, dim=1):
    """Preudo-transfer Entropy from
    R Silini, C Masoller - Scientific reports, 2021

    This causality metric assumes that both `x` and `y` are gaussian
    processes

    Args:
        x (array): slave variable
        y (array): possibly forcing variable
        dim (int, optional): embedding of the past. Defaults to 1.

    Returns:
        float: pseudo Transfer Entropy between `x` and `y` (`y` --> `x`)
    """
    X = np.stack([x[i : i + dim] for i in range(x.size - dim)]).T  # noqa: E203
    assert X.shape == (dim, x.size - dim)
    Y = np.stack([y[i : i + dim] for i in range(x.size - dim)]).T  # noqa: E203
    cov1 = np.cov(np.vstack((X, Y)))
    assert cov1.shape == (2 * dim, 2 * dim)
    cov2 = np.cov(np.vstack((x[dim:], X)))
    cov3 = np.cov(np.vstack((x[dim:], X, Y)))
    cov4 = np.cov(X)
    return 0.5 * (
        np.log(np.linalg.det(cov1))
        + np.log(np.linalg.det(cov2))
        - np.log(np.linalg.det(cov3))
        - np.log(np.linalg.det(cov4) if cov4.size > 1 else cov4)
    )
