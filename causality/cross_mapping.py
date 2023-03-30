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
    X = time_delay_embedding(x, dim=dim, lag=lag)
    neigh = NearestNeighbors(n_neighbors=dim + 1, radius=np.inf).fit(X)
    distances, neigh_index = neigh.kneighbors(return_distance=True)
    distances = distances[:, 1:]
    neigh_index = neigh_index[:, 1:]
    u = np.exp(-distances.T / distances[:, 0]).T
    w = (u.T / np.sum(u, axis=1)).T
    y_hat = (w * np.take(y, neigh_index)).sum(axis=1)
    return pearsonr(y[: y_hat.size], y_hat)
