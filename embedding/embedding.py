import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)


def time_delay_embedding(x, dim, lag):
    embedded = []
    max_idx = (dim - 1) * lag
    for i in range(x.size - max_idx):
        embedded.append(x[i : i + max_idx + 1 : lag])
    return np.stack(embedded)


def cao_criterion(x, tau, max_dim=10, rtol=0.05):
    """Compute the minimumm embedding dimension using the false nearest neighbour
    algorithm variation from Cao 1997. The embedding delay `tau`must be determined
    separately, for example with `compute_autocorrelation_criterion`.

    Args:
        x (array): Time series to embed.
        tau (int): Embedding delay.
        max_dim (int, optional): Maximum embedding dimension. Defaults to 10.
        rtol (float, optional): Relative variation of neighbours distance.
            Defaults to 0.05.

    Raises:
        ValueError: If the relative distance variation does not converge below `rtol`.

    Returns:
        int: Minimum embedding dimension of `x`.
    """
    mean_distances = []
    for dim in range(1, max_dim + 1):
        embx = time_delay_embedding(x, dim, tau)
        embxx = time_delay_embedding(x, dim + 1, tau)
        nndist, nnindex = (
            NearestNeighbors(n_neighbors=1, metric="euclidean").fit(embx).kneighbors()
        )
        distances = []
        for i in range(embxx.shape[0]):
            if nnindex[i] > embxx.shape[0] - 1:
                continue
            dist = np.sqrt(((embxx[i] - embxx[nnindex[i]]) ** 2).sum())
            distances.append(dist / nndist[i])
        distances = np.mean(distances)
        mean_distances.append(distances)
    mean_distances = np.array(mean_distances)
    variation = mean_distances[1:] / mean_distances[:-1]
    for i in range(variation.size):
        if abs(variation[i + 1] - variation[i]) / variation[i] < rtol:
            return i + 2
    raise ValueError("Increase max dimension")


def compute_autocorrelation_criterion(x, criterion="zero"):
    """Compute the autocorrelation length of `x`. If `criterion`is "zero",
    the autocorrelation lenght is the lag at which the autocorrelation becomes
    negative for the first time. If `criterion`is "min" the autocorrelation
    length is the lag for which the series autocorrelation has the first local minimum.
    Note that "min" is rather unstable.

    Args:
        x (nd.array): time series equally sampled
        criterion (str, optional): Criterion to choose the autocorrelation
            length. Defaults to "zero".

    Raises:
        ValueError: If the criterion is not either zero or min

    Returns:
        int: autocorrelation length
    """
    if criterion not in {"zero", "min"}:
        raise ValueError("Unknown value for criterion: try either 'zero' or 'min'")
    old_r = 1
    for i in range(1, x.size):
        r, _ = pearsonr(x[i:], x[:-i])
        if criterion == "zero" and r < 0:
            return i
        elif criterion == "min" and r > old_r and r < 0:
            return i - 1
        old_r = r
    return None


def compute_ragwitz_criterion(x, max_dim=10, n_neighbors=4, verbose=False):
    """Returns the optimal embedding dimension using Ragwitz criterion

    Reference:
        Ragwitz and Kantz (2002). PRE, 65

    Args:
        x (nd.array): input time series
        max_dim (int, optional): Maximum embedding dimension. Defaults to 10.
        n_neighbors (int, optional): Number of neigbours to use to determine
            the optimal dimension. Defaults to 4.

    Returns:
        int: Optimum embedding dimension
    """
    errors = []
    for dim in range(1, max_dim + 1):
        x_past = []
        for i in range(dim, x.size):
            x_past.append(x[i - dim : i])  # noqa: E203
        x_past = np.stack(x_past)
        x_future = x[dim:]
        kdt_xpast = KDTree(x_past, metric="euclidean")
        indices = kdt_xpast.query(x_past, k=n_neighbors, return_distance=False)
        indices = indices[:, 1:]  # discarding the point itself
        prediction = x_future[indices].mean(axis=1)
        error = ((prediction - x_future) ** 2).mean() ** 0.5
        errors.append(error)
    dim = np.argmin(errors) + 1
    if dim == max_dim:
        logger.warning(
            "The result is max_dim: consider increasing the max_dim input value"
        )
    if verbose:
        print(errors)
    return dim
