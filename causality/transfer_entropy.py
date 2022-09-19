import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.special import digamma
from scipy.stats import pearsonr
import logging
from embedding import embedding as emb

logger = logging.getLogger(__name__)


def transfer_entropy(x, y, dim=1, emb_lag=1, n_neighbors=4, normalize=True):
    """    Transfer entropy based on k-nearest neighbors algorithm.
    Computes TE(y-->x) = I(y(t), x(t-1:t-dim)| y(t-1:t-dim))
    that is the mutual information between x and the past of x, conditional
    to the past of y. If dim is not specified (None), it will be computed using
    Ragwitz criterion. Note that we do not include any lags in the embedding
    as for example in Lindner et al. BMC Neuroscience 2011.

    References:
    Gomez-Herreto et al. Entropy 2015. doi:10.3390/e17041958
    Gençağa, D. (2018). Transfer entropy. Entropy, 20(4), 288.
    Zhu, J.et al. (2015). Entropy 2015, 17, 4173-4201. https://doi.org/10.3390/e17064173

    Args:
        x (np.array): forced series
        y (np.array): forcing series
        dim (int, optional): Length of the past of x and y. Defaults to 1.
        emb_lag (int, optional): Embedding lag of the past of `x`and `y`. Defaults to 1.
        n_neighbors (int, optional): number of neighbors point to use in the
            k-nearest-neighbors algorithm. Defaults to 4.

    Returns:
        float: transfer entropy from y to x
    """
    if normalize:
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
    if not dim:
        dim = compute_ragwitz_criterion(x, n_neighbors=n_neighbors)
    x_emb = emb.time_delay_embedding(x, dim=dim + 1, lag=emb_lag)
    x_past = x_emb[:, :-1]
    x_future = x_emb[:, -1].reshape(-1, 1)
    y_past = emb.time_delay_embedding(y, dim=dim, lag=emb_lag)
    y_past = y_past[: x_future.size, :]  # crop out additional points it might have
    superspace = np.hstack((x_future, x_past, y_past))
    past_space = np.hstack((x_past, y_past))
    x_space = np.hstack((x_future, x_past))
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)
    nn.fit(superspace)
    radii, _ = nn.kneighbors()
    biggest_radius = radii[:, -1]
    radius = np.nextafter(biggest_radius, 0)

    kdt_xpast = KDTree(x_past, metric="chebyshev")
    n_xpast = kdt_xpast.query_radius(
        x_past, radius, count_only=True, return_distance=False
    )

    kdt_x = KDTree(x_space, metric="chebyshev")
    n_x = kdt_x.query_radius(x_space, radius, count_only=True, return_distance=False)

    kdt_past = KDTree(past_space, metric="chebyshev")
    n_past = kdt_past.query_radius(
        past_space, radius, count_only=True, return_distance=False
    )

    return digamma(n_neighbors) + np.mean(
        digamma(n_xpast) - digamma(n_past) - digamma(n_x)
    )


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
