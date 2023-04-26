import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.special import digamma
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
        dim = emb.compute_ragwitz_criterion(x, n_neighbors=n_neighbors)
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
