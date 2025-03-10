import logging
from itertools import product
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.special import digamma
from collections import Counter
from entropy.entropy import BayesianEntropyCalculator
from typing import Iterable

logger = logging.getLogger(__name__)


def mutual_information_continuous(x, y, normalize=False, **kwargs):
    if normalize:
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
    return _mutual_information_continuous(x, y, **kwargs)


def _mutual_information_continuous(*args: Iterable[Iterable], n_neighbors=4) -> float:
    """Mutual information of the args computed using k-nearest-neighbors
    algorithm with k = n_neighbors.

        I(args[0], args[1], ..., args[n-1])

    time series should be normalized to zero mean and unitary variance
    or even better gaussianized with a rank transformation.

    References:
        Kraskov et al. Phys. Rev. E 2004, 69

    Args:
        n_neighbors (int, optional): number of neighbors point to use in
        the k-nearest-neighbors algorithm. Defaults to 4.

    Returns:
        float: mutual information between the arguments
    """
    args = [x.reshape((-1, 1)) if len(x.shape) == 1 else x for x in args]
    n_vars = len(args)
    try:
        xx = np.hstack(args)
    except ValueError:
        raise ValueError("Input arguments' dimensions do not match. Try transposing.")
    n_samples = xx.shape[0]
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)
    nn.fit(xx)
    radii, _ = nn.kneighbors()
    biggest_radius = radii[:, -1]
    radius = np.nextafter(biggest_radius, 0)
    ns = []
    for xi in args:
        kd = KDTree(xi, metric="chebyshev")
        nx = kd.query_radius(xi, radius, count_only=True, return_distance=False)
        ns.append(
            nx
        )  # here would go a -1 but we have to sum 1 in the final formula so omit it
    digammas = [np.mean(digamma(nx)) for nx in ns]
    return digamma(n_neighbors) + (n_vars - 1) * digamma(n_samples) - np.sum(digammas)


def mutual_information_discrete(
    series_a: Iterable,
    series_b: Iterable,
    alphabeth_a=None,
    alphabeth_b=None,
    return_variance=False,
) -> float:
    """Calcolation of mutual information between two symbolic time series. According to
    Archer at al. 2013 the best estimator seems to be the sum of the bayesian estimators
    of the entropies. For each series an alphabeth can be supplied. If the alphabeth
    is not supplied it will be inferred by the series.

    References:
        Archer at al. Entropy 2013

    Args:
        series_a (iterable): First series
        series_b (iterable): Second series
        alphabeth_a (set, optional): Alphabeth of first series. Defaults to None.
        alphabet_b (set, optional): Alphabet of second series. Defaults to None.
        return_variance (bool, optional): whether to return the variance of the
            estimator. Defaults to False.

    Returns:
        float: MI between `series_a` and `series_b`
    """
    if not alphabeth_a:
        alphabeth_a = set(series_a)
    if not alphabeth_b:
        alphabeth_b = set(series_b)
    for alphabeth in [alphabeth_a, alphabeth_b]:
        if not isinstance(alphabeth, set):
            alphabeth = set(alphabeth)
    joint_alphabeth = product(alphabeth_a, alphabeth_b)
    counts_a = Counter(series_a)
    for symb in alphabeth_a:
        if symb not in counts_a:
            counts_a[symb] = 0
    counts_b = Counter(series_b)
    for symb in alphabeth_b:
        if symb not in counts_b:
            counts_b[symb] = 0
    counts_ab = Counter(zip(series_a, series_b))
    for symb in joint_alphabeth:
        if symb not in counts_ab:
            counts_a[symb] = 0
    bec_a = BayesianEntropyCalculator(counts_a)
    h_a = bec_a.entropy
    bec_b = BayesianEntropyCalculator(counts_b)
    h_b = bec_b.entropy
    bec_ab = BayesianEntropyCalculator(counts_ab)
    h_ab = bec_ab.entropy
    mi = h_a + h_b - h_ab
    if return_variance:
        var_a = bec_a.entropy_var
        var_b = bec_b.entropy_var
        var_ab = bec_ab.entropy_var
        return mi, var_a + var_b + var_ab
    return mi


def conditional_mutual_information_continuous(
    x: Iterable, y: Iterable, z: Iterable, n_neighbors=4
) -> float:
    """Returns conditional mutal information between x and y given z as possible
    confounding factor.
    If z influences both y and x, it will be less than the mutual information
    between x and y.
    If z is synergetic, it will be higher than the mutual information between x and y

    Args:
        x (Iterable): first series of values from p(x)
        y (Iterable): second series of values from p(y)
        z (Iterable): confounding factor values from p(z)
        n_neighbors (int): default 4

    Returns:
        float: conditional mutual information I(x, y | z)

    Reference:
        Frenzel and Pompe, (2007) Phys. Rev. Lett.
    """
    xyz = np.stack((x, y, z)).T
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)
    nn.fit(xyz)
    radii, _ = nn.kneighbors()
    radius = np.nextafter(radii[:, -1], 0)  # to ensure we only count less than
    xz = np.stack((x, z)).T
    yz = np.stack((y, z)).T
    kd = KDTree(xz, metric="chebyshev")
    nxz = (
        kd.query_radius(xz, radius, count_only=True, return_distance=False) - 1
    )  # minus one to not count dist=0
    kd = KDTree(yz, metric="chebyshev")
    nyz = kd.query_radius(yz, radius, count_only=True, return_distance=False) - 1
    zz = z.reshape((-1, 1))
    kd = KDTree(zz, metric="chebyshev")
    nz = kd.query_radius(zz, radius, count_only=True, return_distance=False) - 1
    maxn = max(max(nz), max(nyz), max(nxz))
    hn = -np.cumsum(1 / np.arange(1, maxn + 1))
    return np.mean(hn[nyz - 1] + hn[nxz - 1] - hn[nz - 1]) - hn[n_neighbors - 2]
