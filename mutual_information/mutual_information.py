import logging
from itertools import product
import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.special import digamma
from collections import Counter
from entropy.entropy import BayesianEntropyCalculator
from typing import Iterable

logger = logging.getLogger(__name__)


def mutual_information_continuous(*args: Iterable[Iterable], n_neighbors=4) -> float:
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
    args = [x.reshape((-1, 1)) for x in args]
    n_vars = len(args)
    xx = np.hstack(args)
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
    series_a: Iterable, series_b: Iterable, alphabeth_a=None, alphabeth_b=None
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
    h_a = BayesianEntropyCalculator(counts_a).entropy
    h_b = BayesianEntropyCalculator(counts_b).entropy
    h_ab = BayesianEntropyCalculator(counts_ab).entropy
    return h_a + h_b - h_ab


def conditional_mutual_information_continuous(
    x: Iterable, y: Iterable, z: Iterable, **kwargs
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

    Returns:
        float: conditional mutual information I(x, y | z)
    """
    return mutual_information_continuous(
        x, y, **kwargs
    ) - mutual_information_continuous(x, y, z, **kwargs)
