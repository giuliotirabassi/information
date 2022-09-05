import numpy as np
from scipy.stats import pearsonr, multivariate_normal
from mutual_information.mutual_information import (
    mutual_information_continuous,
    mutual_information_discrete,
)

random = np.random.RandomState(0)


def test_mutual_information():
    s = multivariate_normal(mean=[0, 0], cov=[[1, 0.9], [0.9, 1]], seed=42).rvs(
        size=100000
    )
    a = s[:, 0]
    b = s[:, 1]
    theor = -0.5 * np.log(1 - pearsonr(a, b)[0] ** 2)
    assert (mutual_information_continuous(a, b) - theor) / theor < 0.001


def test_mutual_information_discrete():
    a = random.choice([0, 1, 2], size=2000000)
    b = random.choice([0, 1, 2], size=2000000)
    mi = mutual_information_discrete(a, b)
    assert -1e-5 < mi < 1e-5
    assert mutual_information_discrete(a, a) > 0
