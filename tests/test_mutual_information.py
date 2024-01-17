import numpy as np
from scipy.stats import pearsonr, multivariate_normal
from mutual_information.mutual_information import (
    mutual_information_continuous,
    mutual_information_discrete,
    conditional_mutual_information_continuous,
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


def test_conditional_mutual_information():
    # from Frenzel & Pompe 2007 PRL (kinda, with 3 instead of 6)
    cov = np.zeros((3, 3)) + 0.9
    for i in range(3):
        cov[i, i] = 1

    def h(c):
        d, _ = c.shape
        return 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.linalg.det(c))

    cmitheo = 2 * h(cov[0:2, 0:2]) - h(cov[0:1, 0:1]) - h(cov)

    mvn = multivariate_normal(mean=[0, 0, 0], cov=cov, seed=42)

    cmis = []
    for i in range(100):
        s = mvn.rvs(size=1000)
        a = s[:, 0]
        b = s[:, 1]
        c = s[:, 2]
        cmi = conditional_mutual_information_continuous(a, b, c, n_neighbors=4)
        cmis.append(cmi)

    assert np.abs(np.mean(cmis) - cmitheo) / cmitheo < 0.05
