from causality import transfer_entropy, pseudo_transfer_entropy, granger_causality
from embedding import embedding
import numpy as np

random = np.random.RandomState(0)


def test_transfer_entropy():
    te1s = []
    te2s = []
    for i in range(100):
        L = 1000
        x = np.zeros(L)
        y = np.zeros(L)
        for i in range(x.size - 1):
            x[i + 1] = -x[i] + 0.1 * random.normal()
            y[i + 1] = 0.9 * y[i] - 0.8 * x[i] + 0.1 * random.normal()
        te1 = transfer_entropy.transfer_entropy(x, y, dim=None)
        te2 = transfer_entropy.transfer_entropy(y, x, dim=None)
        te1s.append(te1)
        te2s.append(te2)
    m1 = np.mean(te1s)
    m2 = np.mean(te2s)
    s1 = np.std(te1s)
    s2 = np.std(te2s)
    assert m1 + s1 < m2 - s2


def test_autocorrelation_criterion():
    t = np.linspace(0, 20 * np.pi, 10000)
    x = np.sin(t)
    lag = embedding.compute_autocorrelation_criterion(x)
    assert np.isclose(t[lag], np.pi / 2, rtol=0.05)
    lag = embedding.compute_autocorrelation_criterion(x, criterion="min")
    assert np.isclose(t[lag], np.pi, rtol=0.05)


def test_pseudo_transfer_entropy():
    te1s = []
    te2s = []
    for i in range(100):
        L = 1000
        x = np.zeros(L)
        y = np.zeros(L)
        for i in range(x.size - 1):
            x[i + 1] = -x[i] + 0.1 * random.normal()
            y[i + 1] = 0.9 * y[i] - 0.8 * x[i] + 0.1 * random.normal()
        te1 = pseudo_transfer_entropy.pseudo_transfer_entropy(x, y, dim=1)
        te2 = pseudo_transfer_entropy.pseudo_transfer_entropy(y, x, dim=1)
        te1s.append(te1)
        te2s.append(te2)
    m1 = np.mean(te1s)
    m2 = np.mean(te2s)
    s1 = np.std(te1s)
    s2 = np.std(te2s)
    assert m1 + s1 < m2 - s2


def test_embed_series_pseudo_transfer():
    x = np.arange(10)
    y = np.arange(20, 30)
    xx, X, Y = pseudo_transfer_entropy._embed_series(x, y, dim=2, emb_lag=3, tau=4)
    assert np.isclose(X.T, [[0, 3], [1, 4], [2, 5]]).all()
    assert np.isclose(Y.T, [[20, 23], [21, 24], [22, 25]]).all()
    assert np.isclose(xx, [7, 8, 9]).all()


def test_granger_causality():
    te1s = []
    te2s = []
    pe1s = []
    pe2s = []
    for i in range(100):
        L = 1000
        x = np.zeros(L)
        y = np.zeros(L)
        for i in range(x.size - 1):
            x[i + 1] = -x[i] + 0.1 * random.normal()
            y[i + 1] = 0.9 * y[i] - 0.8 * x[i] + 0.1 * random.normal()
        te1, pe1 = granger_causality.granger_causality(x, y, dim=1)
        te2, pe2 = granger_causality.granger_causality(y, x, dim=1)
        te1s.append(te1)
        te2s.append(te2)
        pe1s.append(pe1)
        pe2s.append(pe2)
    m1 = np.mean(te1s)
    m2 = np.mean(te2s)
    s1 = np.std(te1s)
    s2 = np.std(te2s)
    assert m1 + s1 < m2 - s2
    assert np.mean(pe2s) + np.std(pe2s) < np.mean(pe1s) - np.std(pe1s)


def test_schwarts_criterion():
    L = 1000
    x = np.zeros(L)
    x[1] = 0.1
    x[2] = 1
    x[0] = -0.2
    for i in range(x.size - 3):
        x[i + 3] = (
            0.9 * x[i] - 0.15 * x[i + 1] + 0.2 * x[i + 2] + 0.05 * random.normal()
        )
    dim = granger_causality.schwartz_criterion(x[6:])
    assert dim == 3
