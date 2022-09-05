import numpy as np
from surrogates import surrogates
from pytest import fixture
from scipy.fft import fft

random = np.random.RandomState(0)


@fixture
def x_series():
    x = np.zeros(1000)
    x[1] = 0.1
    x[2] = 1
    x[0] = -0.2
    for i in range(x.size - 3):
        x[i + 3] = (
            0.9 * x[i] - 0.15 * x[i + 1] + 0.2 * x[i + 2] + 0.05 * random.normal()
        )
    return x


def test_fft_surrogates(x_series):
    xx = surrogates.fourier_transform_surrogate(x_series, random_state=random)
    assert xx.size == x_series.size
    fft_x = fft(xx)
    assert (
        np.isclose(np.abs(fft_x), np.abs(fft(x_series))).sum() == fft_x.size - 1
    )  # highest freq is impossible to get


def test_iterative_fft_surrogates(x_series):
    surr = surrogates.iterative_fourier_transform_surrogate(
        x_series, random_state=random, rtol=1e-7
    )
    xx = surr["surr"]
    assert surr["iter"] < 1000
    assert surr["err"] < 1e-3
    assert xx.size == x_series.size
    fft_x = fft(xx)
    assert (
        np.isclose(np.abs(fft_x), np.abs(fft(x_series)), atol=0.25).sum() == fft_x.size
    )  # highest freq is impossible to get


def test_rank_array():
    x = np.array([3, 5, 7, 1, 6])
    rank = surrogates.rank_array(x)
    assert (rank == [1, 2, 4, 0, 3]).all()
