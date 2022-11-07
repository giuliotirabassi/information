from slotting.slotted_timeseries import GaussianInterpolator
import numpy as np

random = np.random.RandomState(0)


def test_gaussian_interpolator():
    t = random.rand(10)
    x = random.rand(10)
    gi = GaussianInterpolator(0.1)
    xx = gi.interpolate(t, x, np.linspace(0, 1, 20))
    assert xx.size == 20
    assert x.min() <= xx.min() <= xx.max() <= x.max()
