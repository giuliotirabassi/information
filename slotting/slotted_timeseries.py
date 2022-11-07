import numpy as np


class GaussianInterpolator(object):
    def __init__(self, kernel_width):
        self.width = kernel_width

    def interpolate(self, t, y, t_i):
        """Gaussian interpolation of the series (t, y) at times t_i with
        kernel width `width`"""
        dt = t_i[:, np.newaxis] - t
        w = np.exp(-(dt ** 2) / (2 * self.width ** 2))
        y_int = (w * y).sum(axis=1) / w.sum(axis=1)
        return y_int
