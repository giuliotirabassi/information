from causality import cross_mapping
import numpy as np


def test_cross_mapping():
    x = np.arange(1, 10)
    y = x + 11
    assert cross_mapping.convergent_cross_mapping(x, y, 2, 2)


def test_convergent_cross_mapping():
    # https://phdinds-aim.github.io/time_series_handbook/06_ConvergentCrossMappingandSugiharaCausality/ccm_sugihara.html
    def func_1(A, B, r, beta):
        return A * (r - r * A - beta * B)

    # Initialize test dataset
    # params
    r_x = 3.7
    r_y = 3.7
    B_xy = 0  # effect on x given y (effect of y on x)
    B_yx = 0.32  # effect on y given x (effect of x on y)

    X0 = 0.2  # initial val following Sugihara et al
    Y0 = 0.4  # initial val following Sugihara et al
    t = 3000  # time steps

    X = [X0]
    Y = [Y0]
    for i in range(t):
        X_ = func_1(X[-1], Y[-1], r_x, B_xy)
        Y_ = func_1(Y[-1], X[-1], r_y, B_yx)
        X.append(X_)
        Y.append(Y_)

    r, p = cross_mapping.convergent_cross_mapping(X, Y, 2, 1)
    print(r, p)

    r2, p2 = cross_mapping.convergent_cross_mapping(Y, X, 2, 1)
    print(r2, p2)

    assert r2 > r
    assert p2 < 0.1
