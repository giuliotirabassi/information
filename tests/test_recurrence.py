from recurrence import recurrence
import numpy as np


def test_recurrence_matrix():
    timeseries = np.array([0, 1, 3, 5, 0.1, -3, 6, 2])
    rm = recurrence.compute_recurrence_matrix(timeseries, 1)
    assert (
        np.array(
            [
                [0, 1, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
            ]
        )
        == rm
    ).all()
    rm = recurrence.compute_recurrence_matrix(timeseries, 0.099)
    assert (rm == np.zeros((8, 8))).all()
