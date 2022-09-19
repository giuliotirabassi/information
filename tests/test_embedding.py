import numpy as np
from embedding import embedding as emb


def test_time_delay_embedding():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    embedded_x = emb.time_delay_embedding(x, 3, 2)
    assert embedded_x.shape == (6, 3)
    assert np.isclose(
        embedded_x,
        np.array([[1, 3, 5], [2, 4, 6], [3, 5, 7], [4, 6, 8], [5, 7, 9], [6, 8, 10]]),
    ).all()
    embedded_x = emb.time_delay_embedding(x, 3, 3)
    assert embedded_x.shape == (4, 3)
    assert np.isclose(
        embedded_x, np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9], [4, 7, 10]]),
    ).all()
    embedded_x = emb.time_delay_embedding(x, 1, 3)
    assert embedded_x.shape == (x.size, 1)
    assert np.isclose(embedded_x, embedded_x).all()
    embedded_x = emb.time_delay_embedding(x, 2, 1)
    assert embedded_x.shape == (x.size - 1, 2)
    assert np.isclose(
        embedded_x,
        np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]
        ),
    ).all()
