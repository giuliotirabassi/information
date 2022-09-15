import numpy as np
from embedding import embedding as emb


def test_time_delay_embedding():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    embedded_x = emb.time_delay_embedding(x, 3, 2)
    assert embedded_x.shape == (3, 6)
    assert np.isclose(
        embedded_x,
        np.array([[1, 3, 5], [2, 4, 6], [3, 5, 7], [4, 6, 8], [5, 7, 9], [6, 8, 10]]),
    )
