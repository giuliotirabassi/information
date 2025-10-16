from sklearn.neighbors import NearestNeighbors
import numpy as np


def compute_recurrence_matrix(timeseries, epsilon):
    """Recrrence plot of a timeseries, possibly multidimensional
    if the time series is multidimensional, then the input
    is an array where rows represent time steps and columns variables.
    The output is an n_rows X n_rows boolean matrix of False (non recurrence)
    and True (recurrence)"""
    if len(timeseries.shape) == 1:
        timeseries = timeseries.reshape((-1, 1))
    nt, _ = timeseries.shape
    nn = NearestNeighbors(metric="chebyshev", radius=epsilon)
    nn.fit(timeseries)
    nbrs = nn.radius_neighbors(return_distance=False)
    recurrence = np.full(shape=(nt, nt), dtype=bool, fill_value=False)
    for i, neighs in enumerate(nbrs):
        recurrence[i, neighs] = True
    return recurrence
