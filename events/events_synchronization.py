import numpy as np
from scipy.optimize import linear_sum_assignment


def events_synchronization(events_i, events_j, tau=None):
    """Event synchronizarion according to Quiroga et al. (2002) PRE.
    Events are syncronized if happening with a time window `tau`. If `tau` is not
    specified it is determined dynamically as specified in the original paper, that
    is the minimum local interevent distance. If `tau` is specified as a number,
    then the algorithm will match the highest amount of events within a window tau
    avoiding repetitions, that is pairing one event to more than one event in
    the other series. The resulting output is in the range [0, 1].

    Args:
        events_i (list of times marking events A): Iterable[Number]
        events_j (list of times marking events B): Iterbale[Number]
        tau (float, optional): Max lag between synchronized events. Defaults to None.

    Raises:
        ValueError: In case of repeated events within a timeseries.

    Returns:
        float: The symmetric combination quantifying event synchronization
    """
    # SANITY CHECKS
    if len(events_i) > len(set(events_i)) or len(events_j) > len(set(events_j)):
        raise ValueError("Repeated events are not allowed")
    if len(events_i) == 0 or len(events_j) == 0:
        return 0  # there are no events to match
    if len(events_i) == 1 and len(events_j) == 1 and tau is None:
        return 0  # we cannot determine tau

    if tau is None:  # use straight the paper, check also Scholarpedia
        cij = _c_coef(events_i, events_j)
        cji = _c_coef(events_j, events_i)
        return (cij + cji) / np.sqrt(len(events_i) * len(events_j))

    else:  # only count the closest events if they are below or equal tau
        deltat = np.subtract.outer(events_i, events_j)
        deltat = np.abs(deltat).astype(float)

        deltat[deltat > tau] = 1e15
        rows, cols = linear_sum_assignment(deltat)
        Jij = np.full(deltat.shape, np.inf)
        Jij[rows, cols] = deltat[rows, cols]
        Jij = Jij <= tau

        # pairings = _max_keep_smallest(deltat, tau)
        # Jij = ~np.isnan(pairings)
        return Jij.sum() / np.sqrt(len(events_i) * len(events_j))


def _c_coef(events_i, events_j):
    deltat = np.subtract.outer(events_i, events_j)
    tauij = np.zeros((len(events_i), len(events_j)))
    for i in range(tauij.shape[0]):
        for j in range(tauij.shape[1]):
            tauij[i, j] = (
                np.min(
                    [
                        *np.diff(events_i[max(0, i - 1) : i + 2]),
                        *np.diff(events_j[max(0, j - 1) : j + 2]),
                    ]
                )
                / 2
            )
    Jij = ((0 < deltat) & (deltat <= tauij)).astype(float) + 0.5 * (deltat == 0)
    return Jij.sum()


def _bpm(row, adj, match_to_col, visited, visit_token):
    for col in adj[row]:
        if visited[col] != visit_token:
            visited[col] = visit_token
            if match_to_col[col] == -1 or _bpm(
                match_to_col[col], adj, match_to_col, visited, visit_token
            ):
                match_to_col[col] = row
                return True
    return False


def _max_keep_smallest(matrix, threshold):
    """Keep all elements of a matrix smaller than `threshold`
    keeping at most one element per row/column and keeping as many elements
    possible using  maximum bipartite matching algorithm (Kuhn’s algorithm or
    Hungarian algorithm)."""
    n, m = matrix.shape

    # Step 1: Build adjacency list (row → list of columns)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if matrix[i][j] <= threshold:
                adj[i].append(j)

    # Step 2: Initialize matching arrays
    match_to_col = [-1] * m
    visited = [0] * m
    visit_token = 1

    # Step 3: Attempt to match each row
    for row in range(n):
        _bpm(row, adj, match_to_col, visited, visit_token)
        visit_token += 1

    # Step 4: Build output matrix
    output = np.full((n, m), float("NaN"))
    for col, row in enumerate(match_to_col):
        if row != -1:
            output[row][col] = matrix[row][col]

    return output
