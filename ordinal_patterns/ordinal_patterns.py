from itertools import permutations, product
import numpy as np
from scipy.stats import rankdata
from discrete_distributions.discrete_distributions import DiscreteDistribution


class OrdinalPattern(object):
    """Base class for ordinal pattern representation"""

    def __init__(self, data, order=3, step=1):
        self._step = step
        self._order = order
        self._alphabeth = self._compute_alphabeth()
        self._repr = self._compute_representation(data)

    def _compute_representation(self, data):
        raise NotImplementedError("This method is not implemented")

    def _compute_alphabeth(self):
        symbols = np.arange(1, self._order + 1).astype(int)
        alphabeth = []
        for perm in permutations(symbols):
            alphabeth.append("".join([str(x) for x in perm]))
        alphabeth = sorted(alphabeth)
        return tuple(alphabeth)

    def compute_symbol_distributinon(self):
        flatten_list = []
        for ll in self._repr:
            flatten_list.extend(ll)
        return DiscreteDistribution(flatten_list, self._alphabeth)


def _determine_ordinal_pattern(chunk):
    return "".join(rankdata(chunk, method="ordinal").astype(int).astype(str))


class TemporalOrdinalPattern(OrdinalPattern):
    """Temporal Ordinal Patterns representation of a timeseries.

    Parameters:
        timeseries (iterable): Timeseries to be converted in ordinal pattern
        order (int): Length of the ordinal patterns
        step (int): Distance between each elements of the ordinal pattern. `step = 1`
            means that the ordinal pattern will be formed using consecutive values.
    """

    def __init__(self, timeseries, order=3, step=1) -> None:
        if not isinstance(timeseries, np.ndarray):
            timeseries = np.array(timeseries)
        if len(timeseries.shape) > 1:
            raise ValueError("The input timeseries must be a vector")

        super().__init__(timeseries, order=order, step=step)

    def _compute_representation(self, timeseries):
        max_i_diff = (self._order - 1) * self._step + 1
        res = []
        for offset in range(max_i_diff - 1):
            rep = [
                _determine_ordinal_pattern(
                    timeseries[i : i + max_i_diff : self._step]  # noqa: E203
                )
                for i in range(offset, timeseries.size, max_i_diff - 1)
                if i + max_i_diff <= timeseries.size
            ]

            if rep:
                res.append(rep)
        return tuple([tuple(ll) for ll in res])

    def _compute_transitions(self):
        if self._order > 5:
            raise ValueError("Order should be less than 6!")
        transitions = []
        for ll in self._repr:
            for i, symb in enumerate(ll[:-1]):
                transitions.append("%s -> %s" % (symb, ll[i + 1]))
        return transitions

    def compute_transition_distribution(self):
        """Returns the `DiscreteDistribution`representing the
        transition between symbols"""
        transitions_alphabeth = [
            "%s -> %s" % p for p in product(self._alphabeth, self._alphabeth)
        ]
        return DiscreteDistribution(self._compute_transitions(), transitions_alphabeth)

    def flattened_representation(self):
        return [el for ll in self._repr for el in ll]


class SpatialOrdinalPattern(OrdinalPattern):
    """Spatial Ordinal Patterns representation of a spatial field. The input `data`
    must be a 2D array-like object. Ordinal patterns containing NaN will be ignored.
    The shape of the ordinal pattern is dictated by the parameter `order`. If order is
    an integer `n`, the ordinal patterns will be formed using squares of `n x n` points.
    If order is a tuple `(n, m)` the ordinal pattern will be formed using rectangles
    of `n x m` points, where `n` is the horizontal index and `m` the vertical one.

    Parameters
    ----------
    data : `iterable``
        Spatial field
    order : `int` or `tuple`
        Length of the ordinal patterns
    step : `int``
        Distance between each elements of the ordinal pattern. `step = 1` means
        that the ordinal pattern will be formed using consecutive values.
    """

    def __init__(self, data, order=3, step=1, mask=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if len(data.shape) != 2:
            raise ValueError("Input spatial field must be 2D")

        if mask is None:
            if isinstance(order, int):
                nh = nv = order
            else:
                nh, nv = order

        if isinstance(step, int):
            h_step = v_step = step
        else:
            h_step, v_step = step

        mask = np.zeros(((nv - 1) * v_step + 1, (nh - 1) * h_step + 1))
        for i in range(0, mask.shape[0], v_step):
            for j in range(0, mask.shape[1], h_step):
                mask[i, j] = 1

        self._mask = mask
        self._pattern_order = int(np.sum(mask))
        super().__init__(data, order=self._pattern_order, step=step)

    def _compute_representation(self, spatial_field):
        view = np.lib.stride_tricks.sliding_window_view(
            spatial_field, self._mask.shape
        ).reshape(-1, *self._mask.shape)
        idx = np.where(self._mask)
        res = []
        for v in view:
            res.append(_determine_ordinal_pattern(v[idx]))
        return (tuple(res),)


def compute_ordinal_patterns_representation(data, order, step):
    """Factory of ordinal patterns representations. The data is converted either
    in a `SpatialOrdinalPattern` (if the data is 2D) or a `TemporalOrdinalPattern`
    (if the data is 1D).

    Args:
        data (iterable): array-like data to be converted into ordinal patterns,
            can be 1D (temporal) or 2D (spatial)
        order (int, tuple): ordinal pattern dimension
        step (int): distance between points to be used to form the ordinal pattern

    Raises:
        ValueError: if the shape is not 1D or 2D

    Returns:
        OrdinalPattern: the ordinal pattenr representation of the data
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if len(data.shape) == 2 and 1 not in data.shape:
        return SpatialOrdinalPattern(data, order=order, step=step)
    elif len(data.shape) > 1:
        raise ValueError("Data can have 2 dimenstions at most")
    else:
        return TemporalOrdinalPattern(data.ravel(), order=order, step=step)
