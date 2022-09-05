from ordinal_patterns import ordinal_patterns
from discrete_distributions import discrete_distributions
import numpy as np
import pytest


def test_compute_ordinal_patterns():
    x = np.array([1, 2, 3, 4, 5, 6])
    op = ordinal_patterns.compute_ordinal_patterns_representation(x, order=2, step=1)
    assert isinstance(op, ordinal_patterns.TemporalOrdinalPattern)

    x = np.array([[1, 2, 3], [4, 5, 6]])
    op = ordinal_patterns.compute_ordinal_patterns_representation(x, order=2, step=1)
    assert isinstance(op, ordinal_patterns.SpatialOrdinalPattern)

    x = np.array([[[1, 2], [5, 6]], [[1, 2], [5, 6]]])
    with pytest.raises(ValueError):
        op = ordinal_patterns.compute_ordinal_patterns_representation(
            x, order=2, step=1
        )


def test_temporal_ordinal_patterns():
    x = np.array([1, 2, 3, 4, 5, 4])
    op = ordinal_patterns.compute_ordinal_patterns_representation(x, order=2, step=1)
    assert op._repr == (("12", "12", "12", "12", "21"),)
    assert tuple(op._compute_transitions()) == (
        "12 -> 12",
        "12 -> 12",
        "12 -> 12",
        "12 -> 21",
    )
    td = op.compute_transition_distribution()
    assert isinstance(td, discrete_distributions.DiscreteDistribution,)
    assert td._counts["12 -> 12"] == 3
    assert td._counts["21 -> 21"] == 0
    assert td._counts["12 -> 21"] == 1
    assert td._counts["21 -> 12"] == 0
    op = ordinal_patterns.compute_ordinal_patterns_representation(x, order=2, step=2)
    assert op._repr == (("12", "12"), ("12", "12"))
    assert op.flattened_representation() == ["12", "12", "12", "12"]
    op = ordinal_patterns.compute_ordinal_patterns_representation(x, order=3, step=1)
    assert op._repr == (("123", "123"), ("123", "132"))
    op = ordinal_patterns.compute_ordinal_patterns_representation(x, order=3, step=2)
    print(op._repr)
    assert op._repr == (("123",), ("123",))


def test_spatial_ordinal_pattern():
    x = np.array([[1, 2, 3, 4], [2.1, 5, 3.1, 4.1]])
    op = ordinal_patterns.compute_ordinal_patterns_representation(x, order=2, step=1)
    assert len(op._repr) == 1
    assert sorted(op._repr[0]) == sorted(("1234", "1324", "1243"))
    op = ordinal_patterns.compute_ordinal_patterns_representation(
        x, order=(3, 2), step=1
    )
    assert len(op._repr) == 1
    assert sorted(op._repr[0]) == sorted(("124365", "124635"))
    x = np.array([[1, 2, 3, 4], [21, 5, 30, 48], [9, 19, 38, 40], [101, 120, 103, 104]])
    op = ordinal_patterns.compute_ordinal_patterns_representation(x, order=2, step=2)
    assert sorted(op._repr[0]) == sorted(("1234", "1234", "1234", "1243"))
    op = ordinal_patterns.compute_ordinal_patterns_representation(x, order=2, step=3)
    assert sorted(op._repr[0]) == sorted(("1234",))
