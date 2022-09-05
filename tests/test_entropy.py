from entropy.entropy import BayesianEntropyCalculator, ClassicalEntropyCalculator
from collections import Counter
import pytest
import numpy as np


def test_bayesian_entropy_calculator():
    series = ["a"] * 400 + ["b"] * 400 + ["c"] * 400
    counts = Counter(series)
    with pytest.raises(AssertionError):
        bec = BayesianEntropyCalculator(counts, n_classes=2)
    bec = BayesianEntropyCalculator(counts)
    assert np.isclose(bec.entropy, -np.log(1 / 3), rtol=1e-3)
    assert (
        bec.entropy - 2 * bec.entropy_var ** 0.5
        < -np.log(1 / 3)
        < bec.entropy + 2 * bec.entropy_var ** 0.5
    )
    bec = BayesianEntropyCalculator({"a": 10000}, n_classes=2)
    assert bec.entropy - bec.entropy_var ** 0.2 < 0 < bec.entropy


def test_classical_entropy_calculator():
    series = ["a"] * 300 + ["b"] * 300 + ["c"] * 300
    counts = Counter(series)
    with pytest.raises(AssertionError):
        bec = ClassicalEntropyCalculator(counts, n_classes=2)
    bec = ClassicalEntropyCalculator(counts)
    assert np.isclose(bec.entropy, -np.log(1 / 3), rtol=1e-7)
    bec = ClassicalEntropyCalculator({"a": 10000}, n_classes=20)
    assert np.isclose(bec.entropy, 0, rtol=1e-3)
