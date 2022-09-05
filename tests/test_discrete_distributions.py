from discrete_distributions import discrete_distributions


def test_discrete_distribution():
    dd = discrete_distributions.DiscreteDistribution(["a", "a", "b", "b"], ["a", "b"])
    assert dd._counts["a"] == dd._counts["b"] == 2
    assert dd._probs["a"] == dd._probs["b"] == 0.5
    assert dd.compute_entropy(normalize=True) == 1
    assert dd.compute_entropy(normalize=True, bayes=True)
    assert dd.compute_js_div() == 0
    assert dd.compute_kl_div() == 0
