import collections
import numpy as np
from scipy.special import gammaln
from entropy.entropy import BayesianEntropyCalculator, ClassicalEntropyCalculator


class DiscreteDistribution(object):
    def __init__(self, data, alphabeth):
        self.alphabeth = alphabeth
        self._n_classes = len(alphabeth)

        counter = collections.Counter(data)
        total = sum(counter.values())
        self._probs = {}
        self._counts = {}
        for k in alphabeth:
            self._counts[k] = counter.get(k, 0)
            self._probs[k] = self._counts[k] / total

        assert np.isclose(sum(self._probs.values()), 1)

    def compute_entropy(self, bayes=False, normalize=False):
        norm = np.log(self._n_classes) if normalize else 1
        if bayes:
            return self._compute_bayesian_entropy() / norm
        else:
            return self._compute_biased_entropy() / norm

    def _compute_bayesian_entropy(self):
        return BayesianEntropyCalculator(
            self._counts, n_classes=self._n_classes
        ).entropy

    def _compute_bayesian_entropy_var(self):
        return BayesianEntropyCalculator(
            self._counts, n_classes=self._n_classes
        ).entropy_var

    def _compute_biased_entropy(self):
        return ClassicalEntropyCalculator(
            self._counts, n_classes=self._n_classes
        ).entropy

    def _compute_probability_uncertainty(self):
        "according to Samengo 2002"
        counter = self._counts
        total = sum(counter.values())
        moment = 2  # second moment
        beta = 1  # uniform prior
        s = self._n_classes  # n classes
        uncertainty = {}
        for k, v in counter.items():
            uncertainty[k] = np.exp(
                gammaln(v + moment + beta)
                + gammaln(total + s * beta)
                - gammaln(v + beta)
                - gammaln(total + s * beta + moment)
            )
        return uncertainty

    def compute_kl_div(self):
        "compute K-L divergence with Uniform distribution"
        e = 1 / self._n_classes
        return sum(p * (np.log(p) - np.log(e)) for p in self._probs.values() if p)

    def compute_js_div(self):
        """Compute Jansen-Shannon divergence with Uniform distribution"""
        e = 1 / self._n_classes
        m = {k: (v + e) / 2 for k, v in self._probs.items()}
        kl_pm = sum(p * (np.log(p) - np.log(m[k])) for k, p in self._probs.items() if p)
        kl_me = sum(p * (np.log(p) - np.log(e)) for p in m.values())
        return (kl_pm + kl_me) / 2
