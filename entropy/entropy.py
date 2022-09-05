import numpy as np
from scipy.special import digamma, polygamma
from scipy.integrate import quad


class EntropyCalculator(object):
    def __init__(self, counts, n_classes=None):
        if n_classes is None:
            n_classes = len(counts)
        assert n_classes >= len(counts)
        self._n_classes = n_classes
        self._log_n_classes = np.log(self._n_classes)
        self._counts = counts
        self._counts_values = np.array([v for v in self._counts.values()])
        self._total = sum(counts.values())
        self._entropy = None
        self._entropy_var = None


class BayesianEntropyCalculator(EntropyCalculator):
    """
    NSB estimator for entropy of a discrete distribution.

    counts contains a map symbol: n_of_occurrences
    n_classes is an int representing the number of possyble symbols

    From Nemenman et al.2002
    """

    def _exp_entropy(self, beta):
        pseudototal = self._total + beta * self._n_classes
        pseudocounts = self._counts_values + beta
        return (
            digamma(pseudototal + 1)
            - (pseudocounts * digamma(pseudocounts + 1)).sum() / pseudototal
        )

    def _auxiliary_ii(self, nk, ni, pseudototal):
        return (digamma(nk + 1) - digamma(pseudototal + 2)) * (
            digamma(ni + 1) - digamma(pseudototal + 2)
        ) - polygamma(1, pseudototal + 2)

    def _auxiliary_j(self, ni, pseudototal):
        return (
            (digamma(ni + 2) - digamma(pseudototal + 2)) ** 2
            + polygamma(1, ni + 2)
            - polygamma(1, pseudototal + 2)
        )

    def _exp_sq_entropy(self, beta):
        pseudototal = self._total + beta * self._n_classes
        pseudocounts = {k: v + beta for k, v in self._counts.items()}
        res = 0
        for i, ni in pseudocounts.items():
            for k, nk in pseudocounts.items():
                if i == k:
                    res += (ni + 1) * ni * self._auxiliary_j(ni, pseudototal)
                else:
                    res += ni * nk * self._auxiliary_ii(ni, nk, pseudototal)
        return res / (pseudototal * (pseudototal + 1))

    def _hyperprior(self, beta):
        return (
            self._n_classes * polygamma(1, self._n_classes * beta + 1)
            - polygamma(1, beta + 1)
        ) / self._log_n_classes

    def _kernel_entropy(self, beta):
        return self._hyperprior(beta) * self._exp_entropy(beta)

    def _kernel_sq_entropy(self, beta):
        return self._hyperprior(beta) * self._exp_sq_entropy(beta)

    @property
    def entropy(self):
        if self._entropy is None:
            self._entropy = quad(self._kernel_entropy, 0, np.inf)[0]
        return self._entropy

    @property
    def entropy_var(self):
        if self._entropy_var is None:
            sq_entropy = quad(self._kernel_sq_entropy, 0, np.inf)[0]
            self._entropy_var = sq_entropy - self.entropy ** 2
        return self._entropy_var


class ClassicalEntropyCalculator(EntropyCalculator):
    @property
    def entropy(self):
        ps = [c / self._total for c in self._counts.values()]
        return -sum(p * np.log(p) for p in ps)
