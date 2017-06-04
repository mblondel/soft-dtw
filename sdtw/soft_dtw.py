# Author: Mathieu Blondel
# License: Simplified BSD

import numpy as np

from .soft_dtw_fast import _soft_dtw
from .soft_dtw_fast import _soft_dtw_grad


class soft_DTW(object):

    def __init__(self, gamma=1.0):
        """
        Parameters
        ----------

        gamma: float
            Regularization parameter.
            Lower is less smoothed (closer to true DTW).

        Attributes
        ----------
        self.R_: array, shape = [m + 2, n + 2]
            Accumulated cost matrix (stored after calling `compute`).
        """
        self.gamma = gamma

    def compute(self, D):
        """
        Compute soft-DTW by dynamic programming.

        Parameters
        ----------
        D: array, shape = [m, n]
            Distance matrix between elements of both time series.

        Returns
        -------
        sdtw: float
            soft-DTW discrepancy.
        """
        m, n = D.shape

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the backward recursion.
        self.R_ = np.zeros((m+2, n+2))

        _soft_dtw(D, self.R_, gamma=self.gamma)

        return self.R_[m, n]

    def grad(self, D):
        """
        Compute gradient of soft-DTW w.r.t. D by dynamic programming.

        Parameters
        ----------
        D: array, shape = [m, n]
            Distance matrix between elements of both time series.

        Returns
        -------
        grad: array, shape = [m, n]
            Gradient w.r.t. D.
        """
        m, n = D.shape

        # Add an extra row and an extra column to D.
        # Needed to deal with edge cases in the recursion.
        D = np.vstack((D, np.zeros(n)))
        D = np.hstack((D, np.zeros((m+1, 1))))

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the recursion.
        E = np.zeros((m+2, n+2))

        _soft_dtw_grad(D, self.R_, E, gamma=self.gamma)

        return E[1:-1, 1:-1]
