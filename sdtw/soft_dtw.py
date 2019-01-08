# Author: Mathieu Blondel
# License: Simplified BSD

import numpy as np

from .soft_dtw_fast import _soft_dtw
from .soft_dtw_fast import _soft_dtw_grad


class SoftDTW(object):

    def __init__(self, D, gamma=1.0, sakoe_chiba_band=-1):
        """
        Parameters
        ----------
        D: array, shape = [m, n] or distance object
            Distance matrix between elements of two time series.

        gamma: float
            Regularization parameter.
            Lower is less smoothed (closer to true DTW).

        sakoe_chiba_band: int
            If non-negative, the DTW is restricted to a Sakoe-Chiba band around
            the diagonal. The band has a width of 2 * sakoe_chiba_band + 1.

        Attributes
        ----------
        self.R_: array, shape = [m + 2, n + 2]
            Accumulated cost matrix (stored after calling `compute`).
        """
        if hasattr(D, "compute"):
            self.D = D.compute()
        else:
            self.D = D

        self.D = self.D.astype(np.float64)

        self.gamma = gamma
        self.sakoe_chiba_band = sakoe_chiba_band

    def compute(self):
        """
        Compute soft-DTW by dynamic programming.

        Returns
        -------
        sdtw: float
            soft-DTW discrepancy.
        """
        m, n = self.D.shape

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the backward recursion.
        self.R_ = np.zeros((m + 2, n + 2), dtype=np.float64)

        _soft_dtw(self.D, self.R_, gamma=self.gamma,
                  sakoe_chiba_band=self.sakoe_chiba_band)

        return self.R_[m, n]

    def grad(self):
        """
        Compute gradient of soft-DTW w.r.t. D by dynamic programming.

        Returns
        -------
        grad: array, shape = [m, n]
            Gradient w.r.t. D.
        """
        if not hasattr(self, "R_"):
            raise ValueError("Needs to call compute() first.")

        m, n = self.D.shape

        # Add an extra row and an extra column to D.
        # Needed to deal with edge cases in the recursion.
        D = np.vstack((self.D, np.zeros(n)))
        D = np.hstack((D, np.zeros((m + 1, 1))))

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the recursion.
        E = np.zeros((m + 2, n + 2))

        _soft_dtw_grad(D, self.R_, E, gamma=self.gamma,
                       sakoe_chiba_band=self.sakoe_chiba_band)

        return E[1:-1, 1:-1]
