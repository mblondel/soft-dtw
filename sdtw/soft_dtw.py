# Author: Mathieu Blondel
# License: Simplified BSD

import numpy as np

from .soft_dtw_fast import _soft_dtw


def soft_dtw_from_dist(D, gamma=1.0):
    """
    Compute soft-DTW by dynamic programming,
    from a distance matrix.

    Parameters
    ----------
    D: array, shape = [m, n]
        Distance matrix between elements of both time series.

    gamma: float
        Regularization parameter (lower is less smoothed, i.e.,
        closer to true DTW).

    Returns
    -------
    sdtw: float
        soft-DTW discrepancy.
    """
    m, n = D.shape
    # We need +1 because of indexing starting from 1.
    R = np.zeros((m+1, n+1))
    _soft_dtw(D, R, gamma=gamma)
    return R[m, n]
