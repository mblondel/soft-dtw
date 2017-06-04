# Author: Mathieu Blondel
# License: Simplified BSD

import numpy as np

from .soft_dtw_fast import _soft_dtw
from .soft_dtw_fast import _soft_dtw_grad


def soft_dtw_from_dist(D, ret_R=False, gamma=1.0):
    """
    Compute soft-DTW by dynamic programming,
    from a distance matrix.

    Parameters
    ----------
    D: array, shape = [m, n]
        Distance matrix between elements of both time series.

    ret_R: bool
        Whether to return R (accumulated cost matrix) or not.

    gamma: float
        Regularization parameter (lower is less smoothed, i.e.,
        closer to true DTW).

    Returns
    -------
    sdtw: float or array, shape=[m + 2, n + 2]
        soft-DTW discrepancy if ret_R = false
        accumulated cost matrix R if ret_R = true
    """
    m, n = D.shape

    # Allocate memory.
    # We need +2 because we use indices starting from 1
    # and to deal with edge cases in the backward recursion.
    R = np.zeros((m+2, n+2))

    _soft_dtw(D, R, gamma=gamma)

    if ret_R:
        return R
    else:
        return R[m, n]


def soft_dtw_from_dist_grad(D, R, gamma=1.0):
    """
    Compute gradient of soft-DTW w.r.t. D by dynamic programming.

    Parameters
    ----------
    D: array, shape = [m, n]
        Distance matrix between elements of both time series.

    R: array, shape = [m + 2, n + 2]
        Accumulated cost matrix.

    gamma: float
        Regularization parameter (lower is less smoothed, i.e.,
        closer to true DTW).

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

    _soft_dtw_grad(D, R, E, gamma=gamma)

    return E[1:-1, 1:-1]
