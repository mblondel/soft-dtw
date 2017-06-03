# Author: Mathieu Blondel
# License: Simplified BSD

import numpy as np


def delannoy_num(m, n):
    """
    Number of paths from the southwest corner (0, 0) of a rectangular grid to
    the northeast corner (m, n), using only single steps north, northeast, or
    east.

    Named after French army officer and amateur mathematician Henri Delannoy.

    Parameters
    ----------
    m, n : int, int
        Northeast corner coordinates.

    Returns
    -------
    delannoy_num: int
        Delannoy number.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Delannoy_number
    """
    a = np.zeros([m+1, n+1])
    a[0,0] = 1

    for i in range(1, m+1):
        a[i,0] = 1

    for j in range(1, n+1):
        a[0,j] = 1

    for i in range(1, m+1):
        for j in range(1, n+1):
            a[i,j] = a[i-1, j] + a[i, j-1] + a[i-1, j-1]

    return a[m, n]


def gen_all_paths(m, n, start=None, M=None):
    """
    Generator that produces all possible paths between (1, 1) and (m, n), using
    only north, northeast, or east steps. Each path is represented as a (m, n)
    numpy array with ones indicating the path.

    Parameters
    ----------
    m, n : int, int
        Northeast corner coordinates.
    """
    if start is None:
        start = [0, 0]
        M = np.zeros((m, n))

    i, j = start
    M[i, j] = 1
    ret = []

    if i == m-1 and j == n-1:
        yield M
    else:
        if i < m - 1:
            # Can use yield_from starting from Python 3.3.
            for mat in gen_all_paths(m, n, (i+1, j), M.copy()):
                yield mat
        if i < m-1 and j < n-1:
            for mat in gen_all_paths(m, n, (i+1, j+1), M.copy()):
                yield mat
        if j < n-1:
            for mat in gen_all_paths(m, n, (i, j+1), M.copy()):
                yield mat
