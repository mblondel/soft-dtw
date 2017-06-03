# Author: Mathieu Blondel
# License: Simplified BSD

# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
np.import_array()


from libc.float cimport DBL_MAX
from libc.math cimport exp, log
from libc.string cimport memset


cdef inline double _softmin3(double a,
                             double b,
                             double c,
                             double gamma):
    a /= -gamma
    b /= -gamma
    c /= -gamma

    cdef double max_val = max(max(a, b), c)

    cdef double tmp = 0
    tmp += exp(a - max_val)
    tmp += exp(b - max_val)
    tmp += exp(c - max_val)

    return -gamma * (log(tmp) + max_val)


def _soft_dtw(np.ndarray[double, ndim=2] D,
              np.ndarray[double, ndim=2] R,
              double gamma):

    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    cdef int i, j

    # Initialization.
    memset(<void*>R.data, 0, (m+1) * (n+1) * sizeof(double))

    for i in range(m + 1):
        R[i, 0] = DBL_MAX

    for j in range(n + 1):
        R[0, j] = DBL_MAX

    R[0, 0] = 0

    # DP recursion.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # D is indexed starting from 0.
            R[i, j] = D[i-1, j-1] + _softmin3(R[i-1, j],
                                              R[i-1, j-1],
                                              R[i, j-1],
                                              gamma)
