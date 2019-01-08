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


cdef inline int is_outside_sakoe_chiba_band(int sakoe_chiba_band,
                                            int i,
                                            int j,
                                            int m,
                                            int n):
    """True if the Sakoe-Chiba band constraint is used, and if (i, j) is outside

    This constraints the wrapping to a band around the diagonal.
    """
    cdef int diff, bound

    if sakoe_chiba_band < 0:
        return 0
    else:
        # since (i, j) starts at (1, 1)
        i, j = i - 1, j - 1

        diff = i * (n - 1) - j * (m - 1)
        diff = abs(diff * 2)
        bound = max(m, n) * (sakoe_chiba_band + 1)
        is_in_band = diff < bound

        return not is_in_band


def _soft_dtw(np.ndarray[double, ndim=2] D,
              np.ndarray[double, ndim=2] R,
              double gamma,
              int sakoe_chiba_band=-1):

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

            if is_outside_sakoe_chiba_band(sakoe_chiba_band, i, j, m, n):
                R[i, j] = DBL_MAX
            else:
                # D is indexed starting from 0.
                R[i, j] = D[i-1, j-1] + _softmin3(R[i-1, j],
                                                  R[i-1, j-1],
                                                  R[i, j-1],
                                                  gamma)


def _soft_dtw_grad(np.ndarray[double, ndim=2] D,
                   np.ndarray[double, ndim=2] R,
                   np.ndarray[double, ndim=2] E,
                   double gamma,
                   int sakoe_chiba_band=-1):

    # We added an extra row and an extra column on the Python side.
    cdef int m = D.shape[0] - 1
    cdef int n = D.shape[1] - 1

    cdef int i, j
    cdef double a, b, c

    # Initialization.
    memset(<void*>E.data, 0, (m+2) * (n+2) * sizeof(double))

    for i in range(1, m+1):
        # For D, indices start from 0 throughout.
        D[i-1, n] = 0
        R[i, n+1] = -DBL_MAX

    for j in range(1, n+1):
        D[m, j-1] = 0
        R[m+1, j] = -DBL_MAX

    E[m+1, n+1] = 1
    R[m+1, n+1] = R[m, n]
    D[m, n] = 0

    # DP recursion.
    for j in reversed(range(1, n+1)):  # ranges from n to 1
        for i in reversed(range(1, m+1)):  # ranges from m to 1

            if is_outside_sakoe_chiba_band(sakoe_chiba_band, i, j, m, n):
                E[i, j] = 0
                R[i, j] = -DBL_MAX
            else:
                if E[i+1, j] == 0:
                    a = 0
                else:
                    a = exp((R[i+1, j] - R[i, j] - D[i, j-1]) / gamma)

                if E[i, j+1] == 0:
                    b = 0
                else:
                    b = exp((R[i, j+1] - R[i, j] - D[i-1, j]) / gamma)

                if E[i+1,j+1] == 0:
                    c = 0
                else:
                    c = exp((R[i+1, j+1] - R[i, j] - D[i, j]) / gamma)

                E[i, j] = E[i+1, j] * a + E[i, j+1] * b + E[i+1,j+1] * c


def _jacobian_product_sq_euc(np.ndarray[double, ndim=2] X,
                             np.ndarray[double, ndim=2] Y,
                             np.ndarray[double, ndim=2] E,
                             np.ndarray[double, ndim=2] G):
    cdef int m = X.shape[0]
    cdef int n = Y.shape[0]
    cdef int d = X.shape[1]
    cdef int i, j, k

    for i in range(m):
        for j in range(n):

            if E[i,j] == 0:
                continue
            for k in range(d):
                G[i, k] += E[i,j] * 2 * (X[i, k] - Y[j, k])
