import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from .soft_dtw_fast import _jacobian_product_sq_euc


class SquaredEuclidean(object):

    def __init__(self, X, Y):
        """
        Parameters
        ----------
        X: array, shape = [m, d]
            First time series.

        Y: array, shape = [n, d]
            Second time series.
        """
        self.X = X.astype(np.float64)
        self.Y = Y.astype(np.float64)

    def compute(self):
        """
        Compute distance matrix.

        Returns
        -------
        D: array, shape = [m, n]
            Distance matrix.
        """
        return euclidean_distances(self.X, self.Y, squared=True)

    def jacobian_product(self, E):
        """
        Compute the product between the Jacobian
        (a linear map from m x d to m x n) and a matrix E.

        Parameters
        ----------
        E: array, shape = [m, n]
            Second time series.

        Returns
        -------
        G: array, shape = [m, d]
            Product with Jacobian
            ([m x d, m x n] * [m x n] = [m x d]).
        """
        G = np.zeros_like(self.X)

        _jacobian_product_sq_euc(self.X, self.Y, E, G)

        return G
