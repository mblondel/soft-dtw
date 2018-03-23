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

    def jacobian_product(self, E, sakoe_chiba_band=-1):
        """
        Compute the product between the Jacobian
        (a linear map from m x d to m x n) and a matrix E.

        Parameters
        ----------
        E: array, shape = [m, n]
            Second time series.

        sakoe_chiba_band: int
            If non-negative, the Jacobian product is restricted to a
            Sakoe-Chiba band around the diagonal, in E.
            The band has a width of 2 * sakoe_chiba_band + 1.

        Returns
        -------
        G: array, shape = [m, d]
            Product with Jacobian
            ([m x d, m x n] * [m x n] = [m x d]).
        """
        G = np.zeros_like(self.X)

        if sakoe_chiba_band >= 0:
            assert E.shape[0] == E.shape[1]

        _jacobian_product_sq_euc(self.X, self.Y, E, G,
                                 sakoe_chiba_band=sakoe_chiba_band)

        return G
