import numpy as np

from sklearn.metrics.pairwise import euclidean_distances


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
        self.X = X
        self.Y = Y

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
        m = self.X.shape[0]
        n = self.Y.shape[0]
        d = self.X.shape[1]

        G = np.zeros_like(self.X)

        for i in range(m):
            for j in range(n):
                for k in range(d):
                    G[i, k] += E[i,j] * 2 * (self.X[i, k] - self.Y[j, k])

        return G
