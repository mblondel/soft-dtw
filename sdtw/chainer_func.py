import numpy as np

import chainer

from chainer import Function

from .soft_dtw import SoftDTW
from .distance import SquaredEuclidean


class SoftDTWLoss(Function):

    def __init__(self, gamma):
        self.gamma = gamma

    def forward_cpu(self, inputs):
        # Z, X: both are arrays of shape length x n_dim
        Z, X = inputs

        assert Z.shape[1] == X.shape[1]

        D = SquaredEuclidean(Z, X)
        self.sdtw_ = SoftDTW(D, gamma=self.gamma)
        loss = self.sdtw_.compute()

        return np.array(loss),

    def backward_cpu(self, inputs, grad_outputs):
        Z, X = inputs
        # g has the same shape as the output of forward_cpu().
        # g should always be 1 since it's the last function (loss function)
        g, = grad_outputs

        D = SquaredEuclidean(Z, X)
        E = self.sdtw_.grad()
        gZ = D.jacobian_product(E).astype(Z.dtype)

        # We don't need the gradient w.r.t. the 2nd argument.
        return gZ, np.zeros_like(X)
