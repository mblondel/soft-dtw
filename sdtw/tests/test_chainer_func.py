import numpy as np
from chainer import Variable

from sdtw.dataset import load_ucr
from sdtw.chainer_func import SoftDTWLoss
from scipy.optimize import check_grad


def _func(z, X):
    shape = (-1, X.shape[1])
    Z = z.reshape(*shape)
    return SoftDTWLoss(gamma=0.1)(Z, X).data


def _grad(z, X):
    shape = (-1, X.shape[1])
    Z = z.reshape(*shape)
    Z = Variable(Z)
    loss = SoftDTWLoss(gamma=0.1)(Z, X)
    loss.backward(retain_grad=True)
    return Z.grad.ravel()


def test_grad():
    rng = np.random.RandomState(0)
    X = rng.randn(10, 2)
    Z = rng.randn(8, 2)
    print(check_grad(_func, _grad, Z.ravel(), X))
