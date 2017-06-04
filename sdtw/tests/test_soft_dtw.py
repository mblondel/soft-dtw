import numpy as np

from scipy.optimize import approx_fprime

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal

from sdtw.path import gen_all_paths
from sdtw import soft_DTW

# Generate two inputs randomly.
rng = np.random.RandomState(0)
X = rng.randn(5, 4)
Y = rng.randn(6, 4)
D = euclidean_distances(X, Y, squared=True)


def _softmax(z):
    max_val = np.max(z)
    return max_val + np.log(np.exp(z - max_val).sum())


def _softmin(z, gamma):
    z = np.array(z)
    return -gamma * _softmax(-z / gamma)


def _soft_dtw_bf(D, gamma):
    costs = [np.sum(A * D) for A in gen_all_paths(D.shape[0], D.shape[1])]
    return _softmin(costs, gamma)


def test_soft_dtw():
    for gamma in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
        assert_almost_equal(soft_DTW(gamma).compute(D),
                            _soft_dtw_bf(D, gamma=gamma))

def test_soft_dtw_grad():
    def make_func(gamma):
        def func(d):
            D_ = d.reshape(*D.shape)
            return soft_DTW(gamma).compute(D_).ravel()
        return func

    for gamma in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
        sdtw = soft_DTW(gamma)
        sdtw.compute(D)
        E = sdtw.grad(D)
        func = make_func(gamma)
        E_num = approx_fprime(D.ravel(), func, 1e-6).reshape(*E.shape)
        assert_array_almost_equal(E, E_num, 5)
