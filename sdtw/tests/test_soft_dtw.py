import numpy as np

from scipy.optimize import approx_fprime

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_raises

from sdtw.path import gen_all_paths
from sdtw.distance import SquaredEuclidean
from sdtw import SoftDTW

# Generate two inputs randomly.
rng = np.random.RandomState(0)
X = rng.randn(5, 4)
Y = rng.randn(6, 4)
D = euclidean_distances(X, Y, squared=True)

# Generate two inputs with same length.
rng = np.random.RandomState(0)
Xs = rng.randn(6, 4)
Ys = rng.randn(6, 4)
Ds = euclidean_distances(Xs, Ys, squared=True)


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
        assert_almost_equal(SoftDTW(D, gamma).compute(),
                            _soft_dtw_bf(D, gamma=gamma))


def test_soft_dtw_grad():
    def make_func(gamma):
        def func(d):
            D_ = d.reshape(*D.shape)
            return SoftDTW(D_, gamma).compute()
        return func

    for gamma in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
        sdtw = SoftDTW(D, gamma)
        sdtw.compute()
        E = sdtw.grad()
        func = make_func(gamma)
        E_num = approx_fprime(D.ravel(), func, 1e-6).reshape(*E.shape)
        assert_array_almost_equal(E, E_num, 5)


def test_soft_dtw_grad_X():
    def make_func(gamma):
        def func(x):
            X_ = x.reshape(*X.shape)
            D_ = SquaredEuclidean(X_, Y)
            return SoftDTW(D_, gamma).compute()
        return func

    for gamma in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
        dist = SquaredEuclidean(X, Y)
        sdtw = SoftDTW(dist, gamma)
        sdtw.compute()
        E = sdtw.grad()
        G = dist.jacobian_product(E)

        func = make_func(gamma)
        G_num = approx_fprime(X.ravel(), func, 1e-6).reshape(*G.shape)
        assert_array_almost_equal(G, G_num, 5)


def test_soft_dtw_band_check_squared():
    # D is not squared
    SoftDTW(D, 1, sakoe_chiba_band=-1)
    assert_raises(AssertionError, SoftDTW, D, 1, 0)
    assert_raises(AssertionError, SoftDTW, D, 1, 1)

    # Ds is squared
    SoftDTW(Ds, 1, sakoe_chiba_band=-1)
    SoftDTW(Ds, 1, sakoe_chiba_band=0)
    SoftDTW(Ds, 1, sakoe_chiba_band=1)


def test_soft_dtw_grad_band():
    def make_func(gamma, sakoe_chiba_band):
        def func(d):
            D_ = d.reshape(*Ds.shape)
            return SoftDTW(D_, gamma, sakoe_chiba_band).compute()
        return func

    for gamma in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
        for sakoe_chiba_band in [-1, 0, 1, 3]:
            sdtw = SoftDTW(Ds, gamma, sakoe_chiba_band)
            sdtw.compute()
            E = sdtw.grad()
            func = make_func(gamma, sakoe_chiba_band)
            E_num = approx_fprime(Ds.ravel(), func, 1e-6).reshape(*E.shape)
            assert_array_almost_equal(E, E_num, 5)


def _path_is_in_band(A, sakoe_chiba_band):
    if sakoe_chiba_band < 0:
        return True
    A_in_band = np.tril(A, sakoe_chiba_band)
    A_in_band = np.tril(A_in_band.T, sakoe_chiba_band).T
    return np.all(A == A_in_band)


def _soft_dtw_band_bf(Ds, gamma, sakoe_chiba_band):
    costs = [np.sum(A * Ds) for A in gen_all_paths(Ds.shape[0], Ds.shape[1])
             if _path_is_in_band(A, sakoe_chiba_band)]
    return _softmin(costs, gamma)


def test_soft_dtw_band():
    gamma = 0.01
    for sakoe_chiba_band in [-1, 0, 1, 2]:
        assert_almost_equal(SoftDTW(Ds, gamma, sakoe_chiba_band).compute(),
                            _soft_dtw_band_bf(Ds, gamma, sakoe_chiba_band))
