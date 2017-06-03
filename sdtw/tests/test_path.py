from sklearn.utils.testing import assert_equal


from sdtw.path import gen_all_paths
from sdtw.path import delannoy_num


def test_gen_all_paths():
    assert_equal(len(list(gen_all_paths(2, 2))), 3)
    assert_equal(len(list(gen_all_paths(3, 2))), 5)
    assert_equal(len(list(gen_all_paths(4, 2))), 7)
    # delannoy_num counts paths from (0,0),
    # while gen_all_paths starts from (1,1).
    assert_equal(len(list(gen_all_paths(5, 7))), delannoy_num(5-1, 7-1))
    assert_equal(len(list(gen_all_paths(8, 6))), delannoy_num(8-1, 6-1))
