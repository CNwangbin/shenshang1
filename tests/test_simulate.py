import numpy as np
import pytest
from scipy.stats import spearmanr, rankdata

from shenshang.simulate import simulate_correlation, correlate_xy


@pytest.mark.parametrize(
    'strength, x, y, exp',
    [(1,
      np.array([5, 8, 2, 4, 3, 0, 1, 6, 7, 9]),
      np.array([9, 1, 4, 5, 2, 3, 6, 8, 0, 7]),
      np.array([5, 8, 2, 4, 3, 0, 1, 6, 7, 9])),
     (-1,
      np.array([5, 8, 2, 4, 3, 0, 1, 6, 7, 9]),
      np.array([9, 1, 4, 5, 2, 3, 6, 8, 0, 7]),
      np.array([4, 1, 7, 5, 6, 9, 8, 3, 2, 0])),
     (0,
      np.array([5, 8, 2, 4, 3, 0, 1, 6, 7, 9]),
      np.array([9, 1, 4, 5, 2, 3, 6, 8, 0, 7]),
      np.array([9, 1, 4, 5, 2, 3, 6, 8, 0, 7])),
     (0.5,
      np.array([5, 8, 2, 4, 3, 0, 1, 6, 7, 9]),
      np.array([9, 1, 4, 5, 2, 3, 6, 8, 0, 7]),
      np.array([9, 8, 1, 5, 2, 3, 0, 4, 6, 7])),
     (-0.5,
      np.array([5, 8, 2, 4, 3, 0, 1, 6, 7, 9]),
      np.array([9, 1, 4, 5, 2, 3, 6, 8, 0, 7]),
      np.array([9, 0, 6, 5, 2, 3, 8, 4, 1, 7])),
     (-0.8,
      np.array([0] * 5 + list(range(10))),
      np.array([0] * 5 + list(range(9, -1, -1))),
      np.array([9, 8, 7, 0, 6, 5, 4, 2, 1, 0, 0, 3, 0, 0, 0]))])
def test_correlate_xy(strength, x, y, exp):
    assert np.all(correlate_xy(x, y, strength, inplace=False, seed=9) == exp)
    # print(spearmanr(x, y))
    # print(spearmanr(x, exp))
    pos = y != exp
    sx = rankdata(x[pos], 'ordinal')
    if strength < 0:
        sy = rankdata(-exp[pos], 'ordinal')
    else:
        sy = rankdata(exp[pos], 'ordinal')
    np.testing.assert_array_equal(sx, sy)


def test_simulate_cor():
    m = np.array([[ 0,  0, 37, 28, 63,  0,  0, 34,  4, 90],
                  [46, 75, 77, 78, 18,  5,  5,  0, 26, 50],
                  [73,  0,  0, 89, 24, 81, 48,  0, 26, 25],
                  [28, 50, 25, 91, 46,  0,  2, 76,  0, 92],
                  [81, 60, 54, 40, 12, 66, 51, 23,  0, 92],
                  [66,  0, 64, 43, 83, 35,  6, 89,  0, 78],
                  [97, 52, 26, 89, 24,  0, 25, 15, 92, 53],
                  [88,  0, 49, 16,  0, 92, 61, 15,  0, 78],
                  [ 0, 64,  0, 76, 43,  3,  0, 88, 31, 99],
                  [24, 69, 57, 17, 96, 47, 58, 29, 80, 15]])
    structure = ((2, 0.5), (2, -0.5))
    obs = simulate_correlation(m, structure, inplace=False, seed=7)
    np.testing.assert_array_equal(obs[1], np.array([[7, 8],
                                                    [9, 5],
                                                    [6, 3],
                                                    [4, 2]]))
    np.testing.assert_array_equal(obs[2], np.array([ 0.5, 0.5, -0.5, -0.5]))
