import pytest
import numpy as np
from matplotlib.testing.decorators import image_comparison

from shenshang.visualize import (
    plot_sorted_bars, plot_sorted_shades,
    plot_joint_scatter, plot_confusion_matrix)


@pytest.fixture(scope="module")
def parametrize_xy(request):
    arrays = dict(a=np.zeros(500),
                  b=np.ones(500),
                  c=np.array([1] * 2 + [5] * 8),
                  d=np.array([1] * 3 + [2] * 7),
                  e=np.array([9] * 2 + [1] * 8),
                  f=np.array([7] * 5 + [1] * 5),
                  g=np.array([7] * 7 + [1] * 3),
                  h=np.array([0, 1, 20, 30, 40, 50, 60, 70, 80, 90]))

    return [arrays[var_name] for var_name in request.param]


@pytest.mark.parametrize('parametrize_xy',
                         [('a', 'b'),
                          ('c', 'd'),
                          ('e', 'f'),
                          ('g', 'h')],
                         indirect=True)
def test_plot_sorted_bars(parametrize_xy):
    x, y = parametrize_xy
    ax = plot_sorted_bars(x, y)
    exp = np.array([i.get_height() for i in ax.patches])
    n = x.size
    assert n * 2 == len(exp)
    # test x is sorted
    sorted_idx = np.argsort(x)
    assert np.all(x[sorted_idx] == exp[:n])
    # test if x-y are corresponding
    pairs = set(zip(x, -y))
    assert pairs == set(zip(exp[:n], exp[n:]))


@pytest.mark.parametrize('parametrize_xy',
                         [('a', 'b'),
                          ('c', 'd'),
                          ('e', 'f'),
                          ('g', 'h')],
                         indirect=True)
def test_plot_sorted_shades(parametrize_xy):
    x, y = parametrize_xy
    ax = plot_sorted_shades(x, y)
    # [0] is for x-axis, [1] is y-axis
    exp_x, exp_y = [line.get_data()[1] for line in ax.get_lines()]
    # test x is sorted
    sorted_idx = np.argsort(x)
    assert np.all(x[sorted_idx] == exp_x)
    # test if x-y are corresponding
    pairs = set(zip(x, y))
    assert pairs == set(zip(exp_x, exp_y))


@pytest.mark.parametrize('parametrize_xy',
                         [('a', 'b'),
                          ('c', 'd'),
                          ('e', 'f'),
                          ('g', 'h')],
                         indirect=True)
def test_plot_joint_scatter(parametrize_xy):
    x, y = parametrize_xy
    _, ax1, ax2, ax3 = plot_joint_scatter(x, y)
    # check the data points in the scatter plot
    cs = ax1.collections[0]
    obs = cs.get_offsets()
    assert np.all(np.array(obs[:, 0]) == x)
    assert np.all(np.array(obs[:, 1]) == y)


@pytest.mark.parametrize('parametrize_xy, ratio, mask, cutoff, arr',
                         [(('c', 'd'), True, False, 2, [0.29, 0, 0.14, 1.00]),
                          (('c', 'd'), True,  True, 2, [0, 0.14, 1.00]),
                          (('c', 'd'), False, True, 2, [0, 1, 7]),
                          (('c', 'd'), False, False, 2, [2, 0, 1, 7])],
                         indirect=['parametrize_xy'])
def test_plot_confusion_matrix(parametrize_xy, ratio, mask, cutoff, arr):
    x, y = parametrize_xy
    ax = plot_confusion_matrix(x, y, cutoff=cutoff, ratio=ratio, mask=mask)
    ax_arr = [float(i.get_text()) for i in ax.texts]
    assert np.all(arr == ax_arr)
