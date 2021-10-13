from contextlib import ExitStack as does_not_raise

import numpy as np
from scipy.stats import pearsonr
import pytest

from shenshang.cooccur import cooccur_binary, cooccur_logratio, cooccur_rank


# fixture is a bit like setUp in unittest.TestCase
# this fixture causes the parametrize_xy only invoked once for the current module.
@pytest.fixture(scope="module")
def parametrize_xy(request):
    arrays = dict(a=np.zeros(100),
                  b=np.ones(100),
                  # c: --========
                  # d: ---=======
                  # e: ==--------
                  # f: =====-----
                  # g: =======---
                  c=np.array([1] * 2 + [5] * 8),
                  d=np.array([1] * 3 + [2] * 7),
                  e=np.array([9] * 2 + [1] * 8),
                  f=np.array([7] * 5 + [1] * 5),
                  g=np.array([7] * 7 + [1] * 3),

                  h=np.arange(10),
                  i=np.arange(10, 0, -1),
                  j=np.array([0, 1, 20, 30, 40, 50, 60, 70, 80, 90]))

    return [arrays[var_name] for var_name in request.param]


@pytest.mark.parametrize('parametrize_xy',
                         [(('e', 'd')),
                          (('e', 'c')),
                          (('g', 'e')),
                          (('g', 'f')),
                          (('c', 'f')),
                          (('c', 'g'))],
                         indirect=['parametrize_xy'])
def test_cooccur_rank(parametrize_xy):
    x, y = parametrize_xy
    o, p, n, dist = cooccur_rank(x, y, 1000, seed=9)
    if o < 0:
        p = n
    r, pval = pearsonr(x, y)
    print(o, p)
    print(r, pval)


@pytest.mark.parametrize('parametrize_xy',
                         [('a', 'a'),
                          ('a', 'b'),
                          ('a', 'c')],
                         indirect=['parametrize_xy'])
def test_cooccur_binary_raise(parametrize_xy):
    x, y = parametrize_xy
    with pytest.raises(ValueError, match='x or y or both are absent or present in all samples'):
        cooccur_binary(x, y, cutoff=1)


@pytest.mark.parametrize('parametrize_xy, real, pval_pos, pval_neg',
                         [(('e', 'd'), (0+1)/(9+1),  0.9610, 0.0399),
                          (('e', 'c'), (0+1)/(10+1), 0.9900, 0.0109),
                          (('g', 'e'), (2+1)/(7+1),  0.2338, 0.7672),
                          (('g', 'f'), (5+1)/(7+1),  0.0430, 0.9580),
                          (('c', 'f'), (3+1)/(10+1), 0.8891, 0.1118),
                          (('c', 'g'), (5+1)/(10+1), 0.7702, 0.2307)],
                         indirect=['parametrize_xy'])
def test_cooccur_binary(parametrize_xy, real, pval_pos, pval_neg):
    x, y = parametrize_xy
    o, p, n, dist = cooccur_binary(x, y, cutoff=1, seed=9)
    assert o == pytest.approx(real, abs=0.0001)
    assert p == pytest.approx(pval_pos, abs=0.0001)
    assert n == pytest.approx(pval_neg, abs=0.0001)
