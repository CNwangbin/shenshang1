import pytest
import pandas as pd
import numpy as np

from shenshang.util import (
    fetch_tri, convert_format, compare_matrices, argsort_arrays, combine)


@pytest.mark.parametrize(
    'square, lower, exp1, exp2, exp3',
    [(np.array([[1.0, 0.2],
                [0.1, 1.0]]),
      True,
      np.array([0.1]), np.array([1]), np.array([0])),
     (np.array([[1.0, 0.2],
                [0.1, 1.0]]),
      False,
      np.array([0.2]), np.array([0]), np.array([1])),
     (np.array([[np.nan, np.nan, np.nan],
                [np.nan, np.nan, 0.6],
                [0.7,    0.8,    np.nan]]),
      True,
      np.array([np.nan, 0.7, 0.8]), np.array([1, 2, 2]), np.array([0, 0, 1])),
     (np.array([[np.nan, np.nan, 0.6],
                [np.nan, np.nan, np.nan],
                [0.7,    0.8,    np.nan]]),
      False,
      np.array([np.nan, 0.6, np.nan]), np.array([0, 0, 1]), np.array([1, 2, 2]))])
def test_fetch_tri(square, lower, exp1, exp2, exp3):
    p, idx1, idx2 = fetch_tri(square, lower=lower)
    np.testing.assert_array_equal(p, exp1)
    assert np.all(idx1 == exp2)
    assert np.all(idx2 == exp3)


@pytest.mark.parametrize(
    'stat, pval, cutoff, exp',
    [(pd.DataFrame([[np.nan, 9.9],
                    [np.nan, np.nan]]),
      pd.DataFrame([[np.nan, 0.099],
                    [0.91,   np.nan]]),
      0.1,
      pd.DataFrame([[0, 1, 9.9, 0.099, '+']],
                    columns=('feature1', 'feature2', 'stat', 'pval', 'type'))),
     (pd.DataFrame([[np.nan, np.nan, 6.0],
                    [np.nan, np.nan, 7.0],
                    [np.nan, np.nan, np.nan]],
                   index=['OTU1', 'OTU2', 'OTU3'],
                   columns=['OTU1', 'OTU2', 'OTU3']),
      pd.DataFrame([[np.nan, np.nan, 0.001],
                    [np.nan, np.nan, np.nan],
                    [0.7,    0.008,  np.nan]],
                   index=['OTU1', 'OTU2', 'OTU3'],
                   columns=['OTU1', 'OTU2', 'OTU3']),
      0.01,
      pd.DataFrame([['OTU1', 'OTU3', 6.0, 0.001, '+'],
                    ['OTU3', 'OTU2', 7.0, 0.008, '-']],
                    columns=('feature1', 'feature2', 'stat', 'pval', 'type')))])
def test_convert_format(stat, pval, cutoff, exp):
    df = convert_format(stat, pval, cutoff=cutoff)
    pd.util.testing.assert_frame_equal(exp, df)


@pytest.mark.parametrize(
    'pvals, sigs, cutoff, exp',
    [((pd.DataFrame([[np.nan, 0.01],
                     [0.02,   np.nan]]),
       pd.DataFrame([[np.nan, 0.03],
                     [0.03,   np.nan]]),
       pd.DataFrame([[np.nan, 0.05],
                     [0.06,   np.nan]])),
      [True, True, False],
      0.04,
      pd.DataFrame([[0, 1, '+', 0.01, 0.03, 0.05],
                    [1, 0, '-', 0.02, 0.03, 0.06]])),
     ((pd.DataFrame([[np.nan, 0.01],
                     [0.02,   np.nan]],
                    index=['f1', 'f2'],
                    columns=['f1', 'f2']),
       pd.DataFrame([[np.nan, 0.05],
                     [0.06,   np.nan]],
                    index=['f1', 'f2'],
                    columns=['f1', 'f2'])),
      [False, True],
      0.01,
      pd.DataFrame(columns=[0, 1, 2, 3, 4],
                   index=pd.RangeIndex(0),
                   dtype=float)),
     ((pd.DataFrame([[np.nan, 0.03],
                     [0.04,   np.nan]],
                    index=['f1', 'f2'],
                    columns=['f1', 'f2']),
       pd.DataFrame([[np.nan, 0.01],
                     [0.02,   np.nan]],
                    index=['f1', 'f2'],
                    columns=['f1', 'f2']),
       pd.DataFrame([[np.nan, 0.05],
                     [0.06,   np.nan]],
                    index=['f1', 'f2'],
                    columns=['f1', 'f2'])),
      [False, True, False],
      0.03,
      pd.DataFrame([['f1', 'f2', '+', 0.03, 0.01, 0.05],
                    ['f2', 'f1', '-', 0.04, 0.02, 0.06]]))])
def test_compare_matrices(pvals, sigs, cutoff, exp):
    df = compare_matrices(pvals, sigs, cutoff=cutoff)
    pd.util.testing.assert_frame_equal(df, exp)


@pytest.mark.parametrize(
    'arrs, ascending, exps',
    [((np.array([1, 2, 1, 3, 1, 4, 1, 5, 1, 6]),
       np.array([6, 5, 3, 4, 4, 9, 7, 1, 8, 9])),
      True,
      (np.array([1, 1, 1, 1, 1, 2, 3, 4, 5, 6]),
       np.array([3, 4, 6, 7, 8, 5, 4, 9, 1, 9]))),
     ((np.array([1, 6, 0, 1, 3, 1, 4, 1, 5, 1, 2, 0]),
       np.array([6, 5, 2, 3, 4, 4, 9, 7, 1, 8, 9, 7])),
      [True, False],
      (np.array([0, 0, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6]),
       np.array([7, 2, 8, 7, 6, 4, 3, 9, 4, 9, 1, 5]))),
     ((np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 5, 0, 0, 0]),
       np.array([36,8, 20,16,36,3, 10,2, 1, 25,11,0, 2, 0, 6, 4, 10,38,4])),
      [True, False],
      (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 5]),
       np.array([38,36,36,20,16,10,10,8, 6, 4, 3, 2, 2, 1, 0, 0, 11,25,4]))),
     # if the input is a single array:
     ((np.array([3, 1, 2]),),
      False,
      (np.array([3, 2, 1]),))])
def test_argsort_arrays(arrs, ascending, exps):
    idx = argsort_arrays(arrs, ascending)
    for arr, exp in zip(arrs, exps):
        assert np.all(exp == arr[idx])
    # from pprint import pprint
    # pprint(y[idx])
    # pprint(x[idx])


def test_combine():
    ids = ['o1', 'o2']
    pval1 = pd.DataFrame([[np.nan, 0.99],
                          [0.01,   np.nan]],
                         columns=ids,
                         index=ids)
    stat1 = pd.DataFrame([[np.nan, 99],
                          [np.nan,   np.nan]],
                         columns=ids,
                         index=ids)
    pval2 = pd.DataFrame([[np.nan, 0.9],
                          [0.1,   np.nan]],
                         columns=ids,
                         index=ids)
    stat2 = pd.DataFrame([[np.nan, 9],
                          [np.nan,   np.nan]],
                         columns=ids,
                         index=ids)
    pval3 = pd.DataFrame([[np.nan, 0.89],
                          [0.08,   np.nan]],
                         columns=ids,
                         index=ids)
    stat3 = pd.DataFrame([[np.nan, 8.9],
                          [np.nan, np.nan]],
                         columns=ids,
                         index=ids)
    obs = combine(cutoff=1, binary=(stat1, pval1), rank=(stat2, pval2), logratio=(stat3, pval3))
    exp = {'binary_stat': [99.0, 99.0],
           'binary_pval': [0.99, 0.01],
           'binary_type': ['+', '-'],
           'rank_stat': [9.0, 9.0],
           'rank_pval': [0.9, 0.1],
           'rank_type': ['+', '-'],
           'logratio_stat': [8.9, 8.9],
           'logratio_pval': [0.89, 0.08],
           'logratio_type': ['+', '-']}

    assert exp == obs.to_dict('list')
