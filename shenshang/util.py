from logging import getLogger

import numpy as np
import pandas as pd


def set_log_level(level):
    '''Set the log level.

    You can see the logging levels at:
    https://docs.python.org/3.5/library/logging.html#levels

    Parameters
    ----------
    level : int or str
        10 for debug, 20 for info, 30 for warn, etc.
        It is passing to :func:`logging.Logger.setLevel`

    '''
    logger = getLogger(__package__)
    logger.setLevel(level)


def argsort_arrays(arrs, ascending=True):
    '''argsort the input 1-D arrays sequentially.

    This function keeps the corresponding associations across input arrays.

    Parameters
    ----------
    arrs : a list of 1-D arrays
        positional arguments to accept a single or multiple 1-D arrays
    ascending : bool or a list of bool
        ascending or descending for each array

    Returns
    -------
    index_array : ndarray, int
        Array of indices that sort the input arrays in sequential order.
    '''
    idx = np.arange(arrs[0].size)
    if isinstance(ascending, bool):
        ascending = [ascending] * len(arrs)
    for x, a in zip(reversed(arrs), reversed(ascending)):
        idx2 = np.argsort(x[idx], kind='mergesort')
        idx = idx[idx2]
        if not a:
            idx = idx[::-1]
    return idx


def fetch_tri(square, lower=True):
    '''Return values in lower (or upper triangle) and their indices in the
square matrix.

    Parameters
    ----------
    square : 2-D square array

    lower : bool
        whether to fetch values in lower triangle or upper triangle

    Returns
    -------
    tuple of 3 arrays of the same length
        - values
        - row indices in the square matrix
        - column indices in the square matrix
    '''
    n = square.shape[0]
    if lower:
        idx = np.tril_indices(n, -1)
    else:
        idx = np.triu_indices(n, 1)
    v = square[idx]
    return v, idx[0], idx[1]


def convert_format(stat, pval, cutoff=0.01,
                   columns=('feature1', 'feature2', 'stat', 'pval', 'type')):
    '''Convert matrix to edge file format.

    The edge file format has each co-occurrence in a row. The first 2
    columns are the IDs of the the feature pairs.  The next columns
    are the statistic, p-value, and correlation type (negative or
    positive) for each feature pairs.

    The edge file can be provided to Cytoscape for network analysis.

    Parameters
    ----------
    stat, pval : pd.DataFrame
        square matrices of statistics and p-values between features on index and column
    cutoff : float
        p-value cutoff. All pairs with p-values in the interval [0, cutoff) will be kept
    columns : iterable of strings
        the column names for the return pd.DataFrame

    Returns
    -------
    pd.DataFrame
        1st column: feature 1
        2nd column: feature 2
        3rd column: statistic between features 1 and 2
        4th column: p-value between features 1 and 2
        5th column: '+' for positive correlation; '-' for negative correlation
    '''
    l = []
    ids = pval.index
    for lower, t in [(False, '+'), (True, '-')]:
        for p, i, j in zip(*fetch_tri(pval.values, lower=lower)):
            # stat is only located in the upper triangle
            if lower:
                s = stat.values[j, i]
            else:
                s = stat.values[i, j]
            # use `<` sign not only selects significant values, but also
            # discards np.nan too.
            if p < cutoff:
                l.append([ids[i], ids[j], s, p, t])
    return pd.DataFrame(l, columns=columns)


def compare_matrices(pvals, sigs, cutoff=0.01):
    '''Find the items across p-value matrices that satisfy the sigs.

    Parameters
    ----------
    pvals : list of square pd.DataFrame
        p-values from each methods
    sigs : tuple of bool
        filter to keep items that satisfy the significance patterns across pvals
    cutoff : float
        p-value cutoff

    Returns
    -------
    pd.DataFrame
        1st column: feature 1
        2nd column: feature 2
        3rd column: '+' for positive correlation; '-' for negative correlation
        rest columns: each one contains p-value for one method
    '''
    mask = np.ones_like(pvals[0], dtype=bool)
    assert len(pvals) == len(sigs)
    for pval, pat in zip(pvals, sigs):
        m = pval < cutoff
        # do NOT use `>` for False value of pat, because np.nan >
        # cutoff returns False while we want it to return True.
        # This is important for the p-values from rank method.
        if not pat:
            m = ~m
        mask = mask & m

    col1, col2 = np.where(mask)
    # a list of 1-D arrays
    p = [pval.values[mask.values] for pval in pvals]
    ids = pvals[0].index
    df = pd.DataFrame(
        dict(zip(range(3+len(p)),
                 ([ids[i] for i in col1],
                  [ids[i] for i in col2],
                  ['+' if i else '-' for i in col1 < col2],
                  *p))))
    return df


def combine(cutoff=0.05, **results):
    '''the results should be oriented the same.

    Parameters
    ----------
    results : dict
        keyword arguments. Key is the label text; value is the
        tuple/list of stat and pval that are square pd.DataFrame (eg
        `binary=[stat, pval], rank=[stat, pval]`)

    Returns
    -------
    '''
    dfs = []
    for i, name in enumerate(results, 1):
        stat, pval = results[name]
        df = convert_format(
            stat, pval, cutoff,
            columns=('feature1', 'feature2',
                     '{}_stat'.format(name),
                     '{}_pval'.format(name),
                     '{}_type'.format(name)))
        dfs.append(df.set_index(['feature1', 'feature2']))
    # import pdb; pdb.set_trace()
    return pd.concat(dfs, axis=1)
