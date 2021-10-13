import numpy as np
from scipy.stats import rankdata


def permute_table(m, inplace=False, seed=None):
    '''Randomly permute each feature in a sample-by-feature table.

    This creates a table for null model. The advantage of this is that
    it doesn't change distribution for any feature in the original
    table, thus keeping the original data characteristics.

    Parameters
    ----------
    m : 2-D numeric array
        sample-by-feature table. Usually it is a real table from real world.
    inplace : bool, optional
        True to modify current table, False (default) to create a new table.
    seed : int or None (default)
        random seed for random permutation.

    Returns
    -------
    permuted table

    '''
    if not inplace:
        m = np.copy(m)
    rng = np.random.default_rng(seed)
    for i in range(m.shape[1]):
        rng.shuffle(m[:, i])
        # m[:, i] = rng.permutation(m[:, i])
    return m


def simulate_compositionality(m, inplace=False, seed=None):
    '''
    '''


def simulate_correlation(m, structure=((10, 0.1), (10, -0.1),
                                       (10, 0.2), (10, -0.2),
                                       (10, 0.3), (10, -0.3)),
                         inplace=False, seed=None):
    '''Simulate correlation structure (feature-wise) in the sample-by-feature table.

    It shuffles randomly selected features (on the column).

    Recommend to create a null table using `permute_table`
    first and then simulate correlation structure within it.

    Parameters
    ----------
    m : 2-D numeric array
        sample-by-feature table. Usually it is created from `permute_table`.
    structure : list-like of 2-item tuple
        correlation structure. 1st item is the number of pairs of
        features randomly chosen from the table; 2nd item is the
        correlation strength to simulate for those pairs passing to
        `correlate_xy`.

    Returns
    -------
    sample-by-feature table
        updated table with specified correlation structure.
    2-D array
        x-by-2 array. Each row is a pair of feature indices that are correlated.
        The row number is the sum of all 1st item in the tuples of `structure`.
    1-D array
        Each item is the target correlation strength specified in `structure`,
        in the correponding order of the rows in the above 2-D array.
    '''
    if not inplace:
        m = np.copy(m)
    rng = np.random.default_rng(seed)
    select_sizes = [i * 2 for i, _ in structure]
    select = rng.choice(m.shape[1], sum(select_sizes), replace=False)
    select_idx = np.split(select, np.cumsum(select_sizes)[:-1])
    for (_, strength), idx in zip(structure, select_idx):
        it = iter(idx)
        # zip(it, it) runs next() twice
        for i, j in zip(it, it):
            x, y = m[:, i], m[:, j]
            correlate_xy(x, y, strength=strength, inplace=True)
    return m, select.reshape(-1, 2), np.concatenate([[i] * j for j, i in structure])


def correlate_xy(x, y, strength=1, inplace=True, seed=None):
    '''Correlate x and y by sorting y according to x.

    It assumes the input `x` and `y` are uncorrelated. It sorts y but
    keeps x unchanged. The target strength of correlation determines
    the fraction of the vector `y` to be sorted according to `x`. The
    resulting correlation is a rank correlation (spearman).

    ..notice:: y is sorted in place by default.

    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size.
        the abundance of the feature across all samples
    strength : float of [-1, 1]
        the target strength of correlation between x and y after
        sorting. Negative (positive) values means negative (positive)
        correlation. For example, if strength is set to -0.5, randomly half
        values of y array will sorted in the opposite order of
        corresponding values of x array.

    Returns
    -------
    1-D numeric array
        sorted y

    '''
    # from pprint import pprint
    # pprint(x)
    # pprint(y)
    if not inplace:
        y = np.copy(y)
    rng = np.random.default_rng(seed)
    size = round(x.size * abs(strength))
    select = np.sort(rng.choice(x.size, size, replace=False))
    y_select = y[select]
    sx = x[select].argsort()
    # this is how to sort y according to x:
    # y[x.argsort()] = y[y.argsort()]
    if strength < 0:
        sx = sx[::-1]
    y_select[sx] = y_select[y_select.argsort()]
    y[select] = y_select
    # pprint(x[select])
    # pprint(y[select])
    return y


def correlate_xy_TODO1(x, y, noise, positive=False):
    '''Correlate x and y by sorting y according to x.

    It sorts y but keeps x unchanged. The strength of correlation is
    determined by the random perturbance on the argsort:
    1. get argsort that matches the order of y according to x;
    2. add random noise to the argsort;
    3. use the perturbed argsort to sort y.

    ..notice:: y is sorted in place by default.

    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size.
        the abundance of the feature across all samples
    noise :

    Returns
    -------
    1-D numeric array
        sorted y

    '''
    sx = np.argsort(x)
    sy = np.argsort(y_rank + noise)


def correlate_xy_TODO2(x, y, noise, positive=False):
    '''Correlate x and y by sorting y according to x.

    It sorts y but keeps x unchanged. The procedure is:
    1. add random noise to y;
    2. get argsort that matches the order of y according to x;
    3. strip away the random noised added to y to restore original values;
    4. return sorted and restored y

    ..notice:: y is sorted in place by default.

    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size.
        the abundance of the feature across all samples
    positive : bool
        return the sorted y as postively or negatively correlated to x

    Returns
    -------
    1-D numeric array
        sorted y
    '''
    y_rank = rankdata(y)
    sx = np.argsort(x)
    sy = np.argsort(y + noise)
