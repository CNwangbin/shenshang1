from logging import getLogger

import numpy as np
from scipy.stats import trimboth, shapiro
from scipy.sparse import issparse
from pandas.core.algorithms import rank
from joblib import Parallel, delayed


logger = getLogger(__name__)


def cooccur_rank(x, y, perms=1000, seed=None):
    '''This computes pearson correlation between x and y, and use
    permutation to estimate p-value for significance.

    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size.
        the rank of the feature (ranked within each sample) across all samples
    perms : int
        the number of permutations to do for p-value computation.
    seed : int or None
        random seed for random permutation. `None` to not set it.

    Returns
    -------
    float
        The correlation coefficient (in the interval [-1, 1]. It is
        exactly the same with scipy.stats.pearsonr(x, y)).
    float
        p-value for negative correlation
    float
        p-value for positive correlation
    1-D float array
        distribution of correlation coefficients for permutated features.
    '''
    if np.min(x) == np.max(x) or np.min(y) == np.max(y):
        raise ValueError('x or y or both have the same value across all samples!')
    delta_x = x - np.mean(x)
    delta_y = y - np.mean(y)
    real = np.mean(delta_x * delta_y)
    dist = np.zeros(perms)
    rng = np.random.default_rng(seed)
    for i in range(perms):
        shuffled = rng.permutation(delta_y)
        dist[i] = np.mean(delta_x * shuffled)
    pval_neg = (1 + np.sum(dist <= real)) / (1 + perms)
    pval_pos = (1 + np.sum(dist >= real)) / (1 + perms)
    normalize = np.std(x) * np.std(y)
    return real / normalize, pval_pos, pval_neg, dist / normalize


def cooccur_logratio(x, y, clip=1, perms=1000, seed=None):
    '''
    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size.
        the abundance of the feature across all samples
    clip : numeric
        the threshold to clip the feature values on the lower bound.
    perms : int
        the number of permutations to do for p-value computation.
    seed : int or None
        random seed for random permutation. `None` to not set it.

    Returns
    -------
    float
        the correlation (in the interval [0, inf])
    float
        p-value of cooccurence
    float
        p-value of mutual exclusivity
    1-D float array
        distribution of co-occurrence for permutated features.
    '''
    x, y = np.copy(x), np.copy(y)
    # avoid division of zero.
    x[x < clip] = clip
    y[y < clip] = clip
    if np.min(x) == np.max(x) or np.min(y) == np.max(y):
        raise ValueError('x or y or both does not variation across all samples!')
    x, y = np.log2(x), np.log2(y)
    real = logratio_stat(x, y)
    dist = np.zeros(perms)
    rng = np.random.default_rng(seed)
    for i in range(perms):
        dist[i] = logratio_stat(x, rng.permutation(y))

    pval_pos = (1 + np.sum(dist >= real)) / (1 + perms)
    pval_neg = (1 + np.sum(dist <= real)) / (1 + perms)
    return real, pval_pos, pval_neg, dist


def logratio_stat(x, y, trim=0, threshold=0, weighted=True):
    '''Return geometric mean of the sum of positive and the sum of
    negative values of the ratios.

    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size.
        the log(abundance) of the feature across all samples
    trim : float of (0, 0.5)
        trim the fraction of extreme values at both upper and lower
        ends as outliers before calculation.
    threshold : float
        a positive value
    weighted : boolean

    Returns
    -------
    float
        the square of geometric mean. The smaller the value,
        the stronger the negative correlation between x and y; the
        larger the value, the stronger the positive correlation. If it
        is zero, x and y are perfectly correlated.
    '''
    logratio = trimboth(y - x, trim)
    if weighted:
        neg = np.sum(logratio[logratio < -threshold])
        pos = np.sum(logratio[logratio > threshold])
    else:
        neg = -np.sum(logratio < -threshold)
        pos = np.sum(logratio > threshold)
    # return geometric mean. since square root is monotonic, we can skip it
    return neg * pos


def cooccur_binary(x, y, cutoff=0, psudo=1, perms=1000, seed=None):
    '''Compute the overlap or exclusivity of 2 binary features.

    This function computes the overlap statistic, positive cooccurrence
    p-value, and negative cooccurence p-value.

    Parameters
    ----------
    x, y : 1-D numeric np.ndarray of the same size.
        the abundance of the feature across all samples
    cutoff : numeric
        the threshold to determine the absence/presence of the feature
    psudo : numeric
        psudo number passed to `binary_stat`.
    perms : int
        the number of permutations to do for p-value computation.
    seed : int or None
        random seed for random permutation. `None` to not set it.

    Returns
    -------
    float
        overlap stat
    float
        p-value of co-occurrence
    float
        p-value of mutual exclusivity
    1-D float array
        distribution of overlap stats for the permutated features.

    '''
    x = x > cutoff
    y = y > cutoff
    if np.all(x) or np.all(y) or not np.any(x) or not np.any(y):
        raise ValueError('x or y or both are absent or present in all samples!')
    real = binary_stat(x, y, psudo)
    dist = np.ones(perms)
    rng = np.random.default_rng(seed)
    for i in range(perms):
        dist[i] = binary_stat(x, rng.permutation(y), psudo, seed=rng)

    # 1 is added to avoid P-values of zero:
    # https://stats.stackexchange.com/a/112456/39796
    pval_pos = (1 + np.sum(dist >= real)) / (1 + perms)
    pval_neg = (1 + np.sum(dist <= real)) / (1 + perms)

    return real, pval_pos, pval_neg, dist


def binary_stat(x, y, psudo=1, noise_scale=0.00001, seed=None):
    '''Compute the degree of overlap of 2 binary features.

    Parameters
    ----------
    x, y : 1-D boolean arrays of the same size.
        The presence/absence of the feature across all samples.
    psudo : integer
        psudo count to add both nominator and denominator for overlap
        compute. The psudo count adds fine resolution, especially in
        cases like zero intersection between `x` and `y`. Imagine 2
        situations (x1, y1) and (x2, y2). Their vector lengths are all
        100. `x1` has 10 non-zeros and `y1` has 15 non-zeros; they
        don't overlap. `x2` has 50 non-zeros and `y2` has 55
        non-zeros; they don't overlap either. Now:

        1) If the psudo count is 0, then the overlaps are 0 for both
        situations.

        2) If the psudo count is 1, then the overlaps are `1/(25+1)`
        and `1/(105+1)`, respectively.

        The 2nd situation should be considered more mutually exclusive
        than the 1st, because it is more difficult to avoid
        overlap by chance when they have more nonzeros in the array.

    noise_scale : float
        the scale of a tiny noise added to the statistic to avoid
        discrete effect of overlap computation. This makes the
        p-values for a random table are evenly distributed across (0,
        1] interval.

    seed : int or None
        random seed for random permutation. `None` to not set it.

    Returns
    -------
    float
        the degree of overlap. Its value is in the interval (0, 1].

    '''
    rng = np.random.default_rng(seed)
    intersect = np.sum(x & y)
    union = np.sum(x | y)
    return (intersect + psudo) / (union + psudo) + rng.normal(loc=0, scale=noise_scale)


def compute_cooccur(m, method='binary', cpus=1, progress=False, normcheck=True, **kwargs):
    '''Compute co-occurrence matrix.

    Parameters
    ----------
    m : 2-D array
        sample in row and feature in column.
    method : str, {'binary', 'rank', 'logratio'}
        which co-coccur method to run.
    nonzero : bool
        discard zero values. Only compute the co-occurence for the
        parts that both x and y are non-zeros. This is only used by
        `rank` or `logratio` method.
    cpus : int
        number of CPU cores to use.
    progress : bool
        whether to show progress bar.
    normcheck : bool
        whether to check normality of co-occurences computed from permutation.
        It uses Shapiro Wilk test, which has higher power (ie require less permutations)
        than Kolmogorov Smirnov test and Lilliefors test.
    kwargs : dict
        keywords argument passing to the co-occur method.

    Returns
    -------
    square 2-D array
        Its top right triangle contains the zscores of the cooccur
        statistic computed for each pair of features.

    sqaure 2-D array
        It contains the p-values for positive co-occurrence test in
        the top right triangle and p-values for negative co-occurrence
        test (mutual exclusivity) in the bottom left triangle.

    '''
    if issparse(m):
        m = m.todense().A

    n = m.shape[1]
    logger.debug('%d features in the table' % n)

    pbar = ((i, j) for i in range(n) for j in range(i+1, n))
    if progress:
        from tqdm import tqdm
        pbar = tqdm(pbar, mininterval=1, total=int(n*(n-1)/2))

    res = np.full([n, n], np.nan)
    pval = np.full([n, n], np.nan)

    # 1. For the rank method:
    if method == 'rank':
        # rank and slice (in the for loop) happen on different axes,
        # so rank has to be done here separately.

        # rank across columns (within sample); set ascending to False
        # so that different numbers of zeros in the samples doesn't
        # impact the ranks of nonzero values.
        m = rank(m, axis=1, ascending=False, method='max')

    # 2. For the other 2 methods:
    func = {'binary': cooccur_binary,
            'rank': cooccur_rank,
            'logratio': cooccur_logratio}
    try:
        method = func[method]
    except ValueError as e:
        logger.warning('Method %s is not available.' % method )

    mask = np.full(m.shape, True)
    if kwargs['nonzero']:
        mask[mask == 0] = False

    def helper(i, j):
        nonzeros = mask[:, i] & mask[:, j]
        x, y = m[:, i][nonzeros], m[:, j][nonzeros]
        try:
            v, p_pos, p_neg, dist = method(x, y, **kwargs)
        except ValueError as e:
            logger.debug('feature indices {0} and {1}: {2}'.format(i, j, e))
            v = p_pos = p_neg = np.nan
        zscore = (v - np.mean(dist)) / np.std(dist)
        res[i, j] = zscore
        pval[i, j] = p_pos
        pval[j, i] = p_neg
        if normcheck:
            stat, pval = shapiro(dist)
            logger.debug('Shapiro test for indices {0} and {1}: statstic {2}, p-val {3}'.format(i, j, stat, pval))

    with Parallel(n_jobs=cpus) as parallel:
        parallel(delayed(helper)(i, j) for i, j in pbar)

    return res, pval
