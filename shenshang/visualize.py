import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from .cooccur import cooccur_binary, cooccur_logratio
from .util import argsort_arrays


def plot_joint_scatter(x, y, cutoff=0, xlabel='x', ylabel='y',
                       ratio=3, fig=None, grid=None, equal_scale=True, **kwargs):
    '''Plot joint scatter and marginal histograms of x and y.

    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size
        the abundance of the feature across all samples
    cutoff : numeric
        the threshold to determine the absence/presence of the feature
    xlabel, ylabel : str
        the label text
    ratio : numeric
        the width ratio of scatter plot over hist plot
    fig : matplotlib.figure.Figure or None (default)
        Figure object to draw the plot onto. None (default) to create a new figure
    grid : matplotlib.gridspec.GridSpec or None (default)
        GridSpect object to draw the plot onto
    equal_scale : bool
        whether to plot in same x-axis and y-axis view limits
    **kwargs : dict
        keyword arguments passing to :func:`matplotlib.axes.Axes.scatter`

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes, matplotlib.axes.Axes
    '''
    spec = dict(nrows=2, ncols=2,
                width_ratios=[ratio, 1],
                height_ratios=[1, ratio],
                hspace=0, wspace=0)
    if fig is None:
        fig = plt.figure()
        gs = GridSpec(**spec)
    else:
        if grid is None:
            raise ValueError('if `fig` is not None, you must provide value for `grid`.')
        gs = GridSpecFromSubplotSpec(subplot_spec=grid, **spec)

    ax_scatter = fig.add_subplot(gs[2])
    ax_histx = fig.add_subplot(gs[0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[3], sharey=ax_scatter)

    ax_scatter.scatter(x=x, y=y, **kwargs)
    if equal_scale:
        ymin, ymax = ax_scatter.get_ylim()
        xmin, xmax = ax_scatter.get_xlim()
        l, u = min(xmin, ymin), max(xmax, ymax)
        ax_scatter.set(xlim=(l, u), ylim=(l, u))
    ax_scatter.set(xlabel=xlabel, ylabel=ylabel)
    ax_scatter.axvline(cutoff, color='gray', linewidth=1, alpha=0.5)
    ax_scatter.axhline(cutoff, color='gray', linewidth=1, alpha=0.5)

    # ax[0].set_xscale('log')
    # ax[0].set_yscale('log')
    # ax[0].set_xscale('symlog', linthreshx=20)
    # ax[0].set_yscale('symlog', linthreshx=20)
    ax_histx.hist(x, bins=100)
    ax_histx.xaxis.set_visible(False)
    ax_histx.set_ylabel('count')

    ax_histy.hist(y, orientation='horizontal', bins=100)
    ax_histy.yaxis.set_visible(False)
    ax_histy.set_xlabel('count')

    return fig, ax_scatter, ax_histx, ax_histy


def plot(x, y, cutoff, psudo, perms, seed=None,
         xlabel='x', ylabel='y',
         colors='black', title='', ratio=3, height=4, **kwargs):
    '''Plot histogram of distribution of overlap or exclusivity for the permutated features.

    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size
        the abundance of the feature across all samples
    psudo : numeric
        psudo count to add both nominator and denominator for overlap
        compute. The choice of psudo count value should depend on the
        sample size, i.e. the size of `x`.
    cutoff : numeric
        the threshold to determine the absence/presence of the feature
    perms : int
        the number of permutations to do for y.
    seed : int or None (default)
        random seed for random permutation.
    xlabel, ylabel : str
        the label text
    colors : str
        color used for the plot elements
    title : str
        figure title
    ratio : numeric
        the width ratio of hist plot over scatter plot
    height : numeric
        the height of figure
    **kwargs :
        keyword arguments passing to :func:`matplotlib.axes.Axes.scatter`

    Returns
    -------
    matplotlib.figure.Figure
        The Figure object containing the plot
    '''
    v1, p1_pos, p1_neg, dist1 = cooccur_binary(x, y, cutoff=cutoff, psudo=psudo, perms=perms, seed=seed)
    v2, p2_pos, p2_neg, dist2 = cooccur_logratio(x, y, clip=cutoff, perms=perms, seed=seed)

    outer = GridSpec(1, 3, width_ratios=[ratio+1, ratio, ratio])
    gs2 = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], height_ratios=[1, ratio], hspace=0)
    gs3 = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2], height_ratios=[1, ratio], hspace=0)

    fig = plt.figure(figsize=(height / (ratio+1) * (1+ratio*3), height))

    if not title:
        title = 'x:{} y:{}'.format(np.sum(x>=cutoff), np.sum(y>=cutoff))
    fig.suptitle(title)

    plot_joint_scatter(x, y, xlabel=xlabel, ylabel=ylabel,
                       ratio=ratio, fig=fig, grid=outer[0],
                       c=colors, s=9, alpha=0.9, **kwargs)

    ax_hist_binary = fig.add_subplot(gs2[1])
    ax_hist_logratio = fig.add_subplot(gs3[1])

    ax_hist_binary.hist(dist1, bins=50)
    ax_hist_binary.set_xlabel('binary')
    ax_hist_binary.set_title('value: %.2f\npval+: %.2f\npval-: %.2f' % (v1, p1_pos, p1_neg))
    ax_hist_binary.axvline(v1, color='red')

    ax_hist_logratio.hist(dist2, bins=50)
    ax_hist_logratio.set_xlabel('logratio')
    ax_hist_logratio.set_title('value: %.2f\npval+: %.2f\npval-: %.2f' % (v2, p2_pos, p2_neg))
    ax_hist_logratio.axvline(v2, color='red')

    fig.tight_layout()

    return fig


def plot_logratio(x, y, clip=1, perms=50, seed=None, color='red', ax=None, **kwargs):
    '''Plot logratio of feature y and feature x in sorted order.

    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size
        the abundance of the feature across all samples.
    clip : numeric
        the threshold to clip the feature values on the lower bound.
    perms : int
        the number of permutations to do for y.
    seed : int or None (default)
        random seed for random permutation.
    color : matplotlib color, optional
        facecolor of the shades.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.
    kwargs : dict
        keyword arguments passing to the :func:`matplotlib.axes.Axes.set` function


    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the plot.
    '''
    if ax is None:
        fig, ax = plt.subplots()
    x, y = np.copy(x), np.copy(y)
    x[x < clip] = clip
    y[y < clip] = clip
    n = x.size
    logratio = np.log(y/x)
    sorted_logratio = np.sort(logratio)
    ax.plot(sorted_logratio, color=color, linewidth=3, alpha=0.7)
    ax.fill_between(np.arange(n), sorted_logratio, facecolor=color, alpha=0.2)
    ax.set(ylabel='logratio log(y/x)', **kwargs)
    if seed is not None:
        np.random.seed(seed)
    for _ in range(perms):
        logratio = np.log(np.random.permutation(y)/x)
        sorted_logratio = np.sort(logratio)
        ax.plot(sorted_logratio, color='grey', linewidth=0.5, alpha=1)
        #ax.fill_between(np.arange(n), sorted_logratio, facecolor='gainsboro', alpha=0.1)
    return ax


def plot_sorted_bars(x, y, ascending=(True, False),
                     xlabel='x', ylabel='y', colors=None, ax=None, **kwargs):
    '''Plot the abundance of feature x and y in sorted order.

    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size
        the abundance of the feature across all samples.
    xlabel, ylabel : str
        the label text
    colors : tuple of (str, str) or None (optional)
        The colors of the bar faces.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.
    **kwargs :
        keyword arguments passing to :func:`matplotlib.axes.Axes.bar`

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the plot.
    '''
    if ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        colors = plt.get_cmap('Set1').colors[:2]

    idx = argsort_arrays((x, y), ascending)
    ind = np.arange(x.size)
    ax.bar(ind, x[idx], color=colors[0], label=xlabel, **kwargs)
    ax.axhline(0, linewidth=2, color='black')
    ax.bar(ind, -y[idx], color=colors[1], label=ylabel, **kwargs)
    ax.set_xlabel('samples')
    ax.set_ylabel('abundance')
    ax.legend()
    return ax


def plot_sorted_shades(x, y, ascending=(True, False),
                       xlabel='x', ylabel='y', colors=None, ax=None, **kwargs):
    '''Sort x and y in the opposite direction and plot.

    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size
        the abundance of the feature across all samples.
    xlabel, ylabel : str
        the label text
    colors : tuple of (str, str) or None (optional)
        The colors filling in the shades.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.
    kwargs : dict
        keyword arguments passing to the :func:`matplotlib.axes.Axes.set` function


    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the plot.
    '''
    if ax is None:
        fig, ax = plt.subplots()
    if colors is None:
        colors = plt.get_cmap('Set1').colors[:2]

    ind = np.arange(x.size)
    idx = argsort_arrays((x, y), ascending)
    x, y = x[idx], y[idx]
    ax.fill_between(ind, x, color=colors[0], alpha=0.3)
    ax.plot(ind, x, color=colors[0], alpha=0.6, label=xlabel)
    ax.fill_between(ind, y, color=colors[1], alpha=0.3)
    ax.plot(ind, y, color=colors[1], alpha=0.6, label=ylabel)
    ax.set(xlabel='samples', ylabel='abundance', **kwargs)
    ax.legend()
    return ax


def plot_confusion_matrix(x, y, cutoff,
                          xlabel='x', ylabel='y', ratio=True, mask=True, cmap=plt.cm.Blues,
                          ax=None, **kwargs):
    '''Plot confusion matrix of absence-presence for x and y.

    Parameters
    ----------
    x, y : 1-D numeric arrays of the same size
        the abundance of the feature across all samples
    cutoff : numeric
        the threshold to determine the absence/presence of the feature
    xlabel, ylabel : str
        the label text
    ratio : bool
        whether to plot the ratios (the counts in confusion matrix / the counts
        of features present in both samples) or the counts.
    mask : bool
        whether to mask (not show) the absence-absence cell in the heatmap
    ax : matplotlib.axes.Axes or None (default), optional
        Axes object to draw the plot onto; otherwise uses the current Axes
    cmap : string, optional
        matplotlib qualitative colormap
    kwargs : dict
        keyword arguments passing to the :func:`matplotlib.axes.Axes.set` function

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object containing the plot
    '''
    x_bool = np.asarray(x) >= cutoff
    y_bool = np.asarray(y) >= cutoff
    pp = np.sum(x_bool & y_bool) # present in both x and y
    aa = x_bool.size - np.sum(x_bool | y_bool) # absent in both x and y
    pa = np.sum(x_bool) - pp # present in x, absent in y
    ap = np.sum(y_bool) - pp # present in y, absent in x
    # mask out absence-absence because we are not interested in it.
    data = np.ma.array([[aa, pa], [ap, pp]],
                       mask=[[mask, 0], [0, 0]])
    if ratio and pp != 0:
        data = data / pp
        fmt = '.2f'
    else:
        fmt = 'd'
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(data, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.patch.set(hatch='x', edgecolor='lightgray')
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Absent', 'Present'],
           yticklabels=['Absent', 'Present'],
           xlabel=xlabel,
           ylabel=ylabel,
           **kwargs)
    thresh = (data.max() - data.min()) / 2. + data.min()
    for i, j in zip(*np.where(~data.mask)):
        ax.text(i, j, format(data[j, i], fmt),
                ha="center", va="center",
                color="white" if data[j, i] > thresh else "black")
    return ax
