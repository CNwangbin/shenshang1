# -*- coding: utf-8 -*-

"""Console script for shenshang."""
import sys
import os
from functools import partial
from logging import getLogger

import click
import pandas as pd

from .cooccur import compute_cooccur
from .util import convert_format


click.option = partial(click.option, show_default=True)


def decorator_composer(*decorators):
    def deco(f):
        for dec in reversed(decorators):
            f = dec(f)
        return f
    return deco


_common_parameters = decorator_composer(
    click.option('-i', '--input', type=click.Path(exists=True), required=True, help='input sample-by-feature table'),
    click.option('-f', '--format', type=click.Choice(['biom', 'tsv']), default='biom', help='input table format. For the tsv table, the features are in row and samples are in column; columns are separated by TABs; the 1st column must be feature IDs and the 1st row must be sample IDs; lines starting with `#` will be ignored.'),
    click.option('-o', '--output', type=click.STRING, required=True, help='output file prefix. There are 2 output files: 1. co-occur matrix stores the statistics/correlation coefficients; 2. p-value matrix stores the p-values.'),
    click.option('--force', is_flag=True, help='overwrite output file'),
    click.option('-p', '--progress', is_flag=True, help='show progress bar'),
    click.option('-v', '--verbose', count=True, help='Verbosity. Use multiple `v` (eg -vv) to increase verbosity'))


@click.group(context_settings=dict(
    help_option_names=['-h', '--help'],
    # allow case insensitivity for the (sub)commands and their options
    token_normalize_func=lambda x: x.lower()))
def main():
    """Compute co-occurrence or mutual exclusivity between features."""


@main.command()
@_common_parameters
@click.option('--cutoff', type=click.INT, default=2, help='cutoff to determine presence and absence')
@click.option('--psudo', type=click.IntRange(1), default=1, help='psudo number added to nominator and denominator for overlap computation')
@click.option('--perms', type=click.INT, default=1000, help='the number of permutations to compute p-value')
@click.option('-c', '--cpus', type=click.INT, default=1, help='number of CPU cores to use')
@click.option('--seed', type=click.INT, default=0, help='random seed')
def binary(**kwargs):
    '''compute overlap/mutual exclusivity between features.

    This only considers presence and absence of features in the samples.'''
    compute_cooccur_workflow(method='binary', **kwargs)


@main.command()
@_common_parameters
@click.option('--clip', type=click.FLOAT, default=2, help='Clip (limit) the lower bound of feature values')
@click.option('--perms', type=click.INT, default=1000, help='the number of permutations to compute p-value')
@click.option('-c', '--cpus', type=click.INT, default=1, help='number of CPU cores to use')
@click.option('--seed', type=click.INT, default=0, help='random seed')
def logratio(**kwargs):
    '''compute correlation between features using log ratio.

    This is robust to compositionality effects.'''
    compute_cooccur_workflow(method='logratio', **kwargs)


@main.command()
@_common_parameters
@click.option('--clip', type=click.FLOAT, default=2, help='Clip (limit) the lower bound of feature values')
@click.option('--nonzero', is_flag=True, help='discard the values that are below clip in both features.')
def rank(**kwargs):
    '''compute non-parametric correlation between features.

    This ranks features within each sample first and then compute correlation between features'''
    compute_cooccur_workflow(method='rank', **kwargs)


@main.command()
@click.option('-s', '--stat-file', type=click.Path(exists=True), required=True, help='input file')
@click.option('-p', '--pval-file', type=click.Path(exists=True), required=True, help='input file')
@click.option('-o', '--output', type=click.File('w'), help='output file')
@click.option('-c', '--cutoff', type=click.FLOAT, default=0.01, help='p-value cutoff')
def convert(stat_file, pval_file, output, cutoff):
    '''convert file format to edge file format.'''
    if output is None:
        output = os.path.splitext(stat_file)[0] + '.tsv'
    pval = pd.read_csv(pval_file, sep='\t', index_col=0)
    stat = pd.read_csv(stat_file, sep='\t', index_col=0)
    df = convert_format(stat, pval, cutoff)
    df.to_csv(output, sep='\t', index=False)


def compute_cooccur_workflow(
        input, output,
        format='biom', method='binary',
        cpus=1, progress=True, force=False, verbose=1,
        **kwargs):
    '''
    Parameters
    ----------
    input : str
        input file path
    output : str
        output file prefix
    format : str
        input file format
    method : str
        co-occur method
    cpus : int
        CPU cores
    progress : bool
        whether to show progress bar
    force : bool
        whether to overwrite if output files exist
    verbose : int
        verbosity level
    kwargs : dict
        keyword argument passing to co-occur method

    Returns
    -------
    None
    '''
    levels = ['WARNING', 'INFO', 'DEBUG']
    n = len(levels)
    if verbose >= n:
        verbose = n - 1
    logger = getLogger()
    logger.setLevel(levels[verbose])

    o1 = output + '.stat'
    o2 = output + '.pval'
    if not force and os.path.exists(o1):
        raise FileExistsError('%s already exists' % output)
    if format == 'biom':
        from biom import load_table
        table = load_table(input)
        data = table.matrix_data.transpose()
        ids = table.ids(axis='observation')
    elif format == 'tsv':
        df = pd.read_csv(input, sep='\t', index_col=0, comment='#')
        data = df.values.T
        ids = df.index
    logger.debug('run method: %s' % method)
    stat, pval = compute_cooccur(data, method, cpus, progress, **kwargs)
    stat_df = pd.DataFrame(stat, columns=ids, index=ids)
    pval_df = pd.DataFrame(pval, columns=ids, index=ids)
    stat_df.to_csv(o1, sep='\t')
    pval_df.to_csv(o2, sep='\t')
    return stat_df, pval_df


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
