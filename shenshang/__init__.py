# -*- coding: utf-8 -*-

"""Top-level package for shenshang."""


from logging.config import fileConfig
from io import StringIO
from .cooccur import cooccur_binary, cooccur_logratio
from .cli import compute_cooccur_workflow


__version__ = '0.1.0'
__all__ = ['cooccur_binary', 'cooccur_logratio', 'compute_cooccur_workflow']


# setting False allows other logger to print log.
with StringIO('''
[loggers]
keys=root

[handlers]
keys=consoleHandler

[formatters]
keys=simpleFormatter

[logger_root]
handlers=consoleHandler

[handler_consoleHandler]
class=StreamHandler
formatter=simpleFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format=%(asctime)s %(levelname)-8s %(message)s
datefmt=%Y-%m-%d %H:%M:%S
''') as f:
    fileConfig(f, disable_existing_loggers=False)
