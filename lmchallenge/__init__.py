# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''LM Challenge - language modelling evaluation suite.

A set of tools for evaluating language models. We find these tools useful
for making a fair comparison of pure language models, of very different
kinds (e.g. traditional ngram models vs. deep learning), and implemented
in different programming languages / using different frameworks.

LM Challenge is runnable from the command line `lmc --help`, or from
Python `import lmchallenge as lmc`. This documentation is primarily
for the users from Python, for the command line API and examples,
see the [README on GitHub](https://github.com/Microsoft/LMChallenge).

# Python example

Here is a quick example in Python, defining a custom word model, based
on the class `lmchallenge.FilteringWordModel` (for a more general API to
implement, see `lmchallenge.Model` or `lmchallenge.WordModel`).

    #!python
    >>> import lmchallenge as lmc

    >>> class MyModel(lmc.FilteringWordModel):
    ...     def score_word(self, context, candidates):
    ...         return [(c, -len(c)) for c in candidates]
    ...     def predict_word_iter(self, context):
    ...         return [('one', -1), ('two', -2), ('three', -3)]

    >>> my_model = MyModel(n_predictions=3)

    # This is the core LM Challenge API - 'predict'
    >>> my_model.predict('', None)
    [('one', -1), ('two', -2), ('three', -3)]
    >>> my_model.predict('This is ', ['foo', 'a', 'brilliant'])
    [('foo', -3), ('a', -1), ('brilliant', -9)]

To evaluate this model with LM Challenge, we select a challenge -
for example word completion (`wc`), which measures next-word-prediction
hit rate and word completion statistics.

    #!python
    >>> log = list(lmc.wc(my_model, ['one potato two potato three']))
    >>> [x['target'] for x in log]
    ['one', 'potato', 'two', 'potato', 'three']

We now have a _log_ object, which is the core data type of LM Challenge. Here
 are a few things you can do with logs:

    #!python
    # Compute aggregate stats
    >>> stats = lmc.stats.stats(log)
    >>> stats['prediction']['hit1']
    0.2
    >>> stats['prediction']['hit3']
    0.6
    >>> stats['fingerprint']
    'a33f4773'

    # Pretty-print the log
    >>> pretty = lmc.pretty.ansi(log)
    >>> for line in pretty:
    ...     print(line)                       # doctest: +SKIP
    one potato two potato three               #...except more colourful

    # Filter and compute stats
    >>> f_log = lmc.grep.grep('^t|potato', log)
    >>> f_stats = lmc.stats.stats(f_log)
    >>> f_stats['skipped']
    0.2
    >>> f_stats['prediction']['hit3']
    0.5
    >>> f_stats['fingerprint']                # note: different fingerprint
    '2a1ecfab'

Logs are simply iterables of dictionaries that conform to a log schema, and
are usually stored in JSONlines format:

    #!python
    >>> lmc.dump_jsonlines(log, '/tmp/my_log.jsonl')
    >>> log = list(lmc.load_jsonlines('/tmp/my_log.jsonl'))
    >>> [x['target'] for x in log]
    ['one', 'potato', 'two', 'potato', 'three']


# Command Line Interface example

All of the same features are available on the command line (try
`$ lmc --help`). For a guide and examples, please see the
[README on GitHub](https://github.com/Microsoft/LMChallenge).
'''

# Module

import click
from . import core, diff, grep, pretty, run, stats, validate
from .core.model import FilteringWordModel, Model, WordModel
from .core.common import (
    WORD_TOKENIZER, CHARACTER_TOKENIZER,
    load_jsonlines, dump_jsonlines
)
from .run import wc, we, wr, ce

__all__ = [
    # submodules
    'core',
    'grep',
    'diff',
    'pretty',
    'run',
    'stats',
    'validate',

    # specifics
    'Model',
    'WordModel',
    'FilteringWordModel',
    'WORD_TOKENIZER',
    'CHARACTER_TOKENIZER',
    'dump_jsonlines',
    'load_jsonlines',
    'wc', 'we', 'wr', 'ce',
]


# Command line interface

@click.group()
def cli():
    '''The main entry point to LM Challenge.
    Use subcommands to perform specific tasks.
    '''
    pass


# Changes to this list should be synced with setup.py
cli.add_command(diff.cli, 'diff')
cli.add_command(grep.cli, 'grep')
cli.add_command(pretty.cli, 'pretty')
cli.add_command(run.cli, 'run')
cli.add_command(stats.cli, 'stats')
cli.add_command(validate.cli, 'validate')
