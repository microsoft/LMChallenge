# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''LM Challenge - language modelling evaluation suite.
'''

from .core.model import FilteringWordModel, Model, WordModel  # NOQA
from .core.common import WORD_TOKENIZER, CHARACTER_TOKENIZER  # NOQA

import click
from . import diff, grep, pretty, run, stats, validate


@click.group()
def cli():
    '''The main entry point to LMChallenge.
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
