# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''LM Challenge - language modelling evaluation suite.
'''

import click
from . import run, stats, pretty, ic_opt, page


@click.group()
def cli():
    '''The main entry point to LMChallenge.
    Use subcommands to perform specific tasks.
    '''
    pass


cli.add_command(run.cli, 'run')
cli.add_command(stats.cli, 'stats')
cli.add_command(pretty.cli, 'pretty')
cli.add_command(ic_opt.cli, 'ic-opt')
cli.add_command(page.cli, 'page')
