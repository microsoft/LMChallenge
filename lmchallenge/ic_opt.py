# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Script to optimize the mixing parameters for an ``ic`` log file.
(For the Python API, see ``lmchallenge.stats``.)
'''

import click
import json
from .core import common
from .stats import Ic, Domain


@click.command()
@click.argument('log', type=click.Path(exists=True, dir_okay=False))
@click.option('-v', '--verbose', default=0, count=True,
              help='How much human-readable detail to print to STDERR.')
@click.option('-t', '--target', default='accuracy',
              help='Which metric to optimize for.')
@click.option('-n', '--lines', type=click.INT,
              help='Limit input to this number of lines')
@click.option('-f', '--filter', type=common.TokenFilter(),
              default='alphaemoji',
              help='Only count tokens which match this filter.')
@click.option('-a', '--alpha', type=Domain(),
              default=repr(Ic.DEFAULT_ALPHA),
              help='Grid search domain for alpha (model score mixing weight)')
@click.option('-o', '--oov-penalty', type=Domain(),
              default=repr(Ic.DEFAULT_OOV_PENALTY),
              help='Grid search domain for the oov penalty')
def cli(verbose, log, lines, **args):
    '''Optimize the ic stats, and return the optimal mixing settings,
    as a JSON dictionary.

    Use this with the ``-a`` setting of ``lmc stats`` and ``lmc_pretty`` to
    speed up execution (required for ``lmc_pretty``).
    '''
    common.verbosity(verbose)
    best = Ic.optimize(common.read_jsonlines(log, lines), **args)
    click.echo(json.dumps(best, sort_keys=True))


__doc__ += common.shell_docstring(cli, 'lmc ic-opt')
if __name__ == '__main__':
    cli()
