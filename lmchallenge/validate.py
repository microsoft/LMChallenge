# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Check that logs confirm to the correct LM Challenge schema (expressed as a
JSON schema - see [http://json-schema.org](http://json-schema.org)).

This can be used to check that a method of generating log files other than
`lmchallenge.run` (e.g. using your own parallel execution framework) is
compatible with the LM Challenge analysis tools.

N.B. This validator can only check the format of the log, not the _fairness_ of
the log. An example of an unfair log is a word entropy log where the total
probability over the specified vocabulary for a given context is not equal to
one.
'''

import click
import json
import jsonschema
import os
from .core import common


def schema():
    '''Returns the log instance schema as a Python object.

    (Loaded from the schema definition file within the `lmchallenge` package.)
    '''
    with open(os.path.join(os.path.dirname(__file__), 'log.schema')) as f:
        return json.load(f)


def validate(data):
    '''Check that a loaded log conforms to the schema, using `jsonschema`.

    `data` -- iterable of log events, each of which should conform to
    `lmchallenge.validate.schema`

    `raises` -- `jsonschema.exceptions.ValidationError` if the log does
    not conform
    '''
    log_schema = schema()
    for datum in data:
        jsonschema.validate(datum, log_schema)


@click.command()
@click.argument('log', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-v', '--verbose', default=0, count=True,
              help='How much human-readable detail to print to STDERR.')
def cli(log, verbose):
    '''Validate a log file against the standard LM Challenge schema.
    '''
    common.verbosity(verbose)

    log = log or ['-']

    for single_log in log:
        validate(common.load_jsonlines(single_log))


__doc__ += common.shell_docstring(cli, 'lmc validate')
if __name__ == '__main__':
    cli()
