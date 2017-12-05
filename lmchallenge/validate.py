# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Check that logs confirm to the correct LMChallenge schema.
'''

import click
import json
import jsonschema
import os
from .core import common


def schema():
    '''Returns the log instance schema as a Python object, loaded from the schema
    definition file, within the LMChallenge package.
    '''
    with open(os.path.join(os.path.dirname(__file__), 'log.schema')) as f:
        return json.load(f)


@click.command()
@click.argument('log', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def cli(log):
    '''Validate a log file against the standard LMChallenge schema.
    '''
    log = log or ['-']
    log_schema = schema()
    for single_log in log:
        for line in common.read_jsonlines(single_log):
            jsonschema.validate(line, log_schema)


__doc__ += common.shell_docstring(cli, 'lmc validate')
if __name__ == '__main__':
    cli()
