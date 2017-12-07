# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Utility for rendering an interactive HTML results page from ``wr``
(``{wc, we, ce}`` to follow).
'''

import click
import os
import string
import tempfile
import logging
import urllib.request
import hashlib
import json
from io import StringIO
from .core import common
from . import stats


def _read_data_file(name):
    '''Read a file from the bundled 'lmchallenge/data' directory, and return
    the contents as a string.
    '''
    with open(os.path.join(os.path.dirname(__file__), 'data', name)) as f:
        return f.read()


def _download_cache_cdn(url, sha_384):
    '''Download a file from 'url', which should have the SHA384 matching
    'sha_384' (which should be a hex string).
    '''
    root = os.path.join(tempfile.gettempdir(), 'lmc_page')
    if not os.path.isdir(root):
        os.makedirs(root)

    target = os.path.join(root, sha_384)
    if not os.path.isfile(target):
        logging.info('Downloading %s -> %s', url, target)
        urllib.request.urlretrieve(url, target)
        with open(target, 'rb') as f:
            h = hashlib.sha384()
            h.update(f.read())
            if h.hexdigest() != sha_384:
                logging.error('Checksum mismatch between %s, %s',
                              url, target)
                raise IOError('Checksum mismatch between %s, %s:'
                              ' expected %s actual %s',
                              url, target, h.hexdigest(), sha_384)

    with open(target) as f:
        return f.read()


def _get_files():
    '''Returns a dictionary of {KEY: DATA} for all the supplementary js & css
    data files needed to render the standalone html page.
    '''
    # Note: to get the checksums:
    #   wget https://URL -O - | sha384sum
    return dict(
        PAGE=_read_data_file('page.html'),
        LMC_CSS=_read_data_file('lmc.css'),
        LMC_JS=_read_data_file('lmc.js'),
        BOOTSTRAP_CSS=_download_cache_cdn(
            'https://'
            'maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css',
            'd6af264c93804b1f23d40bbe6b95835673e2da59057f0c04'
            '01af210c3763665a4b7a0c618d5304d5f82358f1a6933b3b'
        ),
        BOOTSTRAP_JS=_download_cache_cdn(
            'https://'
            'maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js',
            'd2649b24310789a95f9ae04140fe80e10ae9aeae4e55f5b7'
            'ecf451de3e442eac6cb35c95a8eb677a99c754ff5a27bc52'
        ),
        JQUERY_JS=_download_cache_cdn(
            'https://code.jquery.com/jquery-2.2.4.min.js',
            'ad8fe3bfc98c86a0da6d74a8f940a082a2ad76605f777a82'
            'dbf2afc930cd43a3dc5095dac4ad6d31ea6841d6b8839bc1'
        ),
        D3_JS=_download_cache_cdn(
            'https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.17/d3.min.js',
            '37c10fd189a5d2337b7b40dc5e567aaedfa2a8a53d0a4e9f'
            'd5943e8f6a6ec5ab6706ae24f44f10eafa81718df82cd6e7'
        ),
    )


def _json_dumps_min(data, float_format=''):
    '''Tiny JSON serializer that supports strings, ints, floats, tuples, lists
    and dictionaries.

    Compared to json.dumps, allows a format to specified for floating point
    values.
    '''
    out = StringIO()

    def visit(node):
        if node is None:
            out.write('null')
        elif isinstance(node, str):
            out.write(json.dumps(node))
        elif isinstance(node, bool):
            out.write('true' if node else 'false')
        elif isinstance(node, int):
            out.write(str(node))
        elif isinstance(node, float):
            out.write(format(node, float_format))
        elif isinstance(node, (tuple, list)):
            out.write('[')
            for i, x in enumerate(node):
                if i != 0:
                    out.write(',')
                visit(x)
            out.write(']')
        elif isinstance(node, dict):
            out.write('{')
            for i, k in enumerate(node):
                if i != 0:
                    out.write(',')
                visit(k)
                out.write(':')
                visit(node[k])
            out.write('}')
        else:
            raise ValueError(
                'Unexpected value for JSON conversion: {}'.format(node))
    visit(data)
    return out.getvalue()


def _log_select(data, target_filter):
    '''Add the "select" keyword to the log, to select specific words to show.
    '''
    for datum in data:
        yield dict(select=target_filter(datum['target']), **datum)


def _log_combined_score(data, model):
    '''Add the combined score to the word reranking results in the log, and
    sort descending score.
    '''
    for datum in data:
        datum = datum.copy()
        datum['results'] = list(sorted(
            ((candidate, error_score, lm_score, model(error_score, lm_score))
             for candidate, error_score, lm_score in datum['results']),
            key=lambda x: -x[-1]
        ))
        yield datum


@click.command()
@click.argument('log', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-v', '--verbose', default=0, count=True,
              help='How much human-readable detail to print to STDERR.')
@click.option('-f', '--filter', type=common.TokenFilter(),
              default='alphaemoji',
              help='Only style tokens which match this filter.')
@click.option('-m', '--float-fmt', default='.4g',
              help='The format of floats in the JSON file (use compact'
              ' representations to save file size).')
def cli(log, verbose, filter, float_fmt):
    '''Create an HTML page rendering of a log file.

    Useful for investigating prediction issues. Currently only 'wr' is
    supported.
    '''
    common.verbosity(verbose)

    if len(log) == 0:
        log = '-'
    elif len(log) == 1:
        log = log[0]
    else:
        raise click.ClickException('Cannot handle multiple log files.')

    # Create a snippet to substitute in, containing the data to be examined
    # list(), as have to traverse multiple times
    data = list(_log_select(common.read_jsonlines(log), filter))

    if 'results' in data[0]:
        # Word Reranking
        model = stats.Reranking.build_model(data, target_filter=filter)
        wr_data = list(_log_combined_score(data, model))
        set_data = string.Template(
            'setup_wr(${WR_DATA}, "${WR_MODEL}");'
        ).substitute(
            WR_DATA=_json_dumps_min(wr_data, float_format=float_fmt),
            WR_MODEL=str(model),
        )
    elif 'logp' in data[0]:
        set_data = string.Template(
            'setup_entropy(${DATA});'
        ).substitute(
            DATA=_json_dumps_min(data)
        )
    elif 'completions' in data[0]:
        set_data = string.Template(
            'setup_wc(${DATA});'
        ).substitute(
            DATA=_json_dumps_min(data)
        )

    # Render the HTML file, with all dependencies inlined
    files = _get_files()
    print(string.Template(files['PAGE']).substitute(
        LMC_SETUP=set_data, **files
    ))


__doc__ += common.shell_docstring(cli, 'lmc page')
if __name__ == '__main__':
    cli()
