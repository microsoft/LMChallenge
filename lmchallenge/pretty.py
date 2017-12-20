# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Pretty-print the model performance from an LM Challenge log file,
in ANSI colour or HTML format.
'''

import click
import io
import os
import tempfile
import logging
import urllib.request
import hashlib
import json
import string
import itertools as it
from . import stats
from .core import common


# Rendering utilities

class Renderer:
    '''Base class for rendering selected/unselected tokens.
    '''
    def __call__(self, datum, out):
        '''Called to render a token (which has not been "deselected") to the
        AnsiRender instance "out".

        `datum` -- `dict` -- log datum to render

        `out` -- `lmchallenge.core.common.AnsiRender` -- output of rendering
        '''
        raise NotImplementedError

    def html_setup(self, data, float_format):
        '''The setup command for a standalone HTML page with the given data.
        '''
        raise NotImplementedError


class RenderCompletion(Renderer):
    '''Pretty-print a token to show next-word-prediction/completion.

    If the token is next-word predicted, the entire token is green (and
    bold if it is top-prediction). Otherwise, characters that must be typed
    before the token is predicted @2 are coloured red, and completed
    characters are yellow:

    +---------------------------+--------------+
    | Case                      | Color        |
    +===========================+==============+
    | Next word prediction @1   | Bold Green   |
    | Next word prediction @3   | Green        |
    | Unpredicted characters @2 | Red          |
    | Predicted characters @2   | Black (Grey) |
    +---------------------------+--------------+
    '''
    def __call__(self, datum, out):
        ranks = [common.rank(cs, datum['target'][i:]) or float('inf')
                 for i, cs in enumerate(datum['completions'])]

        # First check for prediction
        if len(ranks) == 0:
            # Should not happen
            out.default()
            out.write(datum['target'])
        elif ranks[0] <= 1:
            out.color(out.GREEN, True)
            out.write(datum['target'])
        elif ranks[0] <= 3:
            out.color(out.GREEN, False)
            out.write(datum['target'])
        else:
            # Then completion
            typed = next((i for i, r in enumerate(ranks) if r <= 2),
                         len(datum['target']))
            out.color(out.RED, False)
            out.write(datum['target'][:typed])
            out.color(out.BLACK, False)
            out.write(datum['target'][typed:])

    def html_setup(self, data, float_format):
        return string.Template(
            'setup_wc(${DATA});'
        ).substitute(DATA=_json_dumps_min(data, float_format=float_format))


class RenderEntropy(Renderer):
    '''Pretty-print a token to show entropy contribution.

        +-------------+------------+
        | Entropy     | Color      |
        +=============+============+
        |    Skip     | Blue       |
        +-------------+------------+
        |   Unknown   | Magenta    |
        +-------------+------------+
        |   0  - i/5  | Bold Green |
        |  i/5 - 2i/5 | Green      |
        | 2i/5 - 3i/5 | Yellow     |
        | 3i/5 - 4i/5 | Red        |
        | 4i/5 - ...  | Bold Red   |
        +-------------+------------+
    '''
    def __init__(self, interval):
        self._interval = interval

    def __call__(self, datum, out):
        logp = datum.get('logp')
        x = self._interval / 5
        if logp is None:
            out.color(out.MAGENTA, False)
        elif -logp < x:
            out.color(out.GREEN, True)
        elif -logp < 2 * x:
            out.color(out.GREEN, False)
        elif -logp < 3 * x:
            out.color(out.YELLOW, False)
        elif -logp < 4 * x:
            out.color(out.RED, False)
        else:
            out.color(out.RED, True)
        out.write(datum['target'])

    def html_setup(self, data, float_format):
        return string.Template(
            'setup_entropy(${DATA}, ${INTERVAL});'
        ).substitute(
            DATA=_json_dumps_min(data, float_format=float_format),
            INTERVAL=self._interval,
        )


class RenderReranking(Renderer):
    '''Pretty-print a token to show correction.

        +-----------+-----------+---------+
        | Before    | After     | Color   |
        +===========+===========+=========+
        |         Skip          | Blue    |
        +-----------+-----------+---------+
        | Incorrect | Incorrect | Yellow  |
        +-----------+-----------+---------+
        | Incorrect | Correct   | Green   |
        +-----------+-----------+---------+
        | Correct   | Incorrect | Red     |
        +-----------+-----------+---------+
        | Correct   | Correct   | White   |
        +-----------+-----------+---------+
    '''
    def __init__(self, model):
        self._model = model

    @staticmethod
    def is_correct(target, results, model):
        scores = ((t, model(e, lm))
                  for t, e, lm in results)
        target_score = next(score for t, score in scores if t == target)
        return all(score < target_score
                   for t, score in scores if t != target)

    def __call__(self, datum, out):
        target = datum['target']
        results = datum['results']
        pre = self.is_correct(target, results, lambda e, lm: e)
        post = self.is_correct(target, results, self._model)
        if (pre, post) == (False, False):
            # unchanged incorrect
            out.color(out.YELLOW, False)
        elif (pre, post) == (False, True):
            # corrected
            out.color(out.GREEN, False)
        elif (pre, post) == (True, False):
            # miscorrected
            out.color(out.RED, False)
        else:
            # unchanged correct
            out.default()
        out.write(datum['target'])

    def html_setup(self, data, float_format):
        data = list(_log_combined_score(data, self._model))
        return string.Template(
            'setup_wr(${DATA}, "${MODEL}");'
        ).substitute(
            DATA=_json_dumps_min(data, float_format=float_format),
            MODEL=str(self._model),
        )


# HTML rendering

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
    root = os.path.join(tempfile.gettempdir(), 'lmc_pretty')
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


def _get_viewer_files():
    '''Returns a dictionary of {KEY: DATA} for all the supplementary js & css
    data files needed to render the standalone html page.
    '''
    # Note: to get the checksums:
    #   wget https://URL -O - | sha384sum
    return dict(
        LMC_VIEWER_HTML=_read_data_file('viewer.html'),
        LMC_VIEWER_CSS=_read_data_file('viewer.css'),
        LMC_VIEWER_JS=_read_data_file('viewer.js'),
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
    out = io.StringIO()

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


# Toplevel rendering functions

def render_ansi(data, renderer):
    '''Render an LMC log to a colourized line-based output.

    `data` -- `generator(dict)` -- LM Challenge log

    `renderer` -- `lmchallenge.pretty.Renderer` -- to render the log

    `return` -- `generator(string)` -- generates ANSI-formatted lines
    '''
    for _, msg_data in it.groupby(
            data, lambda d: (d.get('user'), d.get('message'))):
        with io.StringIO() as f:
            out = common.AnsiRender(f)
            char_n = 0
            for datum in msg_data:
                if char_n < datum['character']:
                    out.write(' ')
                char_n = datum['character'] + len(datum['target'])
                if common.is_selected(datum):
                    renderer(datum, out)
                else:
                    out.color(out.BLUE, bold=False)
                    out.write(datum['target'])
            out.default()
            yield f.getvalue()


def render_html(data, renderer, float_format):
    '''Render an LMC log to a standalone (and mildly interactive) HTML file.

    `data` -- `generator(dict)` -- LM Challenge log

    `renderer` -- `lmchallenge.pretty.Renderer` -- to render the log

    `float_format` -- `string` -- format string for floating point numbers in
                      the resulting HTML file's compact JSON log

    `return` -- `string` -- standalone HTML
    '''
    # Render the HTML file, with all dependencies inlined
    files = _get_viewer_files()
    return string.Template(files['LMC_VIEWER_HTML']).substitute(
        LMC_SETUP=renderer.html_setup(data, float_format), **files
    )


# Script

class ChallengeChoice(common.ChallengeChoice):
    '''Select a pretty printing program.
    '''
    @staticmethod
    def completion(data, **args):
        return RenderCompletion()

    @staticmethod
    def entropy(data, entropy_interval, **args):
        return RenderEntropy(interval=entropy_interval)

    @staticmethod
    def reranking(data, **args):
        return RenderReranking(model=stats.Reranking.build_model(data))


class OutputChoice(common.ParamChoice):
    '''Select an output format.
    '''
    name = 'output'
    choices = ['ansi', 'html']

    @staticmethod
    def ansi(data, renderer, **args):
        for line in render_ansi(data, renderer):
            print(line)

    @staticmethod
    def html(data, renderer, float_format, **args):
        print(render_html(data, renderer, float_format=float_format))


def ansi(data, entropy_interval=10.0):
    '''Render and return an ANSI-formatted pretty-printing of a LM Challenge log.

    `data` -- `iterable(dict)` -- LM Challenge log

    `return` -- `iterable(string)` -- ANSI-coloured rendering of the log
    '''
    data = list(data)
    return render_ansi(
        data,
        ChallengeChoice.auto(
            data, entropy_interval=entropy_interval))


@click.command()
@click.argument('log', nargs=-1, type=click.Path(dir_okay=False))
@click.option('-v', '--verbose', default=0, count=True,
              help='How much human-readable detail to print to STDERR')
@click.option('-c', '--challenge', type=ChallengeChoice(),
              default='auto',
              help='Select which challenge to view (in the case where there'
              ' are multiple challenges in a single log)')
@click.option('-o', '--output', type=OutputChoice(),
              default='ansi',
              help='Select whether to use a simple ANSI format, or an'
              ' all-in-one html page to show results')
@click.option('-i', '--entropy_interval', default=10.0,
              help='Interval to show entropy differences over (should be'
              ' positive)')
@click.option('-m', '--float-format', default='.4g',
              help='The format of floats in the JSON file (use compact'
              ' representations to save file size)')
def cli(log, verbose, challenge, output, entropy_interval, float_format):
    '''Pretty-print results from LM Challenge (using ANSI color codes).
    '''
    common.verbosity(verbose)

    # list() because of multiple traverse (in the case of reranking)
    data = list(common.load_jsonlines(common.single_log(log)))

    renderer = challenge(data, entropy_interval=entropy_interval)

    output(data, renderer, float_format=float_format)


__doc__ += common.shell_docstring(cli, 'lmc pretty')
if __name__ == '__main__':
    cli()
