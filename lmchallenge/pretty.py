# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Utility for pretty-printing the model performance from a ``wp``,
``tc``, or ``ic`` log file.
'''

import click
import io
import itertools as it
from . import stats
from .core import common


# Utilities

class AnsiRender:
    '''A helper for rendering in ANSI color codes'''

    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7
    DEFAULT = 9

    def __init__(self, outf):
        self.f = outf
        self.index = self.DEFAULT
        self.bold = False

    def default(self):
        self.color(self.DEFAULT, False)

    def color(self, index, bold):
        if self.bold and not bold:
            self.f.write(u'\x1b[0;%dm' % (30 + index))
        elif bold and not self.bold:
            self.f.write(u'\x1b[1;%dm' % (30 + index))
        elif self.index != index:
            self.f.write(u'\x1b[%dm' % (30 + index))
        self.index = index
        self.bold = bold

    def write(self, s):
        self.f.write(s)

    def close(self):
        self.f.close()


# Rendering

def render_log(data, render_token):
    '''Render an LMC log to a colourized line-based output.

    data -- a sequence of data from an LMC log (e.g. from read_jsonlines)

    render_token -- callable(datum, AnsiRender) to render a single token
                    to the output AnsiRender

    returns -- a generator of ANSI-formatted lines
    '''
    for _, msg_data in it.groupby(
            data, lambda d: (d.get('user'), d.get('message'))):
        with io.StringIO() as f:
            out = AnsiRender(f)
            c = None
            for datum in msg_data:
                if c is not None and c < datum['character']:
                    out.write(' ')
                c = datum['character'] + len(datum['target'])
                render_token(datum, out)
            out.default()
            yield f.getvalue()


class RenderFiltered:
    '''Base class for rendering filtered tokens.
    '''
    def __init__(self, filter):
        self._filter = filter

    def __call__(self, datum, out):
        if self._filter(datum['target']):
            self._render(datum, out)
        else:
            out.color(AnsiRender.BLUE, False)
            out.write(datum['target'])

    def _render(self, datum, out):
        '''Called to render an unfiltered token to "out".
        '''
        raise NotImplementedError


class RenderCompletion(RenderFiltered):
    '''Pretty-print a token to show next-word-prediction/completion.

    If the token is next-word predicted, the entire token is green (and
    bold if it is top-prediction).

    Otherwise, characters that must be typed before the token is predicted
    @2 are coloured red, and completed characters are yellow.
    '''
    def _render(self, datum, out):
        ranks = [common.rank(cs, datum['target'][i:]) or float('inf')
                 for i, cs in enumerate(datum['completions'])]

        # First check for prediction
        if len(ranks) == 0:
            # Should not happen
            out.default()
            out.write(datum['target'])
        elif ranks[0] <= 1:
            out.color(AnsiRender.GREEN, True)
            out.write(datum['target'])
        elif ranks[0] <= 3:
            out.color(AnsiRender.GREEN, False)
            out.write(datum['target'])
        else:
            # Then completion
            typed = next((i for i, r in enumerate(ranks) if r <= 2),
                         len(datum['target']))
            out.color(AnsiRender.RED, False)
            out.write(datum['target'][:typed])
            out.color(AnsiRender.YELLOW, False)
            out.write(datum['target'][typed:])


class RenderEntropy(RenderFiltered):
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
    def __init__(self, filter, interval):
        super().__init__(filter)
        self._interval = interval

    def _render(self, datum, out):
        logp = datum.get('logp')
        x = self._interval / 5
        if logp is None:
            out.color(AnsiRender.MAGENTA, False)
        elif -logp < x:
            out.color(AnsiRender.GREEN, True)
        elif -logp < 2 * x:
            out.color(AnsiRender.GREEN, False)
        elif -logp < 3 * x:
            out.color(AnsiRender.YELLOW, False)
        elif -logp < 4 * x:
            out.color(AnsiRender.RED, False)
        else:
            out.color(AnsiRender.RED, True)
        out.write(datum['target'])


class RenderReranking(RenderFiltered):
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
    def __init__(self, filter, model):
        super().__init__(filter)
        self._model = model

    @staticmethod
    def is_correct(target, results, model):
        scores = ((t, model(e, lm))
                  for t, e, lm in results)
        target_score = next(score for t, score in scores if t == target)
        return all(score < target_score
                   for t, score in scores if t != target)

    def _render(self, datum, out):
        target = datum['target']
        results = datum['results']
        pre = self.is_correct(target, results, lambda e, lm: e)
        post = self.is_correct(target, results, self._model)
        if (pre, post) == (False, False):
            # unchanged incorrect
            out.color(AnsiRender.YELLOW, False)
        elif (pre, post) == (False, True):
            # corrected
            out.color(AnsiRender.GREEN, False)
        elif (pre, post) == (True, False):
            # miscorrected
            out.color(AnsiRender.RED, False)
        else:
            # unchanged correct
            out.default()
        out.write(datum['target'])


class Challenge(common.ParamChoice):
    '''Select an analysis to run on a generated log.
    '''
    name = 'challenge'
    choices = ['auto', 'completion', 'entropy', 'reranking']

    @classmethod
    def auto(cls, data, **args):
        first, data = common.peek(data)

        is_completion = 'completions' in first
        is_entropy = 'logp' in first
        is_reranking = 'results' in first
        if sum([is_completion, is_entropy, is_reranking]) != 1:
            raise Exception('Cannot infer log type from data')

        if is_completion:
            return cls.completion(data, **args)
        elif is_entropy:
            return cls.entropy(data, **args)
        elif is_reranking:
            return cls.reranking(data, **args)

    @staticmethod
    def completion(data, filter, **args):
        return render_log(
            data, RenderCompletion(filter=filter))

    @staticmethod
    def entropy(data, filter, entropy_interval, **args):
        return render_log(
            data, RenderEntropy(filter=filter, interval=entropy_interval))

    @staticmethod
    def reranking(data, filter, **args):
        data = list(data)  # have to traverse 'data' twice
        model = stats.Reranking.build_model(data, target_filter=filter)
        return render_log(
            data, RenderReranking(filter=filter, model=model))


@click.command()
@click.argument('log', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-v', '--verbose', default=0, count=True,
              help='How much human-readable detail to print to STDERR.')
@click.option('-c', '--challenge', type=Challenge(),
              default='auto',
              help='Select which challenge to view (in the case where there'
              ' are multiple challenges in a single log)')
@click.option('-f', '--filter', type=common.TokenFilter(),
              default='alphaemoji',
              help='Only style tokens which match this filter.')
@click.option('-i', '--entropy_interval', default=10.0,
              help='Interval to show entropy differences over')
def cli(log, verbose, challenge, **args):
    '''Pretty-print results from an LMChallenge game (using ANSI color codes).
    '''
    common.verbosity(verbose)

    if len(log) == 0:
        single_log = '-'
    elif len(log) == 1:
        single_log = log[0]
    else:
        raise click.ArgumentError('Can only process zero or one log files')

    for line in challenge(common.read_jsonlines(single_log), **args):
        print(line)


__doc__ += common.shell_docstring(cli, 'lmc pretty')
if __name__ == '__main__':
    cli()
