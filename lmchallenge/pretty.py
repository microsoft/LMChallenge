# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Utility for pretty-printing the model performance from a ``wp``,
``tc``, or ``ic`` log file.
'''

import errno
import click
import io
import collections
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


# Specific challenge programs

class Wp:
    @staticmethod
    def target_rendering(token_info):
        '''Get a string rendering of a token_info (which may be redacted
        as *****).
        '''
        return token_info.get('target') or ('*' * token_info['targetChars'])

    @staticmethod
    def best_rank(ranks, rank_when_missing=None):
        try:
            return min(r for r in ranks if r is not None)
        except ValueError:
            # min of empty sequence
            return rank_when_missing

    def __init__(self, filter):
        self.filter = filter

    def render_pretty(self, row, out):
        '''Render a word-by-word next-word-prediction display
        - top prediction: bold green
        - 2nd/3rd prediction: green
        - any other prediction: yellow
        - not predicted: red
        '''
        for word_info in row['wordPredictions']:
            word = word_info.get('target')
            rank = word_info.get('rank')
            if word is not None and not self.filter(word):
                out.color(AnsiRender.BLUE, False)
            elif rank is None:
                out.color(AnsiRender.RED, False)
            elif rank < 1:
                out.color(AnsiRender.GREEN, True)
            elif rank < 3:
                out.color(AnsiRender.GREEN, False)
            else:
                out.color(AnsiRender.YELLOW, False)
            out.write(Wp.target_rendering(word_info) + ' ')
        out.default()

    def render_diff(self, baselines, line, out):
        '''Render a comparison of a model's rankings against one or more
        baselines.
        '''
        infos = zip(*(x['wordPredictions'] for x in (line,) + baselines))
        for word_info, *baseline_infos in infos:
            target_render = Wp.target_rendering(word_info)
            baseline_target_render = map(Wp.target_rendering, baseline_infos)
            if not all(bt == target_render for bt in baseline_target_render):
                raise ValueError(
                    'sequence of target terms doesn\'t align between'
                    ' evaluation runs being compared')

            rank = word_info.get('rank', 1e10)
            best_baseline_rank = Wp.best_rank(b.get('rank', 1e10)
                                              for b in baseline_infos)
            bold = ((best_baseline_rank <= 1 and rank > 1) or
                    (rank <= 1 and best_baseline_rank > 1) or
                    (best_baseline_rank <= 3 and rank > 3) or
                    (rank <= 3 and best_baseline_rank > 3))
            target = word_info.get('target')
            if target is not None and not self.filter(target):
                out.color(AnsiRender.BLUE, False)
            elif best_baseline_rank < rank:
                out.color(AnsiRender.RED, bold)
            elif rank < best_baseline_rank:
                out.color(AnsiRender.GREEN, bold)
            else:
                out.default()
            out.write(target_render + ' ')
        out.default()


class Tc:
    CharacterState = collections.namedtuple(
        'CharacterState', ['char', 'rank', 'autopredicted']
    )

    @staticmethod
    def get_character_states(row):
        '''Return a flattened sequence of CharacterStates from a result.
        '''
        for token_info in row['textCompletions']:
            target = token_info.get('target')
            rank = token_info.get('rank')
            auto = False
            for ch in (target or [None] * token_info['targetChars']):
                yield Tc.CharacterState(ch, rank, auto)
                auto = True

    def __init__(self, filter):
        self.filter = filter

    def render_pretty(self, row, out):
        '''Render a colored version of a single model's text completion performance.
        - first predicted character: yellow
        - subsequent predicted characters: green
        - not predicted: red
        - ignored: blue
        '''
        for char, rank, autopredicted in Tc.get_character_states(row):
            if char is not None and not self.filter(char):
                out.color(AnsiRender.BLUE, False)
            elif rank is None:
                out.color(AnsiRender.RED, False)
            elif not autopredicted:
                out.color(AnsiRender.YELLOW, False)
            else:
                out.color(AnsiRender.GREEN, False)
            out.write(char or '*')
        out.default()

    def render_diff(self, baselines, line, out):
        partitions = (Tc.get_character_states(x) for x in (line,) + baselines)
        for (char, rank, autopredicted), *baseline_states in zip(*partitions):
            if not all(state.char == char for state in baseline_states):
                raise ValueError(
                    'sequence of characters doesn\'t align between evaluation'
                    ' runs being compared')
            predicted = rank is not None
            predicted_baseline = any(state.rank is not None
                                     for state in baseline_states)
            if char is not None and not self.filter(char):
                out.color(AnsiRender.BLUE, False)
            elif (not predicted) and (not predicted_baseline):
                out.color(AnsiRender.DEFAULT, False)
            elif predicted and predicted_baseline:
                out.color(AnsiRender.CYAN, False)
            elif predicted and (not predicted_baseline):
                out.color(AnsiRender.GREEN, False)
            elif (not predicted) and predicted_baseline:
                out.color(AnsiRender.RED, False)
            else:
                assert False, 'should-be-unreachable code'
            out.write(char or '*')
        out.default()


class Ic:
    def __init__(self, filter):
        self.filter = filter

    def preprocess(self, data, alpha, oov_penalty):
        '''Compute the target ranking with the given mixing settings,
        and store it back in the log.
        Yields a copy of data, with additional word_info:

        ``rank`` - INT - the rank of the correct word
        '''
        for line in data:
            infos = []
            for word_info in line['inputCorrections']:
                out_info = word_info.copy()
                out_info['rank'] = stats.Ic.rank(
                    word_info, alpha=alpha, oov_penalty=oov_penalty
                )
                infos.append(out_info)
            out_line = line.copy()
            out_line['inputCorrections'] = infos
            yield out_line

    def render_pretty(self, line, out):
        '''Render a word-by-word correction display:

        +-----------+-----------+--------+
        | Before    | After     | Color  |
        +===========+===========+========+
        | Incorrect | Incorrect | Yellow |
        +-----------+-----------+--------+
        | Incorrect | Correct   | Green  |
        +-----------+-----------+--------+
        | Correct   | Incorrect | Red    |
        +-----------+-----------+--------+
        | Correct   | Correct   | White  |
        +-----------+-----------+--------+
        |        (Skip)         | Blue   |
        +-----------+-----------+--------+
        '''
        for word_info in line['inputCorrections']:
            word = word_info.get('target')
            before_after = (
                stats.Ic.before_pass(word_info),
                word_info['rank'] == 1
            )
            if word is not None and not self.filter(word):
                out.color(AnsiRender.BLUE, False)
            elif before_after == (False, False):
                out.color(AnsiRender.YELLOW, False)
            elif before_after == (False, True):
                out.color(AnsiRender.GREEN, False)
            elif before_after == (True, False):
                out.color(AnsiRender.RED, False)
            else:  # (True, True)
                out.color(AnsiRender.DEFAULT, False)
            out.write(Wp.target_rendering(word_info) + ' ')
        out.default()

    def render_diff(self, baselines, line, out):
        '''Render the difference between the best of 'baselines'
        (a tuple of lines) and 'line'.

        Examples are rendered as follows. Where the color is marked with a
        star, and the original was correct, the output is marked bold.

        +-----------+-----------+--------++
        | Baseline  | Actual    | Color   |
        +===========+===========+=========+
        | Incorrect | Incorrect | Yellow* |
        +-----------+-----------+---------+
        | Incorrect | Correct   | Green*  |
        +-----------+-----------+---------+
        | Correct   | Incorrect | Red*    |
        +-----------+-----------+---------+
        | Correct   | Correct   | White   |
        +-----------+-----------+---------+
        |        (Skip)         | Blue    |
        +-----------+-----------+---------+
        '''
        infos = zip(*(x['inputCorrections'] for x in (line,) + baselines))
        for word_info, *baseline_word_infos in infos:
            before = stats.Ic.before_pass(word_info)
            baseline_actual = (
                1 == min(x['rank'] for x in baseline_word_infos),
                1 == word_info['rank']
            )
            word = word_info.get('target')
            if word is not None and not self.filter(word):
                out.color(AnsiRender.BLUE, False)
            elif baseline_actual == (False, False):
                out.color(AnsiRender.YELLOW, before)
            elif baseline_actual == (False, True):
                out.color(AnsiRender.GREEN, before)
            elif baseline_actual == (True, False):
                out.color(AnsiRender.RED, before)
            else:  # (True, True)
                out.color(AnsiRender.DEFAULT, False)
            out.write(Wp.target_rendering(word_info) + ' ')
        out.default()


# Generic pretty printing

def pretty(*data, filter=common.TokenFilter.all, opt_args=None):
    '''Compute and return the pretty-printed difference between ``data`` and
    ``baseline_data`` (if specified, otherwise just the absolute performance of
    ``data``.)

    ``data`` - a list of (1+) lists of log lines
               (e.g. each from ``common.read_jsonlines``)

    ``returns`` - a generator of strings containing ANSI color codes
    '''
    pretty_program = common.autodetect_log(data[0], wp=Wp, tc=Tc, ic=Ic)(
        filter=filter
    )
    if opt_args:
        data = [pretty_program.preprocess(d, **args)
                for d, args in common.zip_special(data, opt_args)]

    for lines in zip(*data):
        out = AnsiRender(io.StringIO())
        if 2 <= len(lines):
            pretty_program.render_diff(tuple(lines[:-1]), lines[-1], out)
        else:
            pretty_program.render_pretty(lines[0], out)
        yield out.f.getvalue()


@click.command()
@click.argument('log', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-f', '--filter', type=common.TokenFilter(),
              default='alphaemoji',
              help='Only count tokens which match this filter.')
@click.option('-a', '--opt-args', type=common.JsonParam(), multiple=True,
              help='Pass these arguments to the pretty program (for ``ic``),'
              ' for example pre-computed using ``lmc ic-opt``.'
              ' If comparing to baselines, we expect multiple arguments - the'
              ' same number as the number of log files passed.')
def cli(log, **args):
    '''Pretty-print results from an LMChallenge game (using ANSI color codes).

    If multiple files are specified, use all but the last as 'baselines',
    then pretty-print the differences against the best of a set of baseline
    results (the best at each individual location is used).

    Otherwise pretty-print the basic quality of results.
    '''
    if len(log) == 0:
        log = ['-']
    try:
        for line in pretty(*map(common.read_jsonlines, log), **args):
            # We don't want click.echo's clever handling of colors - we want to
            # always include them
            print(line)
    except IOError as e:
        if e.errno == errno.EPIPE:
            pass


__doc__ += common.shell_docstring(cli, 'lmc pretty')
if __name__ == '__main__':
    cli()
