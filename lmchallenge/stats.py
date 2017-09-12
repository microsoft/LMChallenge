'''Aggregate results from ``wp``, ``tc`` or ``ic`` challenges, and calculate
summary statistics.
'''

import itertools
import json
import click
import logging
import copy
from collections import defaultdict
from .core import common, monoid
from .core.common import fdiv_null, read_jsonlines


# Optimization helpers

class Domain(click.ParamType):
    '''Parses the domain of a simple linear grid search.
    '''
    class Linear:
        '''A linear sweep through a floating point parameter.
        '''
        slots = ['min', 'max', 'n']

        def __init__(self, min, max, n):
            self.min = min
            self.max = max
            self.n = n

        def values(self):
            '''Returns a list of values to try for this parameter.
            '''
            if self.n == 1:
                return [float(self.min + self.max) / 2]
            step = float(self.max - self.min) / (self.n - 1)
            return [self.min + step * x for x in range(self.n)]

        def __repr__(self):
            return ':'.join(map(str, (self.min, self.max, self.n)))

    name = 'domain'

    def convert(self, value, param, ctx):
        if ':' in value:
            min, max, n = value.split(':')
            return Domain.Linear(min=float(min), max=float(max), n=int(n))
        else:
            fix = float(value)
            return Domain.Linear(min=fix, max=fix, n=1)

    def get_metavar(self, param):
        return '"MIN:MAX:N" | VALUE'


# Stats programs

class Wp:
    '''
    Monoids that may be used to accumulate stats, e.g. (two alternatives)::

      f = common.TokenFilter.alphaonly
      Wp(f).monoid.reduce(log)
      stats.aggregate_by(log, Wp(f).monoid.reduce, stats.Partition.user)
    '''

    # Sentence position metrics, max index
    MAX_INDEX = 5

    @staticmethod
    def _sparse_update(ranks, idx, default, update):
        '''Apply the update function to index idx in the ranks list, extending
        it as necessary with copies of 'default'.'''
        if len(ranks) <= idx:
            ranks.extend([copy.deepcopy(default)
                          for _ in range(1 + idx - len(ranks))])
        ranks[idx] = update(ranks[idx])

    @staticmethod
    def _sparse_merge(aggregate, ranks, default, merge):
        '''Merge the 'ranks' list into the 'aggregate' list.'''
        if len(aggregate) < len(ranks):
            aggregate.extend(map(copy.deepcopy,
                                 [default] * (len(ranks) - len(aggregate))))
        for idx, value in enumerate(ranks):
            aggregate[idx] = merge(aggregate[idx], value)
        return aggregate

    @staticmethod
    def _get_row_ranks(xs):
        ranks = []
        for word_info in xs:
            rank = word_info.get('rank')
            if rank is not None:
                assert rank != 0, 'Rank should be 1-based'
                Wp._sparse_update(ranks, rank - 1, 0, lambda x: x + 1)
        return ranks

    @staticmethod
    def _get_row_ranks_per_idx(xs, max_idx=MAX_INDEX-1):
        '''For each sentence/paragraph, compute ranks for each index in the
        sentence separately, up to max_idx. After max_idx all ranks are
        assigned the same index'''
        ranks = defaultdict(list)
        for idx, word_info in enumerate(xs):
            rank = word_info.get('rank')
            if rank is not None:
                assert rank != 0, 'Rank should be 1-based'
                Wp._sparse_update(ranks[min(max_idx, idx)],
                                  rank - 1, 0, lambda x: x + 1)
        return ranks

    @staticmethod
    def _entry_filter(target_filter):
        '''Convert a target filter into an entry filter (which copes with
        missing targets).'''
        return lambda entry: (('target' not in entry) or
                              target_filter(entry['target']))

    @staticmethod
    def _filtering(filter, include):
        '''Return a Monoid that does custom token filtering on a list of
        inputs'''
        return monoid.Monoid(
            identity=monoid._merge_dicts(include.identity, {'skip': 0}),
            lift=lambda xs: monoid._merge_dicts(
                include.lift([x for x in xs if filter(x)]),
                {'skip': sum(1 for x in xs if not filter(x))}),
            combine=lambda a, b: monoid._merge_dicts(
                include.combine(a, b), {'skip': a['skip'] + b['skip']})
        )

    @staticmethod
    def _merge_idx_dicts(a, b):
        '''Each of a and b is a dict with index of word in the sentence mapped to
        ranks of words within the input'''
        return {k: Wp._sparse_merge(copy.copy(a.get(k, [])), b.get(k, []),
                                    0, lambda x, y: x + y)
                for k in set(a).union(b)}

    per_idx_ranking = monoid.Monoid(
        identity=defaultdict(list),
        lift=_get_row_ranks_per_idx.__func__,
        combine=_merge_idx_dicts.__func__
    )

    ranking = monoid.Monoid(
        identity=[],
        lift=_get_row_ranks.__func__,
        combine=lambda a, b: Wp._sparse_merge(copy.copy(a), b,
                                              0, lambda x, y: x + y)
    )

    word_counting = monoid.map_input(len, monoid.summing)

    per_idx_counting = monoid.Monoid(
        identity=[],
        lift=lambda p: ([int(i < len(p)) for i in range(Wp.MAX_INDEX-1)] +
                        [sum(1 for i, _ in enumerate(p)
                             if i >= Wp.MAX_INDEX-1)]),
        combine=lambda a, b: Wp._sparse_merge(copy.copy(a), b,
                                              0, lambda x, y: x + y))

    char_counting = monoid.map_input(
        lambda xs: sum(len(x['target']) if 'target' in x else x['targetChars']
                       for x in xs),
        monoid.summing
    )

    miss_counting = monoid.map_input(
        lambda xs: sum(1 for x in xs if 'rank' not in x),
        monoid.summing
    )

    stats = monoid.keys(
        ranks=ranking,
        words=word_counting,
        chars=char_counting,
        miss=miss_counting,
        idx_ranks=per_idx_ranking,
        idx_counts=per_idx_counting,
    )

    def __init__(self, filter):
        self.monoid = monoid.map_input(
            lambda row: row['wordPredictions'],
            Wp._filtering(Wp._entry_filter(filter), Wp.stats)
        )

    _rank_metrics = {
        'mrr': lambda r: 1.0 / (1+r),
        'hit1': lambda r: r < 1,
        'hit3': lambda r: r < 3,
        'hit10': lambda r: r < 10,
        'hit20': lambda r: r < 20,
        'hit': lambda r: True
    }
    _idx_rank_metrics = {
        'idx{}_hit3'.format(i): (lambda r: r < 3, i) for i in range(MAX_INDEX)
    }

    def present(self, stats):
        '''A function that transforms the results of Wp statistics.'''
        results = stats.copy()
        results['skip'] = stats['skip'] / stats['words']
        results['miss'] = stats['miss'] / stats['words']

        # ranks
        ranks = results.pop('ranks')
        for name, metric in Wp._rank_metrics.items():
            results[name] = sum(metric(idx) * count for idx, count in
                                enumerate(ranks)) / stats['words']

        # # per-idx ranks
        idx_ranks = results.pop('idx_ranks')
        idx_counts = results.pop('idx_counts')
        for name, (metric, metric_idx) in Wp._idx_rank_metrics.items():
            if metric_idx in idx_ranks:
                results[name] = (sum(metric(idx) * count for idx, count in
                                     enumerate(idx_ranks[metric_idx])) /
                                 idx_counts[metric_idx])
        return results


class Tc:
    '''
    Monoids that may be used to accumulate stats, e.g. (two alternatives)::

        f = common.TokenFilter.alphaonly
        Tc(f).monoid.reduce(log)
        stats.aggregate_by(log, Tc(f).monoid.reduce, stats.Partition.user)
    '''

    @staticmethod
    def _stats(token_filter, row):
        predicted = 0
        calls = 0
        chars = 0
        skip_chars = 0
        for info in row['textCompletions']:
            if 'target' in info:
                allowed = sum(map(token_filter, info['target']))
                skip_chars += len(info['target']) - allowed
            else:
                allowed = info['targetChars']
            chars += allowed
            if info.get('rank') is not None:
                predicted += allowed
                # don't penalize for predictions that don't include allowable
                #  characters
                if allowed != 0:
                    calls += 1
            else:
                calls += allowed
        return {'predicted': predicted,
                'calls': calls,
                'chars': chars,
                'skip_chars': skip_chars}

    @staticmethod
    def completions(filter):
        '''Compute completion statistics from a row of the log given a
        character-level filter.

        If entries include target text, then `filter` is applied.
        Note the target text may be missing if tc was asked to exclude them
        from the evaluation results (e.g. because the source data is private).
        In these cases it is as if the filter allows all.
        '''
        return monoid.Monoid(
            identity={'predicted': 0, 'calls': 0, 'chars': 0, 'skip_chars': 0},
            lift=lambda row: Tc._stats(filter, row),
            combine=lambda x, y: {k: x[k] + y[k]
                                  for k in ['predicted', 'calls',
                                            'chars', 'skip_chars']}
        )

    def __init__(self, filter):
        self.monoid = Tc.completions(filter)

    def present(self, stats):
        '''A function that transforms the results of Tc statistics.'''
        results = stats.copy()
        results['kpc'] = stats['calls'] / stats['chars']
        del results['calls']
        results['pcpc'] = stats['predicted'] / stats['chars']
        del results['predicted']
        results['skip'] = stats['skip_chars'] / stats['chars']
        del results['skip_chars']
        return results


class Ic:
    '''
    Monoids that may be used to accumulate stats, e.g. (two alternatives)::

        f = common.TokenFilter.alphaonly
        Ic(f).monoid.reduce(log)
        stats.aggregate_by(log, Ic(f).monoid.reduce, stats.Partition.user)
    '''

    DEFAULT_ALPHA = Domain.Linear(min=0.0, max=1.0, n=11)
    DEFAULT_OOV_PENALTY = Domain.Linear(min=-8, max=0, n=5)

    @staticmethod
    def optimize(data, target='accuracy', filter=common.TokenFilter.all,
                 alpha=DEFAULT_ALPHA, oov_penalty=DEFAULT_OOV_PENALTY):
        '''Optimize ic predictions (interpolating input and lm probabilities)
        over the given data.
        '''
        def evaluate(args):
            program = Ic(filter, **args)
            score = program.present(program.monoid.reduce(data))[target]
            logging.debug('Run: %s = %f, args = %s', target, score, args)
            return score

        test_args = [dict(alpha=a, oov_penalty=p)
                     for p in oov_penalty.values()
                     for a in alpha.values()]

        logging.debug('Searching: %d configurations', len(test_args))

        scores = [(args, evaluate(args)) for args in test_args]
        best_args, best_score = max(scores, key=lambda x: x[1])

        logging.info('Best: %s = %f, args = %s', target, best_score, best_args)

        return best_args

    @staticmethod
    def scorer(candidates, alpha, oov_penalty):
        '''Return a function that computes the score of a candidate score
        (a candidate is a pair [error_score, lm_score]).
        '''
        language_scores = [x[-1] for x in candidates if x[-1] is not None]
        oov_score = oov_penalty + \
            (min(language_scores) if len(language_scores) else 0)

        def score(x):
            error_score = x[0]
            language_score = x[1] if x[1] is not None else oov_score
            return (1 - alpha) * error_score + alpha * language_score

        return score

    @staticmethod
    def rank(word_info, alpha, oov_penalty):
        '''Return the rank of the correct result with the given mixing
        settings.
        Will always be an integer >= 1.
        '''
        score = Ic.scorer(word_info['candidates'], alpha, oov_penalty)
        target_score = score(word_info['score'])
        return sum(target_score <= score(c[-2:])
                   for c in word_info['candidates'])

    @staticmethod
    def before_pass(word_info):
        '''Return true if the original was already correct (a 'pass').
        '''
        return (word_info['verbatimMatch']
                if 'verbatimMatch' in word_info else
                word_info['target'] == word_info['verbatim'])

    def __init__(self, filter, alpha, oov_penalty):
        self.filter = filter
        self.alpha = alpha
        self.oov_penalty = oov_penalty
        self.monoid = monoid.map_input(
            lambda row: row['inputCorrections'],
            monoid.Monoid(
                identity=Ic._identity, lift=self._lift, combine=Ic._combine
            )
        )

    def _lift(self, results):
        pp = 0
        pf = 0
        fp = 0
        ff = 0
        words = 0
        skip = 0
        for word_info in results:
            word = word_info.get('target')
            words += 1
            if word is not None and not self.filter(word):
                skip += 1
            else:
                before_pass = Ic.before_pass(word_info)
                after_pass = 1 == Ic.rank(word_info,
                                          alpha=self.alpha,
                                          oov_penalty=self.oov_penalty)
                pp += before_pass * after_pass
                pf += before_pass * (not after_pass)
                fp += (not before_pass) * after_pass
                ff += (not before_pass) * (not after_pass)

        return dict(pp=pp, pf=pf, fp=fp, ff=ff, skip=skip, words=words)

    _identity = dict(pp=0, pf=0, fp=0, ff=0, skip=0, words=0)

    @staticmethod
    def _combine(x, y):
        return {k: x[k] + y[k]
                for k in ['pp', 'pf', 'fp', 'ff', 'skip', 'words']}

    def present(self, stats):
        '''A function that transforms the results of Ic statistics.'''
        results = stats.copy()

        results['skip'] = stats['skip'] / stats['words']
        pass_pass = results.pop('pp')
        pass_fail = results.pop('pf')
        fail_pass = results.pop('fp')
        fail_fail = results.pop('ff')

        results['accuracy'] = \
            fdiv_null(fail_pass + pass_pass,
                      fail_pass + pass_pass + fail_fail + pass_fail)
        results['corrected_errors'] = \
            fdiv_null(fail_pass, fail_pass + fail_fail)
        results['miscorrected'] = \
            fdiv_null(pass_fail, pass_fail + pass_pass)
        results['improvement'] = \
            fdiv_null(fail_pass - pass_fail, fail_fail + fail_pass)

        return results


# Generic code to run the different stats programs

class Partition(common.ParamChoice):
    '''Partition functions for stats output.
    '''
    @staticmethod
    def all(s):
        '''Group everything into a single partition.'''
        return True

    @staticmethod
    def user(s):
        '''Group results by user.'''
        return s['userId']

    @staticmethod
    def line(s):
        '''Group results by line.'''
        return object()

    name = 'partition'
    choices = ['all', 'user', 'line']


def aggregate_by(stats, aggregate, partition=Partition.all):
    '''Aggregate a list of stats by a partition function.'''
    return [aggregate(list(s))
            for _, s in itertools.groupby(stats, partition)]


class FragmentProgram:
    monoid = monoid.keys(
        lines=monoid.counting,
        users=monoid.map_input(
            lambda row: row.get('userId'), monoid.uniqing),
        training_chars=monoid.map_input(
            lambda row: row.get('trainingChars'), monoid.maxing)
    )

    def present(self, stats):
        results = stats.copy()
        results['users'] = len(stats['users'])
        return results


class MergeProgram:
    def __init__(self, *programs):
        self.programs = programs
        self.monoid = monoid.merge(*(p.monoid for p in self.programs))

    def present(self, stats):
        results = stats
        for program in self.programs:
            results = program.present(results)
        return results


def stats(data, lines=None, opt_args=None,
          filter=common.TokenFilter.all,
          partition=Partition.all):
    '''Find the correct stats package to use, and compute stats for each
    log file.

    ``data`` - data from a log file to evaluate (e.g. read from
    ``common.read_jsonlines``)

    ``lines`` - optional limit to number of lines to process per file

    ``partition`` - optional partition function to aggregate results over

    ``filter`` - optional filter to select only some tokens to consider

    ``returns`` - a list of [stats_per_aggregate] objects
    '''
    stats_program = common.autodetect_log(data, wp=Wp, tc=Tc, ic=Ic)

    if opt_args is None and hasattr(stats_program, 'optimize'):
        # run the optimizer with default settings
        opt_args = stats_program.optimize(data, filter=filter)

    program = MergeProgram(FragmentProgram(),
                           stats_program(filter, **(opt_args or {})))
    return [program.present(a)
            for a in aggregate_by(data, program.monoid.reduce, partition)]


class Output(common.ParamChoice):
    '''How to print the stats output.
    '''
    @staticmethod
    def json(data):
        '''Dump a results set in json format.'''
        return json.dumps(data[0][1] if len(data) == 1 else dict(data),
                          sort_keys=True)

    @staticmethod
    def csv(data):
        '''Dump a dictionary in csv format (very basic)'''
        if len(data[0][1]) != 1:
            raise ValueError(
                'Can only dump CSV output for unpartitioned data'
                ' (number of partitions: %d)' % len(data[0])
            )
        keys = sorted(data[0][1][0].keys())
        header = ['log'] + keys
        lines = [[log] + [str(results[0][k]) for k in keys]
                 for log, results in data]
        return '\n'.join('\t'.join(row) for row in ([header] + lines))

    name = 'output_format'
    choices = ['json', 'csv']


# Script

@click.command()
@click.argument('log', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-v', '--verbose', default=0, count=True,
              help='How much human-readable detail to print to STDERR.')
@click.option('-n', '--lines', type=click.INT,
              help='Limit input to this number of lines')
@click.option('-o', '--output', type=Output(), default='json',
              help='Output format.')
@click.option('-p', '--partition', type=Partition(), default='all',
              help='Group the results for aggregation in this way.')
@click.option('-f', '--filter', type=common.TokenFilter(),
              default='alphaemoji',
              help='Only count tokens which match this filter.')
@click.option('-a', '--opt-args', type=common.JsonParam(), multiple=True,
              help='Pass these arguments to the stats program (for ``ic``),'
              ' for example pre-computed using ``lmc ic-opt``.')
def cli(log, verbose, lines, output, opt_args, **args):
    '''Extract summary stats from any of the challenge log files.
    Specify one or more similar log files, or pipe in results.
    '''
    common.verbosity(verbose)
    log = log or ['-']
    results = [(file, stats(read_jsonlines(file, lines), opt_args=opt, **args))
               for file, opt in common.zip_special(log, opt_args)]
    click.echo(output(results))


__doc__ += common.shell_docstring(cli, 'lmc stats')
if __name__ == '__main__':
    cli()
