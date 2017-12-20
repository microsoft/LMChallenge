# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Aggregate results from challenges, and calculate summary statistics.
'''

import click
import json
import csv as csvlib
import io
import sys
import struct
import hashlib
import itertools as it
from .core import common


class Accumulator:
    '''Abstract base class for "accumulation" operations that can be performed
    over sequences of events.
    '''
    @classmethod
    def create(cls):
        '''Create an 'empty' accumulator.
        '''
        return cls()

    def update(self, datum):
        '''Feed a datum into the accumulator, updating the internal state.

        `datum` -- `dict` -- single log datum
        '''
        raise NotImplementedError

    @property
    def state(self):
        '''Get the accumulator's current state.

        `return` -- `object` -- current state (snapshot) of the accumulator
        '''
        raise NotImplementedError


class Counter(Accumulator):
    '''Count the number of events.
    '''
    def __init__(self):
        self._count = 0

    def update(self, datum):
        self._count += 1

    @property
    def state(self):
        return self._count


class UniqueCounter(Accumulator):
    '''Count the number of unique datums according to some subclass-defined
    keying function.
    Note that this only removes consecutive duplicates, e.g.
       A A A B B B C C A  -- counts as 4 (A B C A)
    '''
    def __init__(self):
        self._count = 0
        self._previous = object()

    @staticmethod
    def _key(datum):
        '''The key to use to detect consecutive duplicates.
        '''
        raise NotImplementedError

    def update(self, datum):
        current = self._key(datum)
        if current != self._previous:
            self._previous = current
            self._count += 1

    @property
    def state(self):
        return self._count


class UserCounter(UniqueCounter):
    '''Count the number of users.
    '''
    @staticmethod
    def _key(datum):
        return datum.get('user')


class MessageCounter(UniqueCounter):
    '''Count the total number of messages.
    '''
    @staticmethod
    def _key(datum):
        return (datum.get('user'), datum.get('message'))


class CharacterCounter(Accumulator):
    def __init__(self):
        self._sum = 0

    def update(self, datum):
        self._sum += len(datum['target'])

    @property
    def state(self):
        return self._sum


class Hash:
    '''A stable hash, for computing fingerprints.
    '''
    @staticmethod
    def get(*columns):
        '''Generate a hash value from an ordered list of values.
        Supports only: integers, booleans, strings, None.
        '''
        h = 0
        for column in columns:
            # Arbitrary large prime (for combining hashes)
            h *= 48677
            if isinstance(column, str):
                # Use MD5 to generate a 8-byte int hash from the string
                h += struct.unpack(
                    '<Q',
                    hashlib.md5(column.encode('utf8')).digest()[:8])[0]
            elif isinstance(column, (bool, int)):
                h += 50123 * column
            elif column is None:
                h += 49307
            else:
                raise ValueError('unhashable type {}'.format(type(column)))
        return h & 0xffffffff

    @staticmethod
    def merge(a, b):
        '''Commutative, associative, merge.
        '''
        return (a + b) & 0xffffffff

    @staticmethod
    def format(h):
        return '{:08x}'.format(h)


class Fingerprint(Accumulator):
    '''Compute a stable, order-invariant hash of the data (not the results).
    Two logs with the same fingerprint may be safely compared, as they
    correspond to the same data.
    '''
    def __init__(self):
        self._fingerprint = 0

    def update(self, datum):
        self._fingerprint = Hash.merge(
            self._fingerprint,
            Hash.get(*(datum.get(k) for k in [
                'user', 'message', 'token', 'target'
            ])))

    @property
    def state(self):
        return self._fingerprint


class NextWordPrediction(Accumulator):
    '''Accumulate rank-based stats for next-word-prediction.
    '''
    def __init__(self, functions):
        # Each function in the list is a (name, fn(rank)->value) pair
        self._functions = functions
        # These values line up with those of _functions
        self._values = [0 for _ in functions]

    @classmethod
    def create(cls, *hit_ranks):
        # Split this into a function to avoid capturing 'n' lexically
        # in the generator below, which would cause all comparators to be
        # the same
        def comparator(n):
            return lambda rank: rank <= n

        return cls(
            functions=([('srr', lambda rank: 1 / rank),
                        ('hit', lambda rank: 1)]
                       + [('hit{}'.format(n), comparator(n))
                          for n in hit_ranks])
        )

    def update(self, datum):
        completions = datum.get('completions')
        if completions is not None:
            rank = common.rank(completions[0], datum['target'])
            if rank is not None:
                for idx, (name, fn) in enumerate(self._functions):
                    self._values[idx] += fn(rank)

    @property
    def state(self):
        return {name: value
                for (name, _), value in zip(self._functions, self._values)}


class Completion(Accumulator):
    '''Compute word completion stats.
    '''
    def __init__(self, max_rank):
        self._max_rank = max_rank
        self._characters = 0
        self._tokens = 0

    @classmethod
    def create(cls, max_rank):
        return cls(max_rank=max_rank)

    def update(self, datum):
        all_completions = datum.get('completions')
        if all_completions is not None:
            target = datum['target']
            for start, completions in enumerate(all_completions):
                if common.rank(completions,
                               target[start:],
                               max_rank=self._max_rank):
                    self._characters += len(datum['target']) - start
                    self._tokens += 1
                    break

    @property
    def state(self):
        return dict(
            characters=self._characters,
            tokens=self._tokens
        )


class Entropy(Accumulator):
    '''Accumulate cross-entropy stats.
    Includes a custom fingerprint (different from the toplevel fingerprint,
    as the rules for comparing entropy values are stricter than the rules
    for comparing prediction/completion/reranking.
    '''
    def __init__(self):
        self._sum = 0
        self._tokens = 0
        self._fingerprint = Fingerprint.create()

    def update(self, datum):
        logp = datum.get('logp')
        if logp is not None:
            self._sum -= logp
            self._tokens += 1
            self._fingerprint.update(datum)

    @property
    def state(self):
        return dict(
            sum=self._sum,
            tokens=self._tokens,
            fingerprint=self._fingerprint.state
        )


class Reranking(Accumulator):
    '''Optimize a reranking function, by loading all events into
    memory, and using scipy.
    '''
    def __init__(self):
        self._scores_error = []
        self._scores_lm = []

    def update(self, datum):
        results = datum.get('results')
        if results is not None:
            target = datum['target']
            self._scores_error.append(list(it.chain(
                (e for t, e, lm in results if t == target),
                (e for t, e, lm in results if t != target),
            )))
            self._scores_lm.append(list(it.chain(
                (lm for t, e, lm in results if t == target),
                (lm for t, e, lm in results if t != target),
            )))

    def finalize(self):
        '''Finalize the matrices & compute the optimal model.
        '''
        from .core import reranking as R
        max_candidates = max(len(e) for e in self._scores_error)
        self._error = R.jagged_matrix(self._scores_error, max_candidates)
        self._lm = R.jagged_matrix(self._scores_lm, max_candidates)
        self._model = R.InterpolationRerankingModel.optimize(
            error=self._error,
            lm=self._lm
        )

    @property
    def state(self):
        if len(self._scores_error) == 0:
            # Take an explicit branch to avoid the "import core.reranking"
            # unless the reranking model is used (to keep the numpy/scipy
            # dependency optional
            return dict(
                max_candidates=0,
                already_correct=0,
                correct=0,
                args=None,
            )
        else:
            self.finalize()
            from .core import reranking as R
            return dict(
                max_candidates=self._error.shape[1],
                base_correct=R.count_correct(self._error),
                correct=R.count_correct(self._model(self._error, self._lm)),
                args=self._model.args,
            )

    @classmethod
    def build_model(cls, data):
        '''A helper to build a reranking model from data, using this class.
        '''
        a = cls.create()
        for datum in data:
            if common.is_selected(datum):
                a.update(datum)
        a.finalize()
        return a._model


class Composite(Accumulator):
    '''Combine accumulators, providing results as a dictionary.
    '''
    def __init__(self, children):
        self._children = children

    @classmethod
    def create(cls, **children):
        return cls(children)

    def update(self, datum):
        for child in self._children.values():
            child.update(datum)

    @property
    def state(self):
        return {name: accumulator.state
                for name, accumulator in self._children.items()}


class Stats(Composite):
    '''A standard set of useful LM Challenge stats.
    '''
    @classmethod
    def create(cls):
        return super(Stats, cls).create(
            users=UserCounter.create(),
            messages=MessageCounter.create(),
            tokens=Counter.create(),
            characters=CharacterCounter.create(),
            fingerprint=Fingerprint.create(),
            prediction=NextWordPrediction.create(
                1, 3, 10, 20
            ),
            completion=Completion.create(
                max_rank=2
            ),
            entropy=Entropy.create(),
            reranking=Reranking.create(),
        )


class Selection(Accumulator):
    '''Select data based on the "select" tag in each datum.
    '''
    def __init__(self, child):
        self._skipped = 0
        self._child = child

    @classmethod
    def create(cls, child):
        return cls(child=child)

    def update(self, datum):
        if common.is_selected(datum):
            self._child.update(datum)
        else:
            self._skipped += 1

    @property
    def state(self):
        child_state = self._child.state
        return dict(skipped=self._skipped,
                    **(child_state
                       if isinstance(child_state, dict) else
                       dict(value=child_state)))


def humanize(stats):
    '''To be used with the output of the Selection & Stats accumulators.

    `stats` -- `dict` -- as returned by a `lmchallenge.stats.Selection`
               of `lmchallenge.stats.Stats` accumulator `.state`

    `return` -- `dict` -- human-readable staistics
    '''
    stats = stats.copy()
    r = dict()

    # General info
    r['fingerprint'] = Hash.format(stats.pop('fingerprint'))
    r['users'] = stats.pop('users')
    tokens = stats.pop('tokens')
    characters = stats.pop('characters')
    r['messages_per_user'] = stats.pop('messages') / r['users']
    r['tokens_per_user'] = tokens / r['users']
    r['characters_per_token'] = characters / tokens

    # Skipped/unselected
    if 'skipped' in stats:
        skipped = stats.pop('skipped')
        r['skipped'] = skipped / (skipped + tokens)

    # NextWordPrediction
    prediction = stats.pop('prediction')
    if prediction['hit'] != 0:
        r['prediction'] = {
            ('mrr' if k == 'srr' else k): v / tokens
            for k, v in prediction.items()
        }
        # Since we're iterating through all keys, there is no need to check
        # that all are accounted for (unlike Completion, Entropy, top-level)

    # Completion
    completion = stats.pop('completion').copy()
    if completion['tokens'] != 0:
        r['completion'] = dict(
            characters=completion.pop('characters') / characters,
            tokens=completion.pop('tokens') / tokens,
        )
        assert len(completion) == 0, 'Unexpected Completion result keys'

    # Entropy
    entropy = stats.pop('entropy').copy()
    if entropy['tokens'] != 0:
        entropy_tokens = entropy.pop('tokens')
        r['entropy'] = dict(
            fingerprint=Hash.format(entropy.pop('fingerprint')),
            hit=entropy_tokens / tokens,
            mean=entropy.pop('sum') / entropy_tokens
        )
        assert len(entropy) == 0, 'Unexpected Entropy result keys'

    # Reranking
    reranking = stats.pop('reranking').copy()
    if reranking['max_candidates'] != 0:
        reranking.pop('args')  # rarely used information
        r['reranking'] = dict(
            accuracy=reranking.pop('correct') / tokens,
            base_accuracy=reranking.pop('base_correct') / tokens,
            max_candidates=reranking.pop('max_candidates'),
        )
        assert len(reranking) == 0, 'Unexpected Reranking result keys'

    assert len(stats) == 0, 'Unexpected Stats result keys'
    return r


def stats(data, human=True):
    '''Run the standard set of accumulators over 'data'.

    `data` -- `iterable(dict)` -- LM Challenge log

    `human` -- `bool` -- show human-friendly derivative stats
               (instead of machine-friendly sums)

    `return` -- `dict` -- an accumulated dictionary of stats
    '''
    accumulator = Selection.create(child=Stats.create())
    for datum in data:
        accumulator.update(datum)
    return humanize(accumulator.state) if human else accumulator.state


class Output(common.ParamChoice):
    '''How to print the stats output.
    '''
    name = 'output_format'
    choices = ['json', 'csv']

    @staticmethod
    def json(data):
        '''Dump a results set in jsonlines format.'''
        out = io.StringIO()
        for row in data:
            json.dump(row, out, sort_keys=True)
            out.write('\n')
        return out.getvalue()

    @staticmethod
    def csv(data):
        '''Dump a dictionary in csv format.'''
        out = io.StringIO()
        keys = list(common.sort_with_override(
            common.flatten_keys(data[0]).keys(),
            'log',
            'fingerprint',
            'users',
            'messages_per_user',
            'tokens_per_user',
            'characters_per_token',
            'skipped',
        ))
        writer = csvlib.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow(common.flatten_keys(row))
        return out.getvalue()


# Script

@click.command()
@click.argument('log', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-v', '--verbose', default=0, count=True,
              help='How much human-readable detail to print to STDERR.')
@click.option('-n', '--lines', type=click.INT,
              help='Limit input to this number of lines')
@click.option('-o', '--output', type=Output(), default='json',
              help='Output format.')
@click.option('-h/-H', '--human/--no-human', default=True,
              help='Humanize the output.')
def cli(log, verbose, lines, output, human, **args):
    '''Extract summary stats from any of the challenge log files.
    Specify one or more similar log files, or pipe in results.
    '''
    common.verbosity(verbose)
    log = log or ['-']
    results = [dict(log=file,
                    **stats(it.islice(common.load_jsonlines(file), lines),
                            human=human))
               for file in log]
    sys.stdout.write(output(results))


__doc__ += common.shell_docstring(cli, 'lmc stats')
if __name__ == '__main__':
    cli()
