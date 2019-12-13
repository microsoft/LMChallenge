# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import collections
import argparse
import sys
import math
import contextlib
import dbm
import tempfile
import struct
import itertools as it
import lmchallenge as lmc


class Context:
    '''Helper object to store a context's record of children & total
    count.
    '''
    __slots__ = ('count', 'children', '_child_order')

    def __init__(self):
        self.count = 0
        self.children = dict()
        self._child_order = None

    def set(self, child, count):
        if child in self.children:
            raise ValueError('duplicate child {}'.format(child))
        self.count += count
        self.children[child] = count
        self._child_order = None

    def probability(self, child):
        return self.children.get(child, 0) / self.count

    def predictions(self):
        # Lazily compute & cache the order of children
        if self._child_order is None:
            self._child_order = sorted(
                self.children, key=lambda k: -self.children[k])
        return self._child_order


class BackoffContext:
    '''Helper object to represent a backed off context lookup.
    '''
    __slots__ = ('_context_and_weight', '_total_weight')

    def __init__(self, context_and_weight):
        self._context_and_weight = [
            (c, w)
            for c, w in context_and_weight
            if c is not None
        ]
        self._total_weight = sum(w for c, w in self._context_and_weight)

    def log_probability(self, child):
        '''Compute the backed off log-probability of a single child
        in this backed off context.

        child -- term to look up in this context

        returns -- `None` if the context or child was missing,
                   float probability otherwise
        '''
        sum_score = sum(c.probability(child) * w
                        for c, w in self._context_and_weight)
        return (None
                if sum_score == 0 else
                math.log(sum_score / self._total_weight))

    def predictions(self):
        '''Get next-token predictions from this context.

        returns -- a generator of string tokens
        '''
        return (prediction
                for context, _ in self._context_and_weight[::-1]
                for prediction in context.predictions())


class NgramModel:
    '''An n-gram model of sequences of tokens, with interpolation backoff
    and approximate prediction (which isn't backed off).
    '''
    def __init__(self, contexts, order_weights):
        '''Create an NgramModel. See `create()`.

        contexts -- dict[tuple(context..) -> Context()]

        order_weights -- list of weights [unigram, bigram, trigram, ...]
        '''
        self._contexts = contexts
        self._order_weights = order_weights

    @classmethod
    def create(cls, ngrams, order_weights):
        '''Create the ngram sequence model from a flat sequence of
        (ngram, count) pairs.

        ngrams -- iterable of (ngram, count) pairs

        order_weights -- list of weights [unigram, bigram, trigram, ...]
                         (if too short, the last weight is used for
                         every higher order)
        '''
        contexts = dict()
        order = 1
        for (*context, head), count in ngrams:
            context = tuple(context)
            if context not in contexts:
                contexts[context] = Context()
            contexts[context].set(head, count)
            order = max(order, len(context) + 1)

        order_weights = list(it.islice(
            it.chain(order_weights, it.repeat(order_weights[-1])),
            order
        ))
        return cls(contexts, order_weights)

    def lookup(self, context):
        '''Lookup a context and return a BackoffContext instance, which
        can be used to score candidates, or enumerate predictions.

        context -- sequence of tokens (does not need to be padded)
        '''
        # add padding to the start
        context = tuple(
            it.repeat('\x1d', len(self._order_weights) - 1)
        ) + tuple(context)

        return BackoffContext([
            (self._contexts.get(() if n == 0 else context[-n:]),
             weight)
            for n, weight in enumerate(self._order_weights)
        ])


class WordModel(lmc.FilteringWordModel):
    '''A simple ngram word model.
    '''
    def __init__(self, ngrams, order_weights, n_predictions):
        '''Create the word model from a flat sequence of
        (ngram, count) pairs.

        ngrams -- iterable of (ngram, count) pairs

        order_weights -- list of weights for unigram, bigram, etc.
                         (if too short, the last weight is used for
                         every higher order)
        '''
        super().__init__(n_predictions=n_predictions)
        self._model = NgramModel.create(ngrams, order_weights)

    def predict_word_iter(self, context):
        backoff = self._model.lookup(context)
        # Don't bother computing "proper" scores
        # (backoff.log_probability(word)) for performance reasons
        # - as there is no need in this case, so just create fake
        # scores (-rank)
        return ((word, -n)
                for n, word in enumerate(backoff.predictions()))

    def score_word(self, context, candidates):
        backoff = self._model.lookup(context)
        return [(candidate, backoff.log_probability(candidate))
                for candidate in candidates]


class CharacterModel(lmc.Model):
    '''A simple ngram character model
    (only supporting scoring, not prediction).
    '''
    def __init__(self, ngrams, order_weights):
        '''Create the character model from a flat sequence of
        (ngram, count) pairs.

        ngrams -- iterable of (ngram, count) pairs

        order_weights -- list of weights for unigram, bigram, etc.
                         (if too short, the last weight is used for
                         every higher order)
        '''
        self._model = NgramModel.create(ngrams, order_weights)

    def predict(self, context, candidates):
        backoff = self._model.lookup(context)
        return [(candidate, backoff.log_probability(candidate))
                for candidate in (candidates or [])]


def parse_ngram(line):
    '''Parse a string-encoded ngram.

    line -- string -- e.g. "aaa\x1ebbb\x1e777\n"

    returns -- (ngram, count) pair -- e.g. (("aaa", "bbb"), 777)
    '''
    *ngram, count = line.rstrip('\n').split('\x1e')
    return tuple(ngram), int(count)


class DictCounter:
    '''A simple, memory-hungry counter, backed by a Python dictionary.
    '''
    def __init__(self):
        self._d = collections.defaultdict(int)

    @classmethod
    @contextlib.contextmanager
    def open(cls):
        yield cls()

    def increment(self, key):
        self._d[key] += 1

    def items(self):
        return self._d.items()


class DbmCounter:
    '''A slow counter backed by a database.
    '''
    FORMAT = '>I'

    def __init__(self, db):
        self._db = db

    @classmethod
    @contextlib.contextmanager
    def open(cls):
        with tempfile.NamedTemporaryFile() as f:
            with dbm.open(f.name, 'n') as db:
                yield cls(db)

    def increment(self, key):
        key = key.encode('utf8')
        count = self._db.get(key)
        count = (1
                 if count is None else
                 struct.unpack(self.FORMAT, count)[0] + 1)
        self._db[key] = struct.pack(self.FORMAT, count)

    def items(self):
        key = self._db.firstkey()
        while key is not None:
            yield (key.decode('utf8'),
                   struct.unpack(self.FORMAT, self._db[key])[0])
            key = self._db.nextkey(key)


def sequence(lines, order, counter=None):
    '''"Sequence up" the input lines into ngrams of the order "order".

    lines -- an iterable of lists of tokens

    order -- int

    returns -- an iterable of (ngram, count) pairs, where ngram is a
               string separated by ASCII record separator (RS) \x1E
               note that the start-of-sequence is padded with (order-1)
               ASCII group separator (GS) \x1D
    '''
    if counter is None:
        counter = DictCounter.open()
    pad = list(it.repeat('\x1d', order - 1))
    for line in lines:
        line = pad + line
        for n in range(order - 1, len(line)):
            for i in range(order):
                counter.increment('\x1e'.join(line[(n - i):(n + 1)]))
    return counter.items()


# Command line wrappers

def sequence_cli(order, disk, tokenizer):
    '''Command line version of `sequence`, applying a tokenizer regex,
    between stdin & stdout.
    '''
    lines = ([m.group(0) for m in tokenizer.finditer(line.rstrip('\n'))]
             for line in sys.stdin)
    with (DbmCounter if disk else DictCounter).open() as counter:
        for ngram, count in sequence(lines, order, counter):
            sys.stdout.write('{}\x1e{}\n'.format(ngram, count))


def sequence_words_cli(order, disk):
    '''Command line version of `sequence` for words, between stdin & stdout.
    '''
    sequence_cli(order, disk, lmc.core.common.WORD_TOKENIZER)


def sequence_chars_cli(order, disk):
    '''Command line version of `sequence` for characters, between stdin & stdout.
    '''
    sequence_cli(order, disk, lmc.core.common.CHARACTER_TOKENIZER)


def prune_cli(count):
    '''Command line for pruning ngrams that are below a minimum count.
    '''
    for line in sys.stdin:
        if count <= int(line.rstrip('\n').split('\x1e')[-1]):
            sys.stdout.write(line)


def words_cli(ngrams, weights, n_predictions):
    '''Start a word model prediction loop.
    '''
    with open(ngrams) as f:
        WordModel(map(parse_ngram, f), weights, n_predictions).run_loop()


def chars_cli(ngrams, weights):
    '''Start a character model prediction loop.
    '''
    with open(ngrams) as f:
        CharacterModel(map(parse_ngram, f), weights).run_loop()


# Command line

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Example ngram language model'
    )
    subparsers = parser.add_subparsers()

    s = subparsers.add_parser('sequence-words', help='sequence up word ngrams')
    s.add_argument('order', type=int, help='order to sequence up to')
    s.add_argument('-d', '--disk', action='store_true',
                   help='use a slow on-disk sequencer')
    s.set_defaults(execute=sequence_words_cli)

    s = subparsers.add_parser('sequence-chars',
                              help='sequence up character ngrams')
    s.add_argument('order', type=int, help='order to sequence up to')
    s.add_argument('-d', '--disk', action='store_true',
                   help='use a slow on-disk sequencer')
    s.set_defaults(execute=sequence_chars_cli)

    s = subparsers.add_parser('prune', help='prune down ngrams')
    s.add_argument('count', type=int, help='minimum count to allow')
    s.set_defaults(execute=prune_cli)

    s = subparsers.add_parser('words',
                              help='start a character model predictor')
    s.add_argument('ngrams', help='file path to ngrams dataset')
    s.add_argument('-n', '--n-predictions', default=100, type=int,
                   help='number of predictions to return')
    s.add_argument('-w', '--weights', nargs='+', type=float,
                   default=[1, 2, 2],
                   help='weights to apply to each order of prediction'
                   ' (starting with unigram)')
    s.set_defaults(execute=words_cli)

    s = subparsers.add_parser('chars',
                              help='start a character model predictor')
    s.add_argument('ngrams', help='file path to ngrams dataset')
    s.add_argument('-w', '--weights', nargs='+', type=float,
                   default=[1, 1, 10, 100, 1000],
                   help='weights to apply to each order of prediction'
                   ' (starting with unigram)')
    s.set_defaults(execute=chars_cli)

    args = vars(parser.parse_args())
    args.pop('execute')(**args)
