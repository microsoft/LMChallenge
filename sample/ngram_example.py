# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import collections
import argparse
import sys
import math
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
                for context, _ in self._context_and_weight
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


def sequence(lines, order):
    '''"Sequence up" the input lines into ngrams of the order "order".

    lines -- an iterable of lists of tokens

    order -- int

    returns -- an iterable of (ngram, count) pairs, where ngram is a
               string separated by ASCII record separator (RS) \x1E
               note that the start-of-sequence is padded with (order-1)
               ASCII group separator (GS) \x1D
    '''
    ngrams = collections.defaultdict(int)
    pad = list(it.repeat('\x1d', order - 1))
    for line in lines:
        line = pad + line
        for n in range(order - 1, len(line)):
            for i in range(order):
                ngrams['\x1e'.join(line[(n - i):(n + 1)])] += 1
    return ngrams.items()


# Command line wrappers

def sequence_words_cli(order):
    '''Command line version of `sequence` for words, between stdin & stdout.
    '''
    lines = (line.rstrip('\n').split(' ') for line in sys.stdin)
    ngrams = sequence(lines, order)
    for ngram, count in ngrams:
        sys.stdout.write('{}\x1e{}\n'.format(ngram, count))


def sequence_chars_cli(order):
    '''Command line version of `sequence` for characters, between stdin & stdout.
    '''
    lines = (list(line.rstrip('\n')) for line in sys.stdin)
    ngrams = sequence(lines, order)
    for ngram, count in ngrams:
        sys.stdout.write('{}\x1e{}\n'.format(ngram, count))


def prune_cli(count):
    '''Command line for pruning ngrams that are below a minimum count.
    '''
    for line in sys.stdin:
        if count <= int(line.rstrip('\n').split('\x1e')[-1]):
            sys.stdout.write(line)


def model_words_cli(ngrams, weights, n_predictions):
    '''Start a word model prediction loop.
    '''
    with open(ngrams) as f:
        WordModel(map(parse_ngram, f), weights, n_predictions).run_loop()


def model_chars_cli(ngrams, weights):
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
    s.set_defaults(execute=sequence_words_cli)

    s = subparsers.add_parser('sequence-chars',
                              help='sequence up character ngrams')
    s.add_argument('order', type=int, help='order to sequence up to')
    s.set_defaults(execute=sequence_chars_cli)

    s = subparsers.add_parser('prune', help='prune down ngrams')
    s.add_argument('count', type=int, help='minimum count to allow')
    s.set_defaults(execute=prune_cli)

    s = subparsers.add_parser('model-words',
                              help='start a character model predictor')
    s.add_argument('ngrams', help='file path to ngrams dataset')
    s.add_argument('-n', '--n-predictions', default=100,
                   help='number of predictions to return')
    s.add_argument('-w', '--weights', nargs='+', type=float,
                   default=[1, 2, 2],
                   help='weights to apply to each order of prediction'
                   ' (starting with unigram)')
    s.set_defaults(execute=model_words_cli)

    s = subparsers.add_parser('model-chars',
                              help='start a character model predictor')
    s.add_argument('ngrams', help='file path to ngrams dataset')
    s.add_argument('-w', '--weights', nargs='+', type=float,
                   default=[1, 1, 10, 100, 1000],
                   help='weights to apply to each order of prediction'
                   ' (starting with unigram)')
    s.set_defaults(execute=model_chars_cli)

    args = vars(parser.parse_args())
    args.pop('execute')(**args)
