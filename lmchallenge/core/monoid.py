# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import functools


# Helpers

def _merge_dicts(*dicts):
    m = {}
    for d in dicts:
        m.update(d)
    return m


# Basic definition

class Monoid:
    '''A wrapper for a simple triple of operations - identity, lift, and combine.
    These should obey the Monoid laws of associativity and identity, i.e.

        combine(lift(x), identity) == lift(x)
        combine(identity, lift(x)) == lift(x)
        combine(lift(x), combine(lift(y), lift(z)) \
            == combine(combine(lift(x), lift(y)), lift(z))
    '''

    __slots__ = ['identity', 'lift', 'combine']

    def __init__(self, identity, lift, combine):
        self.identity = identity
        self.lift = lift
        self.combine = combine

    def reduce(self, xs):
        '''Lift each x using the Monoid, then combine them all together.
        '''
        return functools.reduce(self.combine,
                                (self.lift(x) for x in xs),
                                self.identity)


# Simple monoids

counting = Monoid(identity=0, lift=lambda x: 1, combine=lambda a, b: a + b)

summing = Monoid(identity=0, lift=lambda x: x, combine=lambda a, b: a + b)

maxing = Monoid(
    identity=None,
    lift=lambda x: x,
    combine=lambda a, b: b if a is None else (None if b is None else max(a, b))
)

uniqing = Monoid(
    identity=set([]),
    lift=lambda x: set([x]),
    combine=lambda a, b: a.union(b)
)


# Monoid transformers

def map_input(f, monoid):
    '''Map the input to a monoid by a preparing function.
    '''
    return Monoid(
        monoid.identity,
        lambda x: monoid.lift(f(x)),
        monoid.combine
    )


def filter_input(pred, monoid):
    '''Filter inputs to a monoid using a predicate (applied to the un-lifted
    input).
    '''
    return Monoid(
        monoid.identity,
        lambda x: monoid.lift(x) if pred(x) else monoid.identity,
        monoid.combine
    )


def merge(*submonoids):
    '''For monoids of 'output type' `dict`, create a `Monoid` that merges
    their keys (which should not collide).
    '''
    return Monoid(
        _merge_dicts(*[m.identity for m in submonoids]),
        lambda x: _merge_dicts(*[m.lift(x) for m in submonoids]),
        lambda a, b: _merge_dicts(*[m.combine(a, b) for m in submonoids])
    )


def keys(**submonoids):
    '''Return a monoid that produces {k: submonoid} mappings.
    '''
    return Monoid(
        {k: m.identity for k, m in submonoids.items()},
        lambda x: {k: m.lift(x) for k, m in submonoids.items()},
        lambda a, b: {k: m.combine(a[k], b[k])
                      for k, m in submonoids.items()}
    )


def multiplex(dispatch, **submonoids):
    '''Return a monoid that selects which submonoid to send an to based on a
    dispatch function.
    '''
    def lift(x):
        k = dispatch(x)
        {k: submonoids[k].lift(x)}
    return Monoid(
        {k: m.identity for k, m in submonoids.items()},
        lift,
        lambda a, b: {k: m.combine(a.get(k, m.identity), b.get(k, m.identity))
                      for k, m in submonoids.items()}
    )
