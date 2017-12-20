# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Utilities for the optimization and evaluation of reranking models.
'''

import numpy as np
import scipy.optimize


# Helpers

def replace_none_vector(values):
    '''Convert 'values' to a vector, replacing None with -infinity.
    '''
    return np.array([(v if v is not None else -np.inf) for v in values],
                    dtype=np.float32)


def jagged_matrix(values, n):
    '''Convert a jagged Python array to a matrix, replacing None & truncated
    values with -infinity.

    `values` -- `list(list(float))` -- a list of 'V' lists, of maximum size 'n'

    `n` -- `int` -- the number of columns in the result

    `return` -- `array[V, n; float]` -- matrix containing `values`
    '''
    result = np.full((len(values), n), -np.inf, dtype=np.float32)
    for i, row in enumerate(values):
        result[i, :len(row)] = replace_none_vector(row)
    return result


def count_correct(scores):
    '''Compute number of correct (rank=1) scores for a matrix of scores, where
    the score of the intended "target" is at index 1 of each row.

    N.B. Uses greater-than rather than greater-than-or-equal,
    although this is possibly a bit harsh (you could have achieved
    the correct via some arbitrary tie-breaking function).

    `scores` -- `array[N, C; float]` -- scores of all terms, where
                `scores[:, 0]` are the intended target's scores

    `return` -- `int` -- number of correct (rank=1) results
                (in the range `[0, N]`)
    '''
    return int((scores[:, 0] > scores[:, 1:].max(axis=1)).sum())


# Reranking

class RerankingModel:
    '''A model that is capable of combining error & language model scores
    to rerank candidates (e.g. for the goal of optimizing combined ranking
    accuracy).
    '''
    @classmethod
    def guess(cls, error, lm):
        '''Return the initial guess at a good set of arguments.

        `error` -- `array[N; float]` -- example error scores

        `lm` -- `array[N; float]` -- example language model scores

        `return` -- `dict` -- `{"arg_name": initial_value}`
        '''
        raise NotImplementedError

    def __init__(self, **args):
        self.args = args
        for k, v in args.items():
            setattr(self, k, v)

    def __call__(self, error, lm):
        '''Evaluate the reranking model for the given error & LM scores.

        `error` -- `array[*; float]` -- error scores (any shape permitted)

        `lm` -- `array[*; float]` -- language model scores (any shape
                permitted, but must match `error`)

        `return` -- `array[*; float]` -- combined scores from the model (same
                    shape as `error` & `lm`)
        '''
        raise NotImplementedError

    @classmethod
    def optimize(cls, error, lm):
        '''Optimize a reranking model for Hit@1 disambiguation.

        `return` -- `lmchallenge.core.reranking.RerankingModel` --
                    an optimized model instance
        '''
        guess = cls.guess(error=error, lm=lm)

        def create(argv):
            return cls(**{k: v for k, v in zip(guess.keys(), argv)})

        return create(scipy.optimize.fmin(
            lambda argv: -count_correct(create(argv)(error, lm)),
            x0=list(guess.values()),
            disp=False,
        ))


class InterpolationRerankingModel(RerankingModel):
    '''Implements an interpolation-with-minimum combination model:

        score = max(alpha * lm_score, beta) + (1 - alpha) * error_score

    Hyperparameters:

        `alpha` -- `float` -- how much to trust the language model

        `beta` -- `float` -- the minimum contribution from the language model
                  (e.g. for protection against OOV)
    '''
    @classmethod
    def guess(cls, error, lm):
        return dict(
            alpha=0.5,
            beta=0.5 * float(np.median(lm[lm != -np.inf])),
        )

    def __call__(self, error, lm):
        return (
            (1 - self.alpha) * error +
            np.maximum(self.alpha * (lm if lm is not None else float('-inf')),
                       self.beta)
        )

    def __str__(self):
        return 'score = {:.3g} * error + max({:.3g} * lm, {:.3g})'.format(
            1 - self.alpha,
            self.alpha,
            self.beta,
        )
