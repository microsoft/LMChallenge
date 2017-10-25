# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Utilities for the optimization and evaluation of reranking models.
'''

import numpy as np
import scipy.optimize


class RerankingModel:
    '''A model that is capable of combining error & language model scores
    to rerank candidates (e.g. for the goal of optimizing combined ranking
    accuracy).
    '''
    def guess(self, error, lm):
        '''Return the initial guess at a good set of arguments.

        error -- array[N; float] -- example error scores

        lm -- array[N; float] -- example language model scores

        returns -- a dictionary {"arg_name": initial_value}
        '''
        raise NotImplementedError

    def __call__(self, error, lm, **args):
        '''Evaluate the reranking model for the given error & LM scores.

        error -- array[*; float] -- error scores (any shape permitted)

        lm -- array[*; float] -- language model scores (any shape permitted,
                                 but must match 'error')

        args -- in the same format as 'guess'

        returns -- array[*; float] -- combined scores from the model (same
                                      shape as error & lm)
        '''
        raise NotImplementedError


class InterpolationRerankingModel(RerankingModel):
    '''Implements an interpolation-with-minimum combination model:

        score = max(alpha * lm_score, beta) + (1 - alpha) * error_score

    Hyperparameters:

        alpha -- how much to trust the language model

        beta -- the minimum contribution from the language model (e.g.
                for protection against OOV)
    '''
    def guess(self, error, lm):
        return dict(
            alpha=0.5,
            beta=0.5 * float(np.median(lm[lm != -np.inf])),
        )

    def __call__(self, error, lm, alpha, beta):
        return np.maximum(alpha * lm, beta) + (1 - alpha) * error


def replace_none_vector(values):
    '''Convert 'values' to a vector, replacing None with -infinity.
    '''
    return np.array([(v if v is not None else -np.inf) for v in values],
                    dtype=np.float32)


def jagged_matrix(values, n):
    '''Convert a jagged Python array to a matrix, replacing None & truncated
    values with -infinity.

    values -- a list of 'V' lists, of maximum size 'n'

    n -- the number of columns in the result

    returns -- a (V, n) numpy array
    '''
    result = np.full((len(values), n), -np.inf, dtype=np.float32)
    for i, row in enumerate(values):
        result[i, :len(row)] = replace_none_vector(row)
    return result


def count_correct(targets_score, others_score):
    '''Compute number of correct (rank=1) scores for an vector of target scores,
    against a matrix of "others" scores.

    N.B. Uses greater-than rather than greater-than-or-equal,
    although this is possibly a bit harsh (you could have achieved
    the correct via some arbitrary tie-breaking function).

    target_score -- array[N; float] -- scores of the desired target

    others_score -- array[N, C; float] -- scores of other terms

    returns -- number of correct (rank=1) results, in the range [0, N]
    '''
    return (targets_score > others_score.max(axis=1)).sum()


def optimize_accuracy(model, targets_error, targets_lm,
                      others_error, others_lm):
    '''Optimize a reranking model for Hit@1 disambiguation.

    returns -- (best_correct, best_model_args)
    '''
    guess = model.guess(
        error=np.append(others_error, targets_error[:, np.newaxis], axis=1),
        lm=np.append(others_lm, targets_lm[:, np.newaxis], axis=1),
    )
    arg_keys = guess.keys()

    def to_args(argv):
        return {k: v for k, v in zip(arg_keys, argv)}

    def fn(argv):
        args = to_args(argv)
        return -count_correct(
            targets_score=model(targets_error, targets_lm, **args),
            others_score=model(others_error, others_lm, **args),
        )

    best_argv, best_neg_ncorrect, *_ = scipy.optimize.fmin(
        fn,
        x0=[guess[k] for k in arg_keys],
        disp=False,
        full_output=True,
    )

    return (-best_neg_ncorrect, to_args(best_argv))
