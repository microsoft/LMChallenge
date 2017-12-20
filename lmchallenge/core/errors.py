# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Utilities for corrupting and correcting text, for use with `lmchallenge.wr`.

Currently the implemented corruption is very simple - corrupt each character
with a low probability of it being replaced by another ASCII letter (or in-word
punctuation character).
'''

import random
import math
import string
import heapq

DEFAULT_CONFIG = dict(
    p_anykey=0.1,
    error_chars=string.ascii_letters + '_-#@'
)


def corrupt(config, word, rand=random):
    '''Generate a corrupted version of a word, as if typed by a sloppy typist.
    '''
    p_anykey = config['p_anykey']
    error_chars = config['error_chars']
    return ''.join(
        rand.choice(error_chars) if rand.random() < p_anykey else ch
        for ch in word
    )


def score(config, input_word, word):
    '''Return an approximate score for this word, given the error model of
    'config'.
    '''
    p_anykey = config['p_anykey']
    n_correct = sum(a == b for a, b in zip(input_word, word))
    return n_correct * math.log(1 - p_anykey) + \
        (len(word) - n_correct) * math.log(p_anykey)


class Search:
    '''Functor for finding a list of nearby candidates to a corrupted word.
    '''
    def __init__(self, words):
        self.words = {}
        for w in words:
            words_l = self.words.get(len(w))
            if words_l is None:
                words_l = []
                self.words[len(w)] = words_l
            words_l.append(w)

    @staticmethod
    def _count_matches(a, b):
        n = 0
        for ch_a, ch_b in zip(a, b):
            if ch_a == ch_b:
                n += 1
        return n

    def __call__(self, input_word, n):
        return heapq.nlargest(
            n, self.words.get(len(input_word), []),
            key=lambda w: Search._count_matches(input_word, w)
        )
