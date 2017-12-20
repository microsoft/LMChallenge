# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Models that don't correctly implement the APIs.
'''

from lmchallenge import Model, FilteringWordModel
import math


# "Good" models that obey the API

class SimpleModel(Model):
    def predict(self, context, candidates):
        return filter(
            lambda x: candidates is None or x[0] in candidates,
            [('the', math.log(0.5)),
             ('a', math.log(0.25)),
             ('cat', math.log(0.25))])


class SimpleCharModel(Model):
    def predict(self, context, candidates):
        return filter(
            lambda x: candidates is None or x[0] in candidates,
            [('e', math.log(0.5)),
             ('s', math.log(0.25)),
             ('\t', math.log(0.125)),
             ('\n', math.log(0.125))])


class SimpleWordModel(FilteringWordModel):
    def score_word(self, context, candidates):
        # score according to length
        return [(c, -len(c)) for c in candidates]

    def predict_word_iter(self, context):
        # return the words from context as predictions, in reverse order
        return [(w, -n) for n, w in enumerate(context[::-1])]


class DynamicWordModel(FilteringWordModel):
    '''A unigram counting dynamic word model.
    '''
    def __init__(self):
        self._words = {}
        self._total = 0

    def score_word(self, context, candidates):
        for candidate in candidates:
            if candidate in self._words:
                yield (candidate,
                       math.log(self._words[candidate] / self._total))

    def predict_word_iter(self, context):
        words = sorted(self._words.keys(), key=lambda w: -self._words[w])
        return ((w, -n) for n, w in enumerate(words))

    def train_word(self, text):
        for word in text:
            self._words[word] = self._words.get(word, 0) + 1
        self._total += len(text)


# "Bad" models that don't obey the API

class NotImplementedModel(Model):
    def predictions(self, context, candidates):
        # wrong name
        return []


class WrongArgumentsModel(Model):
    def predict(self, context):
        # missing candidates
        return []


class WrongResultModel(Model):
    def predict(self, context, candidates):
        # missing result scores
        return ['a', 'b', 'c']
