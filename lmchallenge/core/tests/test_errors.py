# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from .. import errors
from nose.tools import eq_, assert_less, assert_less_equal
from unittest.mock import MagicMock
import random
import math
import string


def test_corrupt():
    config = errors.DEFAULT_CONFIG
    p_anykey = config['p_anykey']
    rand = MagicMock()
    rand.random = MagicMock(
        # ok, error, ok, error, error
        side_effect=[1, p_anykey-0.001, p_anykey+0.001, 0, 0]
    )
    rand.choice = MagicMock(
        side_effect=['q', '_', 'Z']
    )
    eq_(errors.corrupt(config, "hello", rand), "hql_Z")

    eq_(errors.score(config, "hql_Z", "hello"),
        3 * math.log(p_anykey) + 2 * math.log(1 - p_anykey))


def test_corrupt_fuzz():
    config = errors.DEFAULT_CONFIG
    for i in range(1000):
        word = ''.join(random.choice(string.ascii_letters)
                       for j in range(random.randint(1, 10)))

        input_word = errors.corrupt(config, word)
        eq_(len(word), len(input_word))

        input_score = errors.score(config, input_word, input_word)
        assert_less(input_score, 0)
        score = errors.score(config, input_word, word)
        assert_less_equal(score, input_score)


def test_search():
    eq_(errors.Search([])("csn", 3), [])
    eq_(errors.Search(["can", "cs", "dam", "csn", "csna"])("csn", 3),
        ["csn", "can", "dam"])
    # if there is a tie, the first result in the list should be retained
    eq_(errors.Search(["baa", "bba", "aba", "aab", "aaa", "caa"])("aaa", 3),
        ["aaa", "baa", "aba"])


def test_search_fuzz():
    # these are all very similar, so there is some interesting overlap
    words = [
        "bird", "Bird", "bind", "bard", "Aird", "gird", "Hird", "biro", "byrd",
        "birr", "bord", "birt", "Gird", "birs", "birh", "find", "died", "bill",
        "hard", "fire"
    ]
    search = errors.Search(words)
    config = errors.DEFAULT_CONFIG
    for word in words:
        for _ in range(10):
            corrupt = errors.corrupt(config, word)
            top = search(corrupt, 5)
            last_top_score = errors.score(config, corrupt, top[-1])

            for w in words:
                score = errors.score(config, corrupt, w)
                if w in top:
                    assert_less_equal(last_top_score, score)
                else:
                    assert_less_equal(score, last_top_score)
