# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from . import eg_models
import lmchallenge as lmc
import math


def expect_close(expected, actual):
    if expected is None:
        assert actual is None
    else:
        assert abs(expected - actual) < 1e-8


def expect_partial_match(expected, actual):
    for k in expected:
        if isinstance(expected[k], float):
            expect_close(expected[k], actual[k])
        else:
            assert expected[k] == actual[k]


def expect_meta(expected, actual):
    for x, y in zip(expected, actual):
        expect_partial_match(x, y)
    assert len(expected) == len(actual)


def expect_logp(expected, actual_log):
    for logp, actual in zip(expected, actual_log):
        expect_close(logp, actual['logp'])
    assert len(expected) == len(actual_log)


def expect_completions(expected, actual_log):
    for expected_completions, actual in zip(expected, actual_log):
        assert expected_completions == actual['completions']
    assert len(expected) == len(actual_log)


def expect_results(expected, actual_log):
    for expected_results, actual in zip(expected, actual_log):
        remaining = {word: lm_score
                     for word, _, lm_score in actual['results']}
        for word, score in expected_results:
            expect_close(score, remaining.pop(word))
        assert all(score is None for score in remaining.values())
    assert len(expected) == len(actual_log)


class PlainData:
    DATA = [dict(text=line) for line in [
        'the cat ate a cat',
        'The cat',
    ]]
    META = [
        dict(user=None, message=0, token=0, character=0, target='the'),
        dict(user=None, message=0, token=1, character=4, target='cat'),
        dict(user=None, message=0, token=2, character=8, target='ate'),
        dict(user=None, message=0, token=3, character=12, target='a'),
        dict(user=None, message=0, token=4, character=14, target='cat'),

        dict(user=None, message=1, token=0, character=0, target='The'),
        dict(user=None, message=1, token=1, character=4, target='cat'),
    ]
    STATS = dict(
        users=1,
        messages=2,
        tokens=7,
        characters=19,
        skipped=0,
    )
    HUMAN_STATS = dict(
        users=1,
        messages_per_user=2.0,
        tokens_per_user=7.0,
        characters_per_token=19/7,
        skipped=0.0,
    )


class PlainCharData:
    DATA = [dict(text=line) for line in [
        'yes \t!',
        '#',
        '\n',
    ]]
    META = [
        dict(user=None, message=0, token=0, character=0, target='y'),
        dict(user=None, message=0, token=1, character=1, target='e'),
        dict(user=None, message=0, token=2, character=2, target='s'),
        dict(user=None, message=0, token=3, character=3, target=' '),
        dict(user=None, message=0, token=4, character=4, target='\t'),
        dict(user=None, message=0, token=5, character=5, target='!'),

        dict(user=None, message=1, token=0, character=0, target='#'),

        dict(user=None, message=2, token=0, character=0, target='\n'),
    ]
    STATS = dict(
        users=1,
        messages=3,
        tokens=8,
        characters=8,
        skipped=0,
    )
    HUMAN_STATS = dict(
        users=1,
        messages_per_user=3.0,
        tokens_per_user=8.0,
        characters_per_token=1.0,
        skipped=0.0,
    )


def test_simple_model_we():
    # run

    results = list(lmc.we(eg_models.SimpleModel(), PlainData.DATA))
    expect_meta(PlainData.META, results)

    expect_logp(
        [None if p is None else math.log(p)
         for p in [0.5, 0.25, None, 0.25, 0.25, None, 0.25]],
        results)

    # stats

    stats = lmc.stats.stats(results, human=False)
    expect_partial_match(PlainData.STATS, stats)
    expect_partial_match(
        dict(tokens=5,
             sum=(4 * -math.log(0.25) + -math.log(0.5))),
        stats['entropy'])

    human_stats = lmc.stats.stats(results, human=True)
    expect_partial_match(PlainData.HUMAN_STATS, human_stats)
    expect_partial_match(
        dict(hit=5/7,
             mean=(4/5 * -math.log(0.25) + 1/5 * -math.log(0.5))),
        human_stats['entropy'])


def test_simple_model_ce():
    # run

    results = list(lmc.ce(eg_models.SimpleCharModel(), PlainCharData.DATA))
    expect_meta(PlainCharData.META, results)
    expect_logp(
        [None if p is None else math.log(p)
         for p in [None, 0.5, 0.25, None, 0.125, None,
                   None,
                   0.125]],
        results)

    # stats

    stats = lmc.stats.stats(results, human=False)
    expect_partial_match(PlainCharData.STATS, stats)
    expect_partial_match(
        dict(tokens=4,
             sum=(-math.log(0.5) + -math.log(0.25) + 2 * -math.log(0.125))),
        stats['entropy'])

    human_stats = lmc.stats.stats(results, human=True)
    expect_partial_match(PlainCharData.HUMAN_STATS, human_stats)
    expect_partial_match(
        dict(hit=0.5,
             mean=(1/4 * -math.log(0.5) +
                   1/4 * -math.log(0.25) +
                   1/2 * -math.log(0.125))),
        human_stats['entropy'])


def test_simple_model_wc():
    # run

    results = list(lmc.wc(eg_models.SimpleModel(), PlainData.DATA))
    expect_meta(PlainData.META, results)

    w = ['the', 'a', 'cat']
    expect_completions(
        [[w, w, w], [w, w, w], [w, w, w], [w], [w, w, w],
         [w, w, w], [w, w, w]],
        results)

    # stats

    stats = lmc.stats.stats(results, human=False)
    expect_partial_match(PlainData.STATS, stats)
    expect_partial_match(
        dict(
            hit1=1,  # the
            hit3=5,  # the, a, (3*)cat
            hit10=5,
            hit20=5,
            hit=5,
            srr=(1.0 + 1/2 + 3 * 1/3),
        ), stats['prediction'])
    expect_partial_match(
        dict(
            tokens=2,  # the, a
            characters=4,
        ), stats['completion'])

    human_stats = lmc.stats.stats(results, human=True)
    expect_partial_match(PlainData.HUMAN_STATS, human_stats)
    expect_partial_match(
        dict(
            hit1=1/7,  # the
            hit3=5/7,  # the, a, (3*)cat
            hit10=5/7,
            hit20=5/7,
            hit=5/7,
            mrr=(1.0 + 1/2 + 3 * 1/3) / 7,
        ), human_stats['prediction'])
    expect_partial_match(
        dict(
            tokens=2/7,  # the, a
            characters=4/19,
        ), human_stats['completion'])


def test_simple_model_wr():
    # run

    results = list(lmc.wr(
        eg_models.SimpleModel(), PlainData.DATA,
        ['the', 'cat', 'fat']))
    expect_meta(PlainData.META, results)

    result_the = ('the', math.log(0.5))
    result_cat = ('cat', math.log(0.25))
    result_a = ('a', math.log(0.25))
    result_ate = ('ate', None)
    result_The = ('The', None)
    result_fat = ('fat', None)
    expect_results(
        [[result_the, result_cat, result_fat],
         [result_cat, result_the, result_fat],
         [result_ate, result_the, result_cat, result_fat],
         [result_a],
         [result_cat, result_the, result_fat],
         [result_The, result_the, result_cat, result_fat],
         [result_cat, result_the, result_fat]],
        results)

    # stats
    # - it is hard to disentangle the effects of randomness here, so we're
    # not checking any actual statistics
    expect_partial_match(
        PlainData.STATS, lmc.stats.stats(results, human=False))
    expect_partial_match(
        PlainData.HUMAN_STATS, lmc.stats.stats(results, human=True))
