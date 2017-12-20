# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from .. import grep
import emoji


def test_keep():
    a = dict(user=None, message=0, target='a')
    b = dict(user=None, message=0, target='b', select=False)
    c = dict(user=None, message=1, target='c', select=False)
    d = dict(user=None, message=2, target='d', select=True)
    e = dict(user='z', message=2, target='e', select=False)
    assert list(grep.Keep.all([a, b, c, d, e])) == [a, b, c, d, e]
    assert list(grep.Keep.message([a, b, c, d, e])) == [a, b, d]
    assert list(grep.Keep.token([a, b, c, d, e])) == [a, d]


def test_parse_pattern_target():
    # TODO: parameterize
    for pattern, target, expected in [
            # 1. Target pattern
            ('[gh].', 'great', True),
            ('[gh].', 'have', True),
            ('[gh].', 'leg', False),
            ('[gh].', 'yes', False),
            ('foo', 'foo', True),
            ('foo', 'foobar', True),
            ('^foo$', 'foo', True),
            ('^foo$', 'foobar', False),
            # 2. Negated target pattern
            ('~aa', 'have', True),
            ('~aa', 'abba', True),
            ('~aa', 'aa', False),
            ('~aa', 'aardvark', False),
            # 3. Predefined target pattern
            ('$nospace', 'abc-def', True),
            ('$nospace', 'abc\tdef', False),
            ('$alpha', 'abc:-0', True),
            ('$alpha', ':-0', False),
            ('$alphaonly', 'abc', True),
            ('$alphaonly', 'abc:-0', False),
            ('$emoji', 'a😊', True),
            ('$emoji', 'a:-)', False),
            ('$alphaemoji', '😊', True),
            ('$alphaemoji', 'a-)', True),
            ('$alphaemoji', ':-)', False),
    ]:
        assert grep.parse_pattern(pattern)(dict(target=target)) == expected


def test_parse_pattern_other():
    user_alpha = grep.parse_pattern('$user=Alpha')
    assert user_alpha(dict(user='Alpha', target='foo'))
    assert not user_alpha(dict(user='alpha', target='foo'))
    assert not user_alpha(dict(user='AlphaMan', target='foo'))
    assert not user_alpha(dict(user='ManAlpha', target='foo'))

    message_0123 = grep.parse_pattern('$message=[0123]')
    assert message_0123(dict(message=2, target='foo'))
    assert not message_0123(dict(message=4, target='foo'))
    assert not message_0123(dict(message=10, target='foo'))

    token_90 = grep.parse_pattern('$token=90')
    assert token_90(dict(token=90, target='foo'))
    assert not token_90(dict(token=91, target='foo'))
    assert not token_90(dict(token=9090, target='foo'))

    character_1x = grep.parse_pattern('$character=1.')
    assert character_1x(dict(character=18, target='foo'))
    assert not character_1x(dict(character=21, target='foo'))


def test_parse_pattern_emoji():
    emoji_pred = grep.parse_pattern('$emoji')
    alpha_emoji_pred = grep.parse_pattern('$alphaemoji')
    for emo in emoji.UNICODE_EMOJI.keys():
        assert emoji_pred(dict(target=emo))
        assert alpha_emoji_pred(dict(target=emo))


def test_parse_patterns_all():
    pred = grep.parse_patterns_all(
        'p',
        '$nospace',
        '$user=[aA].+')

    assert pred(dict(user='andy', target='open-sesame'))
    # no "p"
    assert not pred(dict(user='andy', target='ouvret-sesame'))
    # contains space
    assert not pred(dict(user='andy', target='open sesame'))
    # user doesn't start "a"
    assert not pred(dict(user='bandy', target='open-sesame'))
