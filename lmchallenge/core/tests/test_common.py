# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from .. import common
import itertools as it
import emoji
import io
import pytest


def test_unique_by():
    # empty
    assert list(common.unique_by([], lambda x: x)) == []

    # unique strings by length
    assert list(common.unique_by(
        ["one", "two", "three", "four", "five", "six"], len)
    ) == ["one", "three", "four"]

    # check we're lazy (pass an infinite generator)
    assert list(
        it.islice(common.unique_by(it.count(0), lambda x: x), 0, 5)
    ) == [0, 1, 2, 3, 4]


def test_not_closing():
    s = io.StringIO()
    assert not s.closed

    with common.not_closing(s) as f:
        f.write('hello')
    assert not s.closed

    # If you don't wrap in "not_closing" the with statement will
    # close the resource
    with s as f:
        f.write(' again')
        # You can't do this after it is closed
        assert f.getvalue() == 'hello again'
    assert f.closed


def test_peek():
    x, items = common.peek('abcdef')
    assert x == 'a'
    assert (''.join(items)) == 'abcdef'

    x, items = common.peek(it.count())
    assert x == 0
    assert list(it.islice(items, 4)) == [0, 1, 2, 3]

    x, items = common.peek([])
    assert x is None
    assert list(items) == []


def test_autodetect_input():
    assert [dict(text="first line"), dict(text="second line")], \
        list(common.autodetect_input(["first line", "second line"]))

    # already dict => unchanged
    data_with_user = [dict(user="a", text="first line"),
                      dict(user="a", text="second line")]
    assert data_with_user, list(common.autodetect_input(data_with_user))

    with pytest.raises(ValueError):
        common.autodetect_input([1, 2])


def test_word_tokenizer():
    def tokenize(x):
        return [m.group(0) for m in common.WORD_TOKENIZER.finditer(x)]

    assert [] == tokenize('')

    assert ["one", "two", "@DIGITS", "#What_you're_like@home"] \
        == tokenize("one two \n\t@DIGITS #What_you're_like@home")

    assert ["ready4this", "...", ":-)", "yeah-buddy", ":", "??"] \
        == tokenize("ready4this... :-) yeah-buddy: ??")

    assert ["this", "is", "\U0001F4A9", "!"] \
        == tokenize("this is\U0001F4A9!")

    for emo in emoji.UNICODE_EMOJI.keys():
        assert ["pre", emo, "post"] == tokenize("pre {} post".format(emo))


def test_character_tokenizer():
    def tokenize(x):
        return [m.group(0) for m in common.CHARACTER_TOKENIZER.finditer(x)]

    assert [] == tokenize('')
    assert ['1', '\t', '2', '#', '😀'] == tokenize('1\t2#😀')
    assert ['\n', '\n', '\r'] == tokenize('\n\n\r')


def test_is_selected():
    assert common.is_selected(dict(target='foo', select=True))
    assert not common.is_selected(dict(target='foo', select=False))
    assert common.is_selected(dict(target='foo'))


def test_zip_combine():
    # As documented
    for x, y, expected in [
            # general case - non-common data is keyed under the name
            (dict(n=1, bar="a"), dict(n=1, bar="b"),
             dict(n=1, x=dict(bar="a"), y=dict(bar="b"))),

            # non-common data can be different/missing
            (dict(n=2, bar="a"), dict(n=2),
             dict(n=2, x=dict(bar="a"), y=dict())),

            # different common data generates an error
            (dict(n=3, bar="a"), dict(n=4),
             ValueError),

            # mismatched-missing common data generates an error
            (dict(n=3, bar="a"), dict(bar="a"),
             ValueError),

            # matched-missing common data is OK
            (dict(bar="a"), dict(bar="b"),
             dict(x=dict(bar="a"), y=dict(bar="b"))),
    ]:
        try:
            assert list(common.zip_combine(['n'], dict(x=[x], y=[y]))) \
                == [expected]
        except ValueError as e:
            assert expected == ValueError, e

    assert list(common.zip_combine(['n'], dict())) == []
    assert list(common.zip_combine(['n'], dict(x=[], y=[]))) == []
    assert list(common.zip_combine(
        ['a'],
        dict(x=[dict(a=1), dict(a=2)], y=[dict(a=1)]))) \
        == [dict(a=1, x={}, y={})],                     \
        'short iterables are truncated, as per zip()'


def test_zip_logs():
    base = dict(user='a', character=1, message=2, token=3,
                target='foo', select=True)
    transforms = dict(user='b', character=10, message=20, token=30,
                      target='bar', select=False)

    log_a = [dict(logp=-2.5, **base)]
    log_b = [dict(logp=-3.5, **base)]
    for key, new_value in transforms.items():
        new_base = base.copy()
        new_base[key] = new_value
        log_a.append(dict(logp=-2.5, **new_base))
        log_b.append(dict(logp=-3.5, **new_base))

    result = list(common.zip_logs(a=log_a, b=log_b))

    assert len(result) == 1 + len(transforms)

    assert result[0] == dict(
        a=dict(logp=-2.5),
        b=dict(logp=-3.5),
        **base)


class Foo:
    BAR = 123


def test_qualified_name():
    assert common.is_qualified_name("abc.def:Ghi")

    assert not common.is_qualified_name("abc.def")

    assert common.is_qualified_name(common.qualified_name(Foo))

    assert Foo == common.lookup_qualified_name(
        common.qualified_name(Foo))

    assert Foo == common.lookup_qualified_name(
        "lmchallenge.core.tests.test_common:Foo")

    assert 123 == common.lookup_qualified_name(
        "lmchallenge.core.tests.test_common:Foo.BAR")
