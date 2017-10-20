# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from .. import common
from nose.tools import assert_equal, assert_raises
import emoji


def test_zip_special():
    for a, b, expected in [
            # some standard cases
            ("", "", []),
            ("x", "", [("x", None)]),
            ("", "y", [(None, "y")]),
            ("x", "y", [("x", "y")]),
            ("abc", [1, 2, 3], [("a", 1), ("b", 2), ("c", 3)]),

            # special 'broadcasting' behaviour
            ("abc", "", [("a", None), ("b", None), ("c", None)]),
            ("", "xyz", [(None, "x"), (None, "y"), (None, "z")]),
            ("abc", [42], [("a", 42), ("b", 42), ("c", 42)]),
            ("a", [1, 2, 3], [("a", 1), ("a", 2), ("a", 3)]),
    ]:
        assert_equal(list(common.zip_special(a, b)),
                     expected)

    with assert_raises(ValueError):
        common.zip_special("abcd", [1, 2, 3])


def test_word_tokenizer():
    def tokenize(x):
        return [m.group(0) for m in common.WORD_TOKENIZER.finditer(x)]

    assert_equal([], tokenize(''))

    assert_equal(["one", "two", "@DIGITS", "#What_you're_like@home"],
                 tokenize("one two \n\t@DIGITS #What_you're_like@home"))

    assert_equal(["ready4this", "...", ":-)", "yeah-buddy", ":", "??"],
                 tokenize("ready4this... :-) yeah-buddy: ??"))

    assert_equal(["this", "is", "\U0001F4A9", "!"],
                 tokenize("this is\U0001F4A9!"))

    for emo in emoji.UNICODE_EMOJI.keys():
        assert_equal(["pre", emo, "post"], tokenize("pre {} post".format(emo)))


def test_character_tokenizer():
    def tokenize(x):
        return [m.group(0) for m in common.CHARACTER_TOKENIZER.finditer(x)]

    assert_equal([], tokenize(''))

    assert_equal(['1', '\t', '2', '#', '😀'],
                 tokenize('1\t2#😀'))


class Foo:
    BAR = 123


def test_qualified_name():
    assert(common.is_qualified_name("abc.def:Ghi"))

    assert(not common.is_qualified_name("abc.def"))

    assert(common.is_qualified_name(common.qualified_name(Foo)))

    assert_equal(Foo, common.lookup_qualified_name(
        common.qualified_name(Foo)))

    assert_equal(Foo, common.lookup_qualified_name(
        "lmchallenge.core.tests.test_common:Foo"))

    assert_equal(123, common.lookup_qualified_name(
        "lmchallenge.core.tests.test_common:Foo.BAR"))
