# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from .. import pretty
import json
import math


def test_get_viewer_files():
    # Mainly just check we don't crash!
    for name, data in pretty._get_viewer_files().items():
        assert isinstance(data, str)
        assert len(data)


def test_json_dumps_min():
    for document in [
            None,
            "text",
            r'\escape\ntext\t\r\"',
            10000,
            [],
            [123],
            dict(abc=1, d=None, e=12.3, g=[4, 5, 0.00000727]),
    ]:
        assert json.loads(
            pretty._json_dumps_min(document, float_format='.3g')) \
            == document

    assert pretty._json_dumps_min(math.pi, '.1f') == '3.1'
    assert pretty._json_dumps_min(math.pi, '.3g') == '3.14'
    # Tuples are written as JSON lists
    assert pretty._json_dumps_min(('abc', 123)) == '["abc",123]'
