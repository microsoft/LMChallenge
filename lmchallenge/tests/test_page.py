# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from .. import page
import json
import math
from nose.tools import eq_


def test_get_files():
    # Mainly just check we don't crash!
    for name, data in page._get_files().items():
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
        eq_(json.loads(page._json_dumps_min(document, float_format='.3g')),
            document)

    eq_(page._json_dumps_min(math.pi, '.1f'), "3.1")
    eq_(page._json_dumps_min(math.pi, '.3g'), "3.14")
