# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import pytest


def pytest_addoption(parser):
    parser.addoption('--run-slow', action='store_true',
                     default=False, help='run slow tests')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--run-slow'):
        skip = pytest.mark.skip(reason='only runs with --run-slow')
        for item in items:
            if 'slow' in item.keywords:
                item.add_marker(skip)
