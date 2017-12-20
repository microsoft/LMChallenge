# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from . import cli
import sys

# Pytest's doctest collector runs this file :-(
if len(sys.argv) == 0 or ('pytest' not in sys.argv[0]):
    cli()
