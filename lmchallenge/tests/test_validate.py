# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from .. import validate
import jsonschema


def test_validate_schema():
    jsonschema.Draft4Validator.check_schema(validate.schema())
