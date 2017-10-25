# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import re
import sys
import regex
import click
import emoji
import logging
import json
import importlib
import gzip
import itertools as it


WORD_TOKENIZER = regex.compile(
    emoji.get_emoji_regexp().pattern +
    """|[\p{L}\p{N}\p{Pc}\p{Pd}'@#]+|[\p{P}\p{S}]+"""
)
'''Our basic word tokenizer regex.'''


CHARACTER_TOKENIZER = regex.compile(
    '.'
)
'''A Unicode character tokenizer regex.'''


def shell_docstring(command, name):
    '''
    Utility for creating docstrings:

      __doc__ += shell_docstring(cli, 'command-name')
    '''
    # Must have a leading newline for Sphinx to process this correctly
    text = '\n'
    text += '**%s**::\n\n  ' % name
    text += click.Context(command, info_name=name)\
                 .get_help()\
                 .replace('\n', '\n  ')
    text += '\n'
    return text


def verbosity(level):
    '''Set logging verbosity to this level (0 to 2 inclusive).
    '''
    LOG_LEVELS = [
        logging.WARNING, logging.INFO, logging.DEBUG
    ]
    logging.basicConfig(
        format='%(levelname)s\t%(message)s',
        level=LOG_LEVELS[min(len(LOG_LEVELS) - 1, level)],
    )


def fdiv_null(a, b):
    '''Return a/b, or ``None`` if b is 0.
    '''
    return None if b == 0 else float(a) / b


def open(filename):
    '''Open a file, and return it (should be used in a ``with``).

    filename -- string -- path to a file (or gzip)

    returns -- file object -- performing gzip decoding if appopriate
    '''
    if filename.endswith('.gz') or filename.endswith('.gzip'):
        return gzip.open(filename, 'rt')
    else:
        return open(filename, 'r')


def read_jsonlines(filename):
    '''Generate json objects from a JSONlines file.

    Note that this relies on the iterator being exhausted, or going out of
    scope, in order to close the file.

    filename -- string -- path to a file (jsonlines, or gzipped jsonlines),
                          or "-" for stdin.
    '''
    f = sys.stdin if filename == '-' else open(filename)
    try:
        for line in f:
            yield json.loads(line.rstrip('\r\n'))
    finally:
        if f is not sys.stdin:
            f.close()


def autodetect_log(data, wp, tc, ic):
    '''Detect which game this log contains, and return the appropriate object.
    '''
    log_keys = data[0].keys()
    if 'wordPredictions' in log_keys:
        return wp
    elif 'textCompletions' in log_keys:
        return tc
    elif 'inputCorrections' in log_keys:
        return ic
    else:
        raise ValueError(
            'Unrecognized log line %s' % data[0]
        )


def zip_special(a, b):
    '''A bit like Python's zip, except:
      - Only works for "lengthable" 'a', 'b'.
      - If either is length=0, treat it as a sequence of None.
      - If either 'a' or 'b' is length=1, "broadcast" it to match the other.
      - Fail if the lengths are not =0, =1, or matching.
    '''
    if len(a) == 0:
        return zip(it.repeat(None), b)
    elif len(b) == 0:
        return zip(a, it.repeat(None))
    elif len(a) == 1:
        return zip(it.repeat(a[0]), b)
    elif len(b) == 1:
        return zip(a, it.repeat(b[0]))
    elif len(a) == len(b):
        return zip(a, b)
    else:
        raise ValueError("Length mismatch for zip_special: %d and %d"
                         % (len(a), len(b)))


def flatten_keys(d, separator='.'):
    '''Flatten a nested dictionary, using 'separator' to separate keys in the
    result. For example:

    {'id': {'name': 'James Bond', 'code': 0x07}, 'job': 'Spy'}
      = flatten_keys =>
    {'id.name': 'James Bond', 'id.code': 0x07, 'job: 'Spy'}
    '''
    result = {}

    def flatten(d, prefix):
        for k, v in d.items():
            if isinstance(v, dict):
                flatten(v, prefix + k + separator)
            else:
                result[prefix + k] = v
    flatten(d, '')
    return result


def rank(items, item, max_rank=None):
    '''Find the rank of 'item' in the list 'items',
    returning None if the item is missing, where the first element is
    considered rank=1.

    returns -- a rank >= 1, or None if the item was not found
    '''
    try:
        stop = max_rank if max_rank is not None else len(items)
        return 1 + items.index(item, 0, stop)
    except ValueError:
        return None


def sort_with_override(items, *first_items):
    '''Sort a list, but move 'first_items' to the front in the given order.
    '''
    primary_order = {k: i for i, k in enumerate(first_items)}
    # use a tuple (primary_order, item) as the sort value, which will
    # move items matching primary_order to the front (as a major sort index)
    return sorted(items, key=lambda item: (
        primary_order.get(item, len(primary_order)),
        item
    ))


class JsonParam(click.ParamType):
    '''Click parameter type for parsing JSON.
    If the parameter is a valid filename, assumes that it is a path to a json
    file, and reads that file instead.
    '''
    name = 'json'

    def convert(self, value, param, ctx):
        try:
            if os.path.exists(value):
                with open(value) as f:
                    return json.load(f)
            else:
                return json.loads(value)
        except ValueError as e:
            self.fail(str(e))

    def get_metavar(self, param):
        return 'JSON'


class ParamChoice(click.ParamType):
    '''Like ``click.Choice``, but looks up attributes on specific subclasses.

    To subclass, define:
    ``name` - the descriptive name for the parameter
    ``choices`` - a list of strings, each of which is a valid attr name
    '''

    def convert(self, value, param, ctx):
        if value in type(self).choices:
            return getattr(self, value)
        else:
            self.fail('expected one of {%s}, actually "%s"' % (
                ', '.join(map(repr, type(self).choices)),
                value
            ))

    def get_metavar(self, param):
        return '(%s)' % ('|'.join(type(self).choices))


class TokenFilter(ParamChoice):
    '''Filters for classes of tokens.
    '''
    SPACE_PATTERN = regex.compile(r'\s')
    ALPHA_PATTERN = regex.compile(r'\p{L}')
    NONALPHA_PATTERN = regex.compile(r'[^\p{L}]')
    EMOJI_PATTERN = emoji.get_emoji_regexp()
    MARKER_PATTERN = regex.compile(r'@\p{Lu}*')

    @staticmethod
    def all(t):
        '''Allow all tokens.'''
        return True

    @staticmethod
    def nospace(t):
        '''Exclude tokens that contain whitespace.'''
        return TokenFilter.SPACE_PATTERN.search(t) is None

    @staticmethod
    def nomarker(t):
        '''Exclude tokens that are 'markers' (such as @ALPHANUM).'''
        return TokenFilter.MARKER_PATTERN.match(t) is None

    @staticmethod
    def alpha(t):
        '''Include tokens that contain alphabetic characters.'''
        return TokenFilter.nomarker(t) and \
            (TokenFilter.ALPHA_PATTERN.search(t) is not None)

    @staticmethod
    def alphaonly(t):
        '''Include tokens that are all alphabetic characters.'''
        return TokenFilter.nomarker(t) and \
            (TokenFilter.NONALPHA_PATTERN.search(t) is None)

    @staticmethod
    def alphaemoji(t):
        '''Include tokens that are contain alphabetic characters,
        or an emoji.'''
        return (TokenFilter.nomarker(t) and
                (TokenFilter.ALPHA_PATTERN.search(t) is not None)) or \
            (TokenFilter.EMOJI_PATTERN.search(t) is not None)

    @staticmethod
    def emoji(t):
        '''Include tokens that are an emoji.'''
        return TokenFilter.EMOJI_PATTERN.search(t) is not None

    name = 'token_filter'
    choices = ['all', 'nospace', 'nomarker',
               'alpha', 'alphaonly', 'alphaemoji', 'emoji']


def qualified_name(x):
    '''Return the qualified name of 'x' (including the module).
    The format is: ``module.submodule:attr.subattr``.
    '''
    return '%s:%s' % (x.__module__, x.__qualname__)


QUALIFIED_NAME_PATTERN = re.compile('^([^: ]+):([^: ]+)$')


def is_qualified_name(name):
    '''Determine if this is a qualified name of a type (with module).
    '''
    return QUALIFIED_NAME_PATTERN.match(name) is not None


def lookup_qualified_name(name, base_package=None):
    '''Return the attribute for the qualified 'name'
    (which should be module.submodule:attr.subattr,
    e.g. from `qualified_name`).
    '''
    m = QUALIFIED_NAME_PATTERN.match(name)
    if m is None:
        raise ValueError('could not parse qualified name "%s"' % name)

    module_name = m.group(1)
    attr_names = m.group(2)
    module = importlib.import_module(module_name, base_package)
    try:
        obj = module
        for attr_name in attr_names.split('.'):
            obj = getattr(obj, attr_name)
        return obj
    except AttributeError:
        raise AttributeError(
            'module %s has no attribute %r'
            % (module, attr_names))
