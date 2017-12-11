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
import contextlib
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


def unique_by(iterable, func):
    '''General itertools-like function for lazy-uniquing via a key function.
    '''
    added = set([])
    for x in iterable:
        fx = func(x)
        if fx not in added:
            added.add(fx)
            yield x


def all_predicates(*predicates):
    '''Combine predicates into a single predicate, testing if all of them match.
    '''
    return lambda x: all(predicate(x) for predicate in predicates)


@contextlib.contextmanager
def not_closing(f):
    '''A context manager that doesn't call close on the resource.

    For example, use this when you want to run:

        with sys.stdin as f:
            # do something with f, closes sys.stdin at the end

    Instead, you can do:

        with not_closing(sys.stdin) as f:
           # do something with f, not closing sys.stdin at the end

    '''
    yield f


def auto_open(filename):
    '''Open a file, and return it (should be used in a ``with``).

    filename -- string -- path to a file (or gzip), or "-" for stdin

    returns -- file object -- performing gzip decoding if appopriate
    '''
    if filename == '-':
        return not_closing(sys.stdin)
    elif filename.endswith('.gz') or filename.endswith('.gzip'):
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
    with auto_open(filename) as f:
        for line in f:
            yield json.loads(line.rstrip('\r\n'))


def dump_jsonlines(data, out=sys.stdout):
    '''Dump data to stdout in jsonlines format.

    data -- iterable of dict -- data to dump

    out -- stream/file -- destination to write
    '''
    for d in data:
        out.write(json.dumps(d, sort_keys=True) + "\n")


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


def peek(iterable):
    '''Get the first item out of an iterable, then reattach it, so you can
    dispatch based on the first item, then process all items.

    iterable -- an iterable or collection of items (of any type)

    returns -- (first_item, iterable) -- a pair of the first item, and an
               iterable containing all items (including the first)
    '''
    iterable = iter(iterable)
    try:
        first_item = next(iterable)
        # rebuid an iterable of all items
        all_items = it.chain([first_item], iterable)
        return (first_item, all_items)
    except StopIteration:
        return (None, ())


def is_selected(datum):
    '''Is this datum selected (note that a missing 'select' key implicitly means
    it should be selected.
    '''
    return datum.get('select', True)


class JsonParam(click.ParamType):
    '''Click parameter type for parsing JSON.
    If the parameter is a valid filename, assumes that it is a path to a json
    file, and reads that file instead.
    '''
    name = 'json'

    def convert(self, value, param, ctx):
        try:
            if os.path.exists(value):
                with auto_open(value) as f:
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


class ChallengeChoice(ParamChoice):
    '''Select processing to run on a generated log.
    '''
    name = 'challenge'
    choices = ['auto', 'completion', 'entropy', 'reranking']

    @classmethod
    def auto(cls, data, **args):
        first, data = peek(data)

        is_completion = 'completions' in first
        is_entropy = 'logp' in first
        is_reranking = 'results' in first
        if sum([is_completion, is_entropy, is_reranking]) != 1:
            raise Exception('Cannot infer log type from data')

        if is_completion:
            return cls.completion(data, **args)
        elif is_entropy:
            return cls.entropy(data, **args)
        elif is_reranking:
            return cls.reranking(data, **args)

    @staticmethod
    def completion(data, **args):
        raise NotImplementedError

    @staticmethod
    def entropy(data, **args):
        raise NotImplementedError

    @staticmethod
    def reranking(data, **args):
        raise NotImplementedError


def single_log(logs):
    '''When using @click.argument('log', nargs=-1), this limits the log to zero
    (for stdin) or one log file.

    logs -- a list of log files

    returns -- exactly one log file name
    '''
    if len(logs) == 0:
        return '-'
    elif len(logs) == 1:
        return logs[0]
    else:
        raise ValueError('Can only process zero or one log files')


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
