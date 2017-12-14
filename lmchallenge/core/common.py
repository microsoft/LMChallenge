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
    '.|\n', flags=regex.MULTILINE
)
'''A Unicode character tokenizer regex.'''


def shell_docstring(command, name):
    '''
    Utility for creating docstrings:

      __doc__ += shell_docstring(cli, 'command-name')
    '''
    # Comment and indentation to recognize code segment
    text = '\nCommand line:\n\n    #!sh'
    text += '\n    ' + click.Context(command, info_name=name) \
                            .get_help().replace('\n', '\n    ')
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


def auto_open(filename, mode='rt'):
    '''Open a file, and return it (should be used in a ``with``).

    filename -- string -- path to a file (or gzip), or "-" for stdin/stdout

    returns -- file object -- performing gzip decoding if appopriate
    '''
    if filename == '-':
        if 'r' in mode:
            return not_closing(sys.stdin)
        elif '+' in mode:
            raise ValueError('Cannot return stdout/stdin with mode "r+"')
        else:
            return not_closing(sys.stdout)
    elif filename.endswith('.gz') or filename.endswith('.gzip'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)


def load_jsonlines(filename):
    '''Generate json objects from a JSONlines file.

    Note that this relies on the iterator being exhausted, or going out of
    scope, in order to close the file.

    filename -- string -- path to a file (jsonlines, or gzipped jsonlines),
                          or "-" for stdin.
    '''
    with auto_open(filename) as f:
        for line in f:
            yield json.loads(line.rstrip('\r\n'))


def dump_jsonlines(data, filename='-'):
    '''Dump data to stdout in jsonlines format.

    data -- iterable of dict -- data to dump

    out -- stream/file -- destination to write
    '''
    with auto_open(filename, 'wt') as f:
        for d in data:
            f.write(json.dumps(d, sort_keys=True) + '\n')


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


def autodetect_input(data):
    '''Convert plain text input data to the dictionary-based format.

    data -- either iterable of dictionaries or strings

    returns -- iterable dict -- if `data` is plain strings, each dictionary is
               {"text": line}, otherwise returns `data` unchanged
    '''
    first, data = peek(data)
    if isinstance(first, str):
        # auto-detection if passed an iterable of plain strings
        return (dict(text=line) for line in data)
    elif isinstance(first, dict):
        return data
    else:
        raise ValueError(
            'unexpected data item {} (expected str or dict)'.format(first))


def zip_combine(common_keys, dict_iterables):
    '''Combine a set of iterables, checking that they have identical values
    for `common_keys`, nesting any other keys under the iterable's name.

    e.g. zip_combine(["n"], dict(x=xs, y=ys))

    | x              | y              | result                           |
    |----------------|----------------|----------------------------------|
    | {n:1, bar:"a"} | {n:1, bar:"b"} | {n:1, x:{bar:"a"}, y:{bar:"b"}}  |
    | {n:2, bar:"a"} | {n:2}          | {n:1, x:{bar:"a"}, y:{}}         |
    | {n:3, bar:"a"} | {n:4}          | throws ValueError                |

    common_keys -- a list of keys that should be equal in the zipped dicts

    dict_iterables -- dict(name -> data) -- the iterables to be zipped together

    returns -- lazy sequence of dict, where the keys =
               common_keys + dict_iterables.keys()
    '''
    common_keys = set(common_keys)
    for items in zip(*dict_iterables.values()):
        # The first iterable defines the expected values for common_keys
        result = {k: items[0][k] for k in common_keys if k in items[0]}
        for name, item in zip(dict_iterables.keys(), items):
            # Check validity
            for k in common_keys:
                if result.get(k) != item.get(k):
                    raise ValueError(
                        'zip_combine mismatch between {} and {}'
                        ' ("{}": {} != {})'.format(
                            next(iter(dict_iterables.keys())), name,
                            k, result.get(k), item.get(k)))
            # Match - add in the result
            result[name] = {k: v
                            for k, v in item.items()
                            if k not in common_keys}
        yield result


def zip_logs(**data):
    '''Zip a dictionary of LMChallenge logs together, failing if the logs
    don't "match up" (i.e. were generated from different source data).

    The keys that must match (user, character, message, token, target, select)
    are returned in the root element of each result, and the log-specific
    results are included under the name of that log.

    data -- dict(name -> data) -- the logs to be zipped together

    returns -- lazy sequence of dict:
               {"user", "character", "message", "token", "target", "select",
               "log_1_name": {"logp"|"completions"|"results"...},
               "log_2_name": {"logp"|"completions"|"results"...}}
    '''
    return zip_combine(
        ["user", "character", "message", "token", "target", "select"],
        data)


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
    def auto(cls, *data, **args):
        first, data = zip(*list(peek(d) for d in data))

        is_completion = all('completions' in x for x in first)
        is_entropy = all('logp' in x for x in first)
        is_reranking = all('results' in x for x in first)
        if sum([is_completion, is_entropy, is_reranking]) != 1:
            raise Exception('Cannot infer log type from data')

        if is_completion:
            return cls.completion(*data, **args)
        elif is_entropy:
            return cls.entropy(*data, **args)
        elif is_reranking:
            return cls.reranking(*data, **args)

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


class AnsiRender:
    '''A helper for rendering in ANSI color codes.
    Usage:

    r = AnsiRender(sys.stdout)
    r.color(r.RED, bold=True)
    r.write("Hello\n")
    r.default()
    '''

    BLACK = 0
    RED = 1
    GREEN = 2
    YELLOW = 3
    BLUE = 4
    MAGENTA = 5
    CYAN = 6
    WHITE = 7
    DEFAULT = 9

    def __init__(self, outf):
        self.f = outf
        self.index = self.DEFAULT
        self.bold = False

    def default(self):
        self.color(self.DEFAULT, False)

    def color(self, index, bold):
        if self.bold and not bold:
            self.f.write(u'\x1b[0;%dm' % (30 + index))
        elif bold and not self.bold:
            self.f.write(u'\x1b[1;%dm' % (30 + index))
        elif self.index != index:
            self.f.write(u'\x1b[%dm' % (30 + index))
        self.index = index
        self.bold = bold

    def write(self, s):
        self.f.write(s)

    def close(self):
        self.f.close()
