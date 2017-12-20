# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Search through logs for specific token instances (for further processing).
'''

import click
import regex
import emoji
import itertools as it
from .core import common


def _target_contains(pattern):
    xp = regex.compile(pattern)
    return lambda datum: xp.search(datum['target']) is not None


def _target_does_not_contain(pattern):
    xp = regex.compile(pattern)
    return lambda datum: xp.search(datum['target']) is None


def _key_matches(key, pattern):
    xp = regex.compile(pattern)
    return lambda datum: xp.fullmatch(str(datum[key])) is not None


def parse_pattern(pattern):
    '''Parses token filter patterns (regexes, with a few predefined values).

    `pattern` -- `string` -- see `lmchallenge.grep` for the grammar
                 (essentially a regex).

    `return` -- `predicate(dict)` -- a predicate taking a datum from a log file
    '''
    if pattern.startswith('$'):
        # predefined target pattern
        if pattern == '$nospace':
            return _target_does_not_contain(r'\s')
        elif pattern == '$alpha':
            return _target_contains(r'\p{L}')
        elif pattern == '$alphaonly':
            return _target_does_not_contain(r'[^\p{L}]')
        elif pattern == '$emoji':
            return _target_contains(emoji.get_emoji_regexp().pattern)
        elif pattern == '$alphaemoji':
            return _target_contains(r'(\p{L})|' +
                                    emoji.get_emoji_regexp().pattern)

        # other patterns
        m = regex.fullmatch(r'\$(user|message|token|character)=(.+)',
                            pattern)
        if not m:
            raise ValueError(
                'Unrecognized special pattern "{}"'.format(pattern))
        return _key_matches(m.group(1), m.group(2))

    elif pattern.startswith('~'):
        # negated target pattern
        return _target_does_not_contain(pattern[1:])

    else:
        # target pattern
        return _target_contains(pattern)


def parse_patterns_all(*patterns):
    '''Parse each pattern according to `parse_pattern`, then combine them into
    a single predicate, which requires all of them to match.

    See `lmchallenge.grep.parse_pattern`.
    '''
    return common.all_predicates(*(parse_pattern(p) for p in patterns))


def select(data, predicate):
    '''Return a copy of "data" with the "select" key added to each datum, based
    on the outcome of the predicate.

    `data` -- `iterable(dict)` -- LM Challenge log. If a datum already includes
              `"select"`, it is combined, `datum["select"] and
              predicate(datum)`).

    `predicate` -- `predicate(dict)` -- accepts a datum from `data`

    `return` -- `iterable(dict)` -- LM Challenge log, with the `"select"` key
    '''
    for datum in data:
        datum = datum.copy()
        datum['select'] = common.is_selected(datum) and predicate(datum)
        yield datum


class Keep(common.ParamChoice):
    '''After passing a log through a 'selector', the methods of this class can
    remove parts of the log that are unnecessary, for example:

    `"all"` -- keeps everything

    `"message"` -- keeps any message containing a selected token

    `"token"` -- keeps only selected tokens themselves
    '''
    name = 'keep'
    choices = ['all', 'message', 'token']

    @staticmethod
    def all(log):
        '''Keep every token in the log.'''
        return log

    @staticmethod
    def message(log):
        '''Keeps every message in the log which contains a selected token.'''
        for _, tokens in it.groupby(log, lambda x: (x['user'], x['message'])):
            tokens = list(tokens)  # need to pass through twice
            if any(map(common.is_selected, tokens)):
                yield from tokens

    @staticmethod
    def token(log):
        '''Keep only selected tokens.'''
        return filter(common.is_selected, log)


def grep(pattern, data, keep='all', and_patterns=[]):
    '''Search for `pattern` in the log `data`, returning a tagged log
    which selects part of the original data.

    `pattern` -- `string` -- see `lmchallenge.grep` for the grammar
                 (essentially a regex).

    `data` -- `iterable(dict)` -- LM Challenge log. If a selection has already
              been applied, sub-selects the log satisfying both selections.

    `keep` -- `string` -- either `"all"`, `"message"`, or `"token"`, what
              elements of the log to return (e.g. `"all"` returns the whole
              log, with a tagged selection, whereas `"token"` only returns
              tokens that match the selection).

    `and_patterns` -- `list(string)` -- additional patterns to apply, all of
                   which must match.

    `return` -- `iterable(dict)` -- LM Challenge log.
    '''
    predicate = parse_patterns_all(*([pattern] + and_patterns))
    return getattr(Keep, keep)(select(data, predicate))


# Script

@click.command()
@click.argument('pattern')
@click.argument('log', nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option('-v', '--verbose', default=0, count=True,
              help='How much human-readable detail to print to STDERR.')
@click.option('-k', '--keep', type=Keep(), default='all',
              help='After applying the pattern to select tokens, what should'
              ' be kept in the log.')
@click.option('-a', '--and', '--and-pattern', multiple=True,
              help='Specify additional patterns that must all match.')
def cli(pattern, log, verbose, keep, and_pattern):
    '''Search through logs for specific token instances.

    Pattern grammar:

    # 1. Target regex:

    REGEX -- select any token where the target contains a match of the regular
             expression (see the Python regex syntax guide for supported regex)
             (N.B. REGEX cannot start with "$" or "~")

    Note that the REGEX matches anywhere in the target, so to match only a
    whole string, use the start and end markers, i.e.

    "foo" -- matches "foo", "foobar", "only-foo", etc.

    "^foo$" -- matches "foo" but not "foobar", "only-foo", etc.


    # 2. Negated target regex:

    "~REGEX" -- select any token where the target does not contain REGEX

    e.g. "~\s" matches tokens that do not contain whitespace


    # 3. Predefined target patterns:

    "$nospace" -- tokens that don't contain whitespace

    "$alpha" -- tokens that contain alphabetic characters

    "$alphaonly" -- tokens that ONLY contain alphabetic characters

    "$emoji" -- tokens that contain emoji

    "$alphaemoji" -- tokens that contain alphabetic characters or emoji


    4. User, message, token, character patterns:

    "$user=REGEX" -- select any user matching REGEX completely

    "$message=REGEX" -- select any message number matching REGEX completely

    "$token=REGEX" -- select any token number matching REGEX completely

    "$character=REGEX" -- select any character number matching REGEX completely

    e.g. "$user=MrKim" matches the user "MrKim" but not "MrKim2"
         "$message=[0123]" matches the first 4 messages for each user
         "$token=0" matches the first token in each message

    '''
    common.verbosity(verbose)

    data = common.load_jsonlines(common.single_log(log))
    predicate = parse_patterns_all(*([pattern] + list(and_pattern)))
    common.dump_jsonlines(keep(select(data, predicate)))


__doc__ += common.shell_docstring(cli, 'lmc grep')
if __name__ == '__main__':
    cli()
