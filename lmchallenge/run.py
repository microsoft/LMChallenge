# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Run evaluation utilities over a model, to generate LM Challenge logs.
'''

import click
import json
import sys
import logging
import random
import itertools as it
import functools as ft
from .core import common, errors, model


# Helpers

def find_tokens(text, tokenizer):
    '''Return a generator of `{"token", "character", "target"}` dictionaries by
    tokenizing some text.

    `text` -- `string` -- the text to tokenize

    `tokenizer` -- `regex.Regex` -- the tokenizer to use on the text
                   (supporting `finditer`)

    `return` -- `generator(dict)` -- a generator of tokens, which have keys:

      - `token` -- `int` -- index of the token in the message
      - `character` -- `int` -- index of the first character of the token in
                       the message
      - `target` -- `string` -- the text content of the token itself
    '''
    for i, m in enumerate(tokenizer.finditer(text)):
        yield dict(token=i, character=m.span()[0], target=m.group())


def get_completions(model, context, target):
    '''A generator of completions from a model, with successive "typed" prefixes.

    `model` -- `lmchallenge.core.model.Model` -- model to evaluate

    `context` -- `string` -- the text before the target

    `target` -- `string` -- the token being typed

    `return` -- `generator(list(string))` -- generates lists of string
                predictions, at each point while typing out `target`
    '''
    for i in range(len(target)):
        yield [w for w, s in model.predict(context + target[:i], None)]


def get_logp(model, context, target):
    '''Wrap the model.predict API to query the score of a single target.
    '''
    results = list(model.predict(context, [target]))
    if 2 <= len(results):
        logging.warning('multiple results returned for a single candidate')
    try:
        return next(s for w, s in results if w == target)
    except StopIteration:
        return None


# Evaluation

def evaluate_completions(model, context, target, next_word_only):
    '''Evaluator for Word Completion.

    `next_word_only` -- `bool` -- save time by only producing results for
                        next-word-prediction (i.e. no completion)
    '''
    return dict(completions=list(it.islice(
        get_completions(
            model=model,
            context=context,
            target=target,
        ),
        1 if next_word_only else None
    )))


def evaluate_entropy(model, context, target):
    '''Evaluator for Word/Character Entropy.
    '''
    return dict(logp=get_logp(model=model, context=context, target=target))


def evaluate_reranking(model, context, target,
                       error_config, search, num_candidates, rand):
    '''Evaluator for Error Reranking.

    `error_config` -- `dict` -- for `lmchallenge.core.errors`

    `serach` -- `lmchallenge.core.errors.Search` -- find nearby words

    `num_candidates` -- `int` -- number of candidates to search

    `rand` -- `random.Random` -- use for corrupting the text
    '''
    corrupted = errors.corrupt(error_config, target, rand=rand)

    candidates = search(corrupted, num_candidates)

    # clip off trailing candidates if needed to ensure corrupted & target
    # can be added - this should keep the maximum size of candidates
    # clamped at 'n' and ensuring that it contains 'target' and 'corrupted'
    candidates = set(
        candidates[:(num_candidates -
                     (target not in candidates) -
                     (corrupted not in candidates))]
    ) | set([corrupted, target])

    lm_scores = dict(model.predict(context, candidates))

    results = [
        (candidate,
         errors.score(error_config, corrupted, candidate),
         lm_scores.get(candidate, None))
        for candidate in candidates
    ]
    return dict(
        verbatim=corrupted,
        results=list(sorted(results, key=lambda x: -x[1]))
    )


def run_tokens(model, data, train, tokenizer, evaluate):
    '''Run a per-token evaluation.

    `model` -- `lmchallenge.core.model.Model` -- to evaluate

    `data` -- `iterable(dict)` -- an iterable of "message" dictionaries,
              containing:

     - `text` -- `string` -- the contents of the message
     - `user` -- `string` (optional) -- a user ID

    `train` -- `bool` -- should we train after every message?

    `tokenizer` -- `regex.Regex` -- tokenizer for finding tokens in messages

    `evaluate` -- `callable(model, context, target) -> dict` -- runs evaluation
                  for a single target

    `return` -- `generator(dict)` -- generator of results, containing:

     - `user` -- `string` -- from the message
     - `message` -- `int` -- index of message
     - `token` -- `int` -- index of token within the message
     - `character` -- `int` -- index of character within the message
     - `target` -- `string` -- token being typed
    '''
    data_by_user = it.groupby(data, lambda x: x.get('user'))
    for user, messages in data_by_user:
        if train:
            model.clear()
        for message_n, message in enumerate(messages):
            for token in find_tokens(text=message['text'],
                                     tokenizer=tokenizer):
                yield dict(
                    user=user,
                    message=message_n,
                    **evaluate(
                        model=model,
                        context=message['text'][:token['character']],
                        target=token['target']
                    ),
                    **token
                )
            if train:
                model.train(message['text'])


def wc(model, data, train=False, next_word_only=False):
    '''Run the Word Completion task over `model` and `data` to generate a
    result log.

    `model` -- `lmchallenge.core.model.Model` -- to evaluate

    `data` -- `iterable(string)` or `iterable(dict)` -- text data

    `train` -- `bool` -- should the model be trained after each line?

    `next_word_only` -- `bool` -- speed up evaluation by only evaluating
                        next-word-prediction, not completion

    `return` -- `iterable(dict)` -- LM Challenge log
    '''
    return run_tokens(
        model,
        common.autodetect_input(data),
        train=train,
        tokenizer=common.WORD_TOKENIZER,
        evaluate=ft.partial(
            evaluate_completions,
            next_word_only=next_word_only
        ))


def we(model, data, train=False):
    '''Run the Word Entropy task over `model` and `data` to generate a
    result log.

    Word Entropy scans through the text word-by-word and asks
    the model to score each word based only upon previous context. The
    model is responsible for returning a normalized log probability as the
    score for any context and target queried. Two models may only be compared
    if they share the same vocabulary.

    `model` -- `lmchallenge.core.model.Model` -- to evaluate

    `data` -- `iterable(string)` or `iterable(dict)` -- text data

    `train` -- `bool` -- should the model be trained after each line?

    `return` -- `iterable(dict)` -- LM Challenge log
    '''
    return run_tokens(
        model,
        common.autodetect_input(data),
        train=train,
        tokenizer=common.WORD_TOKENIZER,
        evaluate=evaluate_entropy)


def ce(model, data, train=False):
    '''Run the Character Entropy task over `model` and `data` to generate a
    result log.

    Character Entropy scans through the text character-by-character and asks
    the model to score each character based only upon previous context. The
    model is responsible for returning a normalized log probability as the
    score for any context and target queried. Two models may only be compared
    if they share the same vocabulary.

    `model` -- `lmchallenge.core.model.Model` -- to evaluate

    `data` -- `iterable(string)` or `iterable(dict)` -- text data

    `train` -- `bool` -- should the model be trained after each line?

    `return` -- `iterable(dict)` -- LM Challenge log
    '''
    return run_tokens(
        model,
        common.autodetect_input(data),
        train=train,
        tokenizer=common.CHARACTER_TOKENIZER,
        evaluate=evaluate_entropy)


def wr(model, data, vocab, train=False, seed=42,
       num_candidates=100, error_config=errors.DEFAULT_CONFIG):
    '''Run the Word Reranking task over `model` and `data` to generate a
    result log.

    Word Reranking corrupts the original text word-by-word using a simple
    character substitution error model, then looks up nearby candidate words
    in a (large) vocabulary of words. Each candidate is paired with the
    probability under the error model of generating the corruption (a "perfect"
    correction model score), and the task for model is to generate language
    model scores for each candidate that can be combined linearly with the
    error score to rank the candidates and recover the original text.

    `model` -- `lmchallenge.core.model.Model` -- to evaluate

    `data` -- `iterable(string)` or `iterable(dict)` -- text data

    `vocab` -- `iterable(string)` -- vocabulary of words to corrupt the data to

    `train` -- `bool` -- should the model be trained after each line?

    `seed` -- `int` or `None` -- random seed to use to generate corrupted
              candidates

    `num_candidates` -- `int` -- number of corrupted candidates to consider

    `error_config` -- `dict` -- see `lmchallenge.core.errors` - defines the
                      generation and scoring of error candidates

    `return` -- `iterable(dict)` -- LM Challenge log
    '''
    return run_tokens(
        model,
        common.autodetect_input(data),
        train=train,
        tokenizer=common.WORD_TOKENIZER,
        evaluate=ft.partial(
            evaluate_reranking,
            error_config=error_config,
            search=errors.Search(words=set(vocab)),
            num_candidates=num_candidates,
            rand=random.Random(seed),
        ))


# Command line helpers

class PredictorSpec(click.ParamType):
    '''Loads a predictor, either from a Python module or a shell command.
    '''
    class PythonModel:
        def __init__(self, ctor):
            self.ctor = ctor

        def __call__(self, options):
            return self.ctor(**options)

    class ShellModel:
        def __init__(self, cmd):
            self.cmd = cmd

        def __call__(self, options):
            return model.ShellModel(self.cmd, options)

    name = 'predictor_spec'

    def get_metavar(self, param):
        return 'SPEC'

    def convert(self, value, param, ctx):
        if common.is_qualified_name(value):
            return self.PythonModel(common.lookup_qualified_name(value))
        else:
            return self.ShellModel(value)


class InputFormat(common.ParamChoice):
    '''Input handling - text or json,
    '''
    name = 'input_format'
    choices = ['auto', 'text', 'json']

    @staticmethod
    def _is_json(line):
        try:
            # if you can parse it as JSON
            d = json.loads(line)
            # and is an object containing a key called "text"
            # then assume it is our "json" format
            return isinstance(d, dict) and 'text' in d
        except json.JSONDecodeError:
            return False

    @classmethod
    def auto(cls, lines):
        first_line, lines = common.peek(lines)
        if first_line is None:
            pass
        elif cls._is_json(first_line):
            yield from cls.json(lines)
        else:
            yield from cls.text(lines)

    @staticmethod
    def text(lines):
        for line in lines:
            yield dict(text=line.rstrip('\r\n'))

    @staticmethod
    def json(lines):
        for line in lines:
            yield json.loads(line)


# Command lines

@click.group()
@click.argument('predictor', type=PredictorSpec())
@click.option('-v', '--verbose', default=0, count=True,
              help='How much human-readable detail to print to STDERR.')
@click.option('-t', '--train/--no-train', default=False,
              help='Train the model on lines of text after predictions have'
              ' been given for them (and any others with the same timestamp),'
              ' resetting for each userId. This is mainly useful for dynamic'
              ' modelling experiments with the json corpus format.')
@click.option('-f', '--format', default='auto', type=InputFormat(),
              help='Format for test data. text is just lines of plain text;'
              ' json is our json-lines corpus format with rows like'
              ' {"text": "A line of text", "user": "user1234"},'
              ' grouped (e.g. ordered) by userId. Default "auto" is to try to'
              ' auto-detect the format, based on the first line of the log.')
@click.option('-o', '--options', type=common.JsonParam(), default='{}',
              help='Additional JSON-formatted options to be parsed and'
              ' passed to a Python module predictor, or converted to command'
              ' line arguments for a shell predictor. In the case of shell'
              ' - the arguments are passed as "--key value", unless key'
              ' already starts with a hyphen (in which case just "key value"),'
              ' and an optional list of arguments with the key "positional".'
              ' For example: \'{"abc": 123, "-n" 10,'
              ' "positional": ["hello", "world"]}\' will be passed as'
              ' `<cmd> hello world -n 10 --abc 123`.')
@click.pass_context
def cli(ctx, verbose, predictor, train, format, options):
    '''Run a challenge for a predictor over some test text.
    Pipe in text to record an evaluation run of a pipeable predictor, on a
    language modelling task.

    Analyse the output by piping it into `lmchallenge.stats` or
    `lmchallenge.pretty`.

    PREDICTOR - either a shell command to run a pipeable predictor
    e.g. "./my-predictor model.lm",
    or the qualified name of a Python class or function
    e.g. "mymodule.MyClass".
    '''
    common.verbosity(verbose)

    def _runner(tokenizer, evaluate):
        with predictor(options) as model:
            common.dump_jsonlines(run_tokens(
                model=model,
                data=format(sys.stdin),
                train=train,
                tokenizer=tokenizer,
                evaluate=evaluate,
            ))
    ctx.obj = ctx.obj or {}
    ctx.obj['run'] = _runner


@cli.command('wc')
@click.option('-p', '--next-word-only/--no-next-word-only',
              default=False,
              help='Only compute next-word-predictions - don\'t produce'
              ' results for prefix completions (for performance)')
@click.pass_context
def cli_wc(ctx, next_word_only):
    '''Word Completion Challenge (next-word prediction & completion).
    '''
    ctx.obj['run'](
        tokenizer=common.WORD_TOKENIZER,
        evaluate=ft.partial(
            evaluate_completions,
            next_word_only=next_word_only)
    )


@cli.command('we')
@click.pass_context
def cli_we(ctx):
    '''Word Entropy Challenge.
    '''
    ctx.obj['run'](
        tokenizer=common.WORD_TOKENIZER,
        evaluate=evaluate_entropy,
    )


@cli.command('wr')
@click.argument('vocab', type=click.File('r'))
@click.option('-r', '--seed', default=42,
              help='Random seed to fix (default: fixed), or 0 to get a'
              ' pseudorandom seed from the clock')
@click.option('-n', '--num-candidates', default=100,
              help='Number of candidates to consider for each word.')
@click.option('--error-chars', default=errors.DEFAULT_CONFIG['error_chars'],
              help='Set of characters to sample from when adding'
              ' global character errors')
@click.option('-p', '--p-anykey', default=errors.DEFAULT_CONFIG['p_anykey'],
              help='Probability of substituting an \'anykey\' error')
@click.pass_context
def cli_wr(ctx, vocab, seed, num_candidates, error_chars, p_anykey):
    '''Word Reranking Challenge.
    '''
    rand = random.Random(seed)
    words = set(line.rstrip('\r\n') for line in vocab)
    error_config = dict(error_chars=error_chars, p_anykey=p_anykey)
    search = errors.Search(words=words)
    ctx.obj['run'](
        tokenizer=common.WORD_TOKENIZER,
        evaluate=ft.partial(
            evaluate_reranking,
            error_config=error_config,
            search=search,
            num_candidates=num_candidates,
            rand=rand,
        ),
    )


@cli.command('ce')
@click.pass_context
def cli_ce(ctx):
    '''Character Entropy Challenge.
    '''
    ctx.obj['run'](
        tokenizer=common.CHARACTER_TOKENIZER,
        evaluate=evaluate_entropy,
    )


__doc__ += common.shell_docstring(cli, 'lmc run')
__doc__ += common.shell_docstring(cli_wc, 'lmc run wc')
__doc__ += common.shell_docstring(cli_we, 'lmc run we')
__doc__ += common.shell_docstring(cli_wr, 'lmc run wr')
__doc__ += common.shell_docstring(cli_ce, 'lmc run ce')
if __name__ == '__main__':
    cli()
