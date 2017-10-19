# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Main entry point into the evaluation utilities.
'''

import click
import json
import sys
import logging
import regex
import itertools as it
import functools as ft
from .core import common, runner


# Helpers

def find_tokens(text, tokenizer):
    '''Return a generator of {"token", "character", "target"} dictionaries by
    tokenizing some text.

    text -- string -- the text to tokenize

    tokenizer -- regex.Regex -- the tokenizer to use on the text (supporting
                                ``finditer``)

    returns -- a generator of tokens, which are:
        token -- index of the token in the message
        character -- index of the first character of the token in the message
        target -- the text content of the token itself
    '''
    for i, m in enumerate(tokenizer.finditer(text)):
        yield dict(token=i, character=m.span()[0], target=m.group())


def get_completions(model, context, target):
    '''A generator of completions from a model, with successive "typed" prefixes.

    model -- core.runner.BaseModel

    context -- string -- the text before the target

    target -- string -- the token being typed

    returns -- a generator of lists of string predictions, at each point
               while typing out ``target``
    '''
    for i in range(len(target)):
        yield [c[0] for c in model.predict(context + target[:i], None)]


def get_cost(model, context, target):
    '''Wrap the model.predict API to query the score of a single target.
    '''
    results = list(model.predict(context, [target]))
    if 2 <= len(results):
        logging.warning('multiple results returned for a single candidate')
    try:
        return next(r[1] for r in results if r[0] == target)
    except StopIteration:
        return None


def dump_jsonl(data):
    '''Dump data to stdout in jsonlines format.

    data -- iterable of dict -- data to dump
    '''
    for d in data:
        sys.stdout.write(json.dumps(d, sort_keys=True) + "\n")


def evaluate_completions(model, context, target, next_word_only):
    '''Evaluator for Word Completion.
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
    return dict(cost=get_cost(model=model, context=context, target=target))


def run_tokens(model, data, train, tokenizer, evaluate):
    '''Run a per-token evaluation.

    model -- core.runner.BaseModel

    data -- an iterable of "message" dictionaries, containing:
        text -- string -- the contents of the message
        user -- string (optional) -- a user ID

    train -- bool -- should we train after every message?

    tokenizer -- regex.Regex -- tokenizer for finding tokens in messages

    evaluate -- callable(model, context, target) -> dict -- run evaluation
                for a single target

    returns -- a generator of result dictionaries, containing:
        user -- string -- from the message
        message -- int -- index of message
        token -- int -- index of token within the message
        character -- int -- index of character within the message
        target -- string -- token being typed
    '''
    data_by_user = it.groupby(data, lambda x: x.get('user', object()))
    for _, messages in data_by_user:
        if train:
            model.clear()
        for message_n, message in enumerate(messages):
            for token in find_tokens(text=message['text'],
                                     tokenizer=tokenizer):
                yield dict(
                    user=message.get('user'),
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


# Command line helpers

class PredictorSpec(click.ParamType):
    '''Loads a predictor, either from a Python module or a shell command.
    '''

    class PythonModel:
        def __init__(self, ctor):
            self.ctor = ctor

        def __call__(self, options):
            return self.ctor(options.copy())

    class ShellModel:
        def __init__(self, cmd):
            self.cmd = cmd

        def __call__(self, options):
            return runner.ShellModel(self.cmd, options)

    name = 'predictor_spec'

    def convert(self, value, param, ctx):
        if common.is_qualified_name(value):
            return self.PythonModel(common.lookup_qualified_name(value))
        else:
            return self.ShellModel(value)

    def get_metavar(self, param):
        return 'SPEC'


class InputFormat(common.ParamChoice):
    '''Input handling - text or json,
    '''
    name = 'input_format'
    choices = ['auto', 'text', 'json']

    @staticmethod
    def is_json(line):
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
        lines = iter(lines)
        try:
            first_line = next(lines)
            # rebuid an iterable of all the lines
            all_lines = it.chain([first_line], lines)
            yield from (cls.json(all_lines)
                        if cls.is_json(first_line) else
                        cls.text(all_lines))
        except StopIteration:
            # An empty iterable, generate an empty sequence
            pass

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
def cli(ctx, predictor, train, format, options):
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
    def _runner(tokenizer, evaluate):
        with predictor(options) as model:
            dump_jsonl(run_tokens(
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
        tokenizer=common.TOKENIZER,
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
        tokenizer=common.TOKENIZER,
        evaluate=ft.partial(evaluate_entropy),
    )


@cli.command('ce')
@click.pass_context
def cli_ce(ctx):
    '''Character Entropy Challenge.
    '''
    ctx.obj['run'](
        tokenizer=regex.compile('.'),
        evaluate=ft.partial(evaluate_entropy),
    )


__doc__ += common.shell_docstring(cli, 'lmc run')
__doc__ += common.shell_docstring(cli_wc, 'lmc run wc')
__doc__ += common.shell_docstring(cli_we, 'lmc run we')
__doc__ += common.shell_docstring(cli_ce, 'lmc run ce')
if __name__ == '__main__':
    cli()
