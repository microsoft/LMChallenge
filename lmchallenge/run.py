# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

'''Main entry point into the evaluation utilities.
'''

import click
import itertools
import json
import sys
import random
from .core import common, runner, errors


# Programs

class Wp:
    def __init__(self, targets=True, predictions=False):
        self.targets = targets
        self.predictions = targets and predictions

    @staticmethod
    def contexts_and_targets(line):
        '''Return a generator of (context, target) pairs by parsing a line.
        '''
        for m in common.TOKENIZER.finditer(line):
            start, end = m.span()
            target = m.group()
            yield line[:start], target

    def __call__(self, model, line):
        '''Return a stats object for a single line of wp evaluation.
        '''
        result = []
        for context, target in Wp.contexts_and_targets(line):
            word_result = {}
            if self.targets:
                word_result['target'] = target
            else:
                word_result['targetChars'] = len(target)
            predictions_with_scores = model.predict(context, None)
            if self.predictions:
                word_result['predictions'] = predictions_with_scores
            target_score = None
            for word, score in predictions_with_scores:
                if word == target:
                    target_score = score
                    break
            if target_score is not None:
                word_result['rank'] = sum(
                    target_score <= score
                    for word, score in predictions_with_scores
                )
                word_result['score'] = target_score

            result.append(word_result)

        return {'wordPredictions': result}


class Tc:
    DEFAULT_SELECT_FROM = 3

    def __init__(self, targets=True, select_from=DEFAULT_SELECT_FROM):
        self.targets = targets
        self.select_from = select_from

    def __call__(self, model, line):
        '''Return a stats object for a single line of tc evaluation.
        '''
        result = []
        typing = ''

        def flush_typing():
            nonlocal typing
            if len(typing) != 0:
                result.append({'target': typing}
                              if self.targets else
                              {'targetChars': len(typing)})
                typing = ''
        i = 0
        while i < len(line):
            context = line[:i]
            tail = line[i:]

            predictions = model.predict(context, None)[:self.select_from]
            matches = [(prediction, rank, score)
                       for rank, (prediction, score) in enumerate(predictions)
                       if len(prediction) != 0 and tail.startswith(prediction)]

            if len(matches) == 0:
                # Typing a character
                typing += line[i]
                i += 1
            else:
                # Accepting a prediction
                flush_typing()
                match, match_rank, match_score = max(matches,
                                                     key=lambda m: len(m[0]))
                entry = {'rank': match_rank,
                         'score': match_score}
                if self.targets:
                    entry['target'] = match
                else:
                    entry['targetChars'] = len(match)
                result.append(entry)
                i += len(match)

        flush_typing()
        return {'textCompletions': result}


class Ic:
    DEFAULT_NUM_CANDIDATES = 100

    def __init__(self, vocab, targets=True,
                 num_candidates=DEFAULT_NUM_CANDIDATES,
                 error_config=errors.DEFAULT_CONFIG):
        self.targets = targets
        self.num_candidates = num_candidates
        self.error_config = error_config
        self.search = errors.Search(words=vocab, n=num_candidates)

    def corruption_candidates(self, target):
        '''Corrupt the candidate, and return a pair ``(corruption, input_scores)``
        with error model scores for each candidate.
        '''

        corrupted = errors.corrupt(self.error_config, target)

        candidates = self.search(corrupted)

        # clip off trailing candidates if needed to ensure corrupted & target
        # can be added - this should keep the maximum size of candidates
        # clamped at 'n' and ensuring that it contains 'target' and 'corrupted'
        candidates = set(
            candidates[:(self.num_candidates - (target not in candidates) -
                         (corrupted not in candidates))]
        ) | set([corrupted, target])

        return corrupted, \
            {c: errors.score(self.error_config, corrupted, c)
             for c in candidates}

    def __call__(self, model, line):
        '''Return a stats object for a single line of ic evaluation.
        '''
        result = []
        for context, target in Wp.contexts_and_targets(line):
            corrupted, input_scores = self.corruption_candidates(target)
            lm_scores = dict(model.predict(context, input_scores.keys()))

            word_result = {}
            word_result['score'] = [input_scores[target],
                                    lm_scores.get(target, None)]
            if self.targets:
                word_result['target'] = target
                word_result['verbatim'] = corrupted
                word_result['candidates'] = [
                    [candidate, input_score, lm_scores.get(candidate, None)]
                    for candidate, input_score in input_scores.items()
                ]
            else:
                word_result['targetChars'] = len(target)
                word_result['verbatimMatch'] = (corrupted == target)
                word_result['candidates'] = [
                    [input_score, lm_scores.get(candidate, None)]
                    for candidate, input_score in input_scores.items()
                ]

            result.append(word_result)

        return {'inputCorrections': result}


# Generic code to run the programs

def run(model, task, data, train=False):
    '''Run an evaluation of ``model`` over some lines of evaluation data.

    ``model`` - an instance of ``runner.BaseModel`` to train & predict

    ``task`` - callable ``task(model, text)`` to return a results object
    for a single line

    ``data`` - a sequence of dictionaries containing at least {"text": "..."}

    ``returns`` - a sequence of dictionaries for each line of text
    '''
    lines_by_user = itertools.groupby(
        data, lambda line: line.get('userId')
    )
    for user_id, user_lines in lines_by_user:
        if train:
            model.clear()
            training_chars = 0

        lines_by_time = itertools.groupby(
            user_lines, lambda line: line.get('timestamp', object())
        )
        for timestamp, same_timestamp_lines in lines_by_time:
            training_buffer = []
            for row in same_timestamp_lines:
                result_row = row.copy()
                text = result_row.pop('text')
                result_row.update(task(model, text))

                if train:
                    training_buffer.append(text)
                    result_row['trainingChars'] = training_chars

                yield result_row

            # When multiple lines of text have the same timestamp
            # (e.g. if they came from the same UsageFragment and we
            # don't know the original order they occurred in), we
            # can't train on one line before predicting another, we
            # have to wait until we've predicted for all lines first.
            if train:
                for line in training_buffer:
                    model.train(line)
                    training_chars += len(text)


class InputFormat(common.ParamChoice):
    '''Input handling - text or json,
    '''
    name = 'input_format'
    choices = ['text', 'json']

    @staticmethod
    def text(line):
        return {'text': line.rstrip('\r\n')}

    @staticmethod
    def json(line):
        return json.loads(line.rstrip('\r\n'))


def run_stream(model, task,
               input=sys.stdin, format=InputFormat.text,
               output=sys.stdout, train=False):
    '''As per ``run``, except wraps model in a ``with``, parses
    input & formats output to file streams.
    '''
    with model as enter_model:
        for line in run(model=enter_model,
                        data=map(format, input),
                        task=task,
                        train=train):
            output.write('%s\n' % json.dumps(line, sort_keys=True))


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


@click.group()
@click.argument('predictor', type=PredictorSpec())
@click.option('-w/-W', '--targets/--no-targets', default=True,
              help='Omit the target words, leaving only the score and rank'
              ' for each target word. This is useful in case you don\'t want'
              ' to disclose the source data, or to reduce size of the output.'
              ' Not compatible with --predictions (in `wp`).')
@click.option('-t', '--train/--no-train', default=False,
              help='Train the model on lines of text after predictions have'
              ' been given for them (and any others with the same timestamp),'
              ' resetting for each userId. This is mainly useful for dynamic'
              ' modelling experiments with the json corpus format.')
@click.option('-f', '--format', default='text', type=InputFormat(),
              help='Format for test data. text is just lines of plain text;'
              ' json is our json-lines corpus format with rows like'
              ' {"text": "A line of text", "timestamp": 1234567,'
              ' "userId": "user1234"}, ordered by userId then timestamp (where'
              ' present; timestamp is optional).')
@click.option('-o', '--options', type=common.JsonParam(), default='{}',
              help='Additional JSON-formatted options to be parsed and'
              ' passed to a Python module predictor, or converted to command'
              ' line arguments for a shell predictor. In the case of shell'
              ' - the arguments are passed as "--key value", unless key'
              ' already starts with a hyphen (in which case just "key value"),'
              ' and an optional list of arguments with the key "positional".'
              ' For example: \'{"abc": 123, "-n" 10,'
              ' "positional": ["hello", "world"]}\' will be passed as'
              ' `<cmd> hello world -n 10 --a 123`.')
@click.pass_context
def cli(ctx, predictor, targets, train, format, options):
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
    ctx.obj = ctx.obj or {}
    ctx.obj['run'] = lambda task: run_stream(predictor(options), task,
                                             format=format,
                                             train=train)
    ctx.obj['targets'] = targets


@cli.command('wp')
@click.option('-p', '--predictions/--no-predictions', default=False,
              help='Include the full list of predictions in the result.')
@click.pass_context
def cli_wp(ctx, predictions):
    '''Word Prediction Challenge.
    '''
    ctx.obj['run'](Wp(targets=ctx.obj['targets'], predictions=predictions))


@cli.command('tc')
@click.option('-n', '--select-from', type=click.INT,
              default=Tc.DEFAULT_SELECT_FROM,
              help='Number of predictions to select text completions from.')
@click.pass_context
def cli_tc(ctx, select_from):
    '''Text Completion Challenge.
    '''
    ctx.obj['run'](Tc(targets=ctx.obj['targets'], select_from=select_from))


@cli.command('ic')
@click.argument('vocab', type=click.File('r'))
@click.option('-r', '--seed', default=42,
              help='Random seed to fix (default: fixed), or 0 to get a'
              ' pseudorandom seed from the clock')
@click.option('-n', '--num-candidates', default=Ic.DEFAULT_NUM_CANDIDATES,
              help='Number of candidates to consider for each'
              ' word \'typed\'.')
@click.option('--error-chars', default=errors.DEFAULT_CONFIG['error_chars'],
              help='Set of characters to sample from when adding'
              ' global character errors')
@click.option('-p', '--p-anykey', default=errors.DEFAULT_CONFIG['p_anykey'],
              help='Probability of substituting an \'anykey\' error')
@click.pass_context
def cli_ic(ctx, vocab, seed, num_candidates, error_chars, p_anykey):
    '''Input Correction Challenge.

    VOCAB - File containing a list of all words to consider as correction
    candidates (this should generally be a long list!)
    '''
    if seed != 0:
        random.seed(seed)
    vocab = set(line.rstrip('\r\n') for line in vocab)
    error_config = dict(error_chars=error_chars, p_anykey=p_anykey)
    ctx.obj['run'](Ic(vocab=vocab, targets=ctx.obj['targets'],
                      num_candidates=num_candidates,
                      error_config=error_config))


__doc__ += common.shell_docstring(cli, 'lmc run')
__doc__ += common.shell_docstring(cli_wp, 'lmc run wp')
__doc__ += common.shell_docstring(cli_tc, 'lmc run tc')
__doc__ += common.shell_docstring(cli_ic, 'lmc run ic')
if __name__ == '__main__':
    cli()
