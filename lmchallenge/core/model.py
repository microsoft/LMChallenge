# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import sys
import subprocess
import shlex
import regex
import itertools as it
from . import common

__doc__ = '''Core LM Challenge Model APIs for LMC.

`lmchallenge.core.model.Model` documents the core API, which can be
implemented by subclassing, or by duck-typing the same API.
    '''


class Model:
    '''Base class for implementing the Model API for LM Challenge.

    **Subclasses must implement:**

      - `lmchallenge.core.model.Model.predict`

    **Optional:**

      - `lmchallenge.core.model.Model.train`
      - `lmchallenge.core.model.Model.clear`
      - `__enter__`
      - `__exit__`
    '''

    def predict(self, context, candidates):
        '''Get text completions (or score candidates) following a context.

        `context` -- `string` -- preceding text that should be treated as
                     fixed, all predictions and candidates follow this.

        `candidates` -- `list(string)` or `None` -- optional candidates to
                        score following `context`.
                        If `None`, the model should generate the most likely
                        candidates itself to score, otherwise it need only
                        return results from this list (with scores).

        `return` -- `list((string, float))` -- ordered list of result
                    completions & scores.
                    The list should be ordered from most to least likely -
                    in general, that should correspond to descending score
                    order.
                    In some cases (e.g. for computing entropy), it is
                    important that the score is a normalized log-probability.
                    Each completion (as candidates) should be treated as if it
                    follows `context` directly (i.e. if `context` stops in the
                    middle of a word, the results are completions of that
                    word).

        For example (e.g. 1):

            predict("I am the", None)
            -> [("re", -0.5), ("y", -4)]

        This means the model thinks the completion "the" -> "there" is most
        likely, followed by the completion "they".

        For example (e.g. 2):

            predict("I am ", ["they", "the", "inevitably"])
            -> [("the", -0.5), ("they", -4)]

        This means the model has been asked to score three words, of which:

        - "the" is the most likely, with a high score
        - "they" is unlikely, with a low score
        - "inevitably" is out-of-vocab, so is not scored & returned at all

        '''
        raise NotImplementedError

    def train(self, text):
        '''Adapt the model to the current user, providing text that has been
        entered.

        `text` -- `string` -- the contents of a single message, for the current
                  user

        User-specific language information should be aggregated across `train`
        calls until `clear` is called.

        (Default implementation: do nothing)
        '''
        pass

    def clear(self):
        '''Reset all trained state in the model (i.e. we're about to start with
        a new user).

        After calling `clear`, the results from `predict` should be as if the
        `Model` has been newly created.

        (Default implementation: do nothing)
        '''
        pass

    def __enter__(self):
        '''Default implementation.'''
        return self

    def __exit__(self, type, value, traceback):
        '''Default implementation: do nothing.'''
        pass

    def run_loop(self,
                 input_stream=sys.stdin,
                 output_stream=sys.stdout,
                 error_stream=sys.stderr):
        '''Run the model as a pipeable predictor process between
        `input_stream` and `output_stream`.

        `input_stream` -- `stream` -- commands to run (default: STDIN)

        `output_stream` -- `stream` -- prediction output (default: STDOUT)

        `error_stream` -- `stream` -- error message output (default: STDERR)

        This method does not return until `input_stream` is exhausted.

        E.g.

            class MyModel(Model):
                ...

            if __name__ == '__main__':
                model = MyModel()
                model.run_loop()
        '''
        for line in input_stream:
            parts = line.strip('\n').split('\t')
            cmd = parts[0]
            if cmd == 'predict':
                context = parts[1] if 2 <= len(parts) else ''
                candidates = parts[2:] if 3 <= len(parts) else None
                results = ((x, s)
                           for x, s in self.predict(context, candidates)
                           if s is not None)
                response = '\t'.join('%s\t%f' % (candidate, score)
                                     for candidate, score in results) + '\n'
                output_stream.write(response)
            elif cmd == 'train':
                self.train(parts[1])
            elif cmd == 'clear':
                self.clear()
            else:
                error_stream.write('Unrecognized command "' + cmd + '"\n')
            output_stream.flush()


class WordModel(Model):
    '''Optional helper subclass for defining a word-by-word prediction model,
    based on a regex tokenizer.

    **Subclasses must implement:**

      - `lmchallenge.core.model.WordModel.predict_word`
      - `lmchallenge.core.model.WordModel.score_word`

    **Optional:**

      - `lmchallenge.core.model.WordModel.train_word`
    '''
    def __init__(self, token_pattern=None):
        if token_pattern is None:
            self._tokenizer = common.WORD_TOKENIZER
        else:
            self._tokenizer = regex.compile(token_pattern)

    def predict(self, context, candidates):
        tokens = list(self._tokenizer.finditer(context))
        if len(tokens) and tokens[-1].end() == len(context):
            # there is an "in-progress" word
            prefix = tokens.pop(-1).group(0)
        else:
            prefix = ''
        context_tokens = [m.group(0) for m in tokens]

        if candidates is None:
            return self.predict_word(context_tokens, prefix)
        else:
            return self.score_word(
                context_tokens,
                [prefix + candidate for candidate in candidates])

    def predict_word(self, context, prefix):
        '''Predict a next word, or complete the current word.

        `context` -- `list(string)` -- word tokens in the context

        `prefix` -- `string` -- the prefix of the word being typed (if empty,
                   returns next word predictions)

        `return` -- `list((string, float))` -- a list of pairs (suffix, score)
        '''
        raise NotImplementedError

    def score_word(self, context, candidates):
        '''Score a set of candidates which follow a context.

        `context` -- `list(string)` -- word tokens in the context

        `candidates` -- `set(string)` -- should return scores for each of
                        these, if possible

        `return` -- `list((string, float))` -- a list of pairs
                    (candidate, score)
        '''
        raise NotImplementedError

    def train(self, text):
        return self.train_word(self._tokenizer.findall(text))

    def train_word(self, text):
        '''Add this sequence of words to a user-adaptive model.

        `text` -- `list(string)` -- word tokens in the message

        (Default implementation: do nothing.)
        '''
        pass


class FilteringWordModel(WordModel):
    '''Specialization of WordModel, which automatically filters prefixes
    & limits results.

    **Subclasses must implement:**

      - `lmchallenge.core.model.FilteringWordModel.predict_word_iter`
      - `lmchallenge.core.model.WordModel.score_word`
    '''
    def __init__(self, n_predictions, filter_pattern='.', **args):
        '''Create a filtering word model.

        `n_predictions` -- `int` -- how many predictions/completions to return
                           (does not apply when scoring candidates).

        `filter_pattern` -- `string` -- a regex pattern to apply to filter
                            results (does not apply when scoring candidates).
                            A result will be allowed if the pattern is matched
                            anywhere in the string.

        `**args` -- see `lmchallenge.core.model.WordModel`
        '''
        super().__init__(**args)
        self.n_predictions = n_predictions
        self.filter_xp = regex.compile(filter_pattern)

    def predict_word_iter(self, context):
        '''As per `lmchallenge.core.model.WordModel.predict_word`, but should
        return a lazy generator/iterator of next words, which may include
        duplicates.

        `context` -- `list(string)` -- list of preceding tokens

        `return` -- `generator(list((string, float)))` -- lazy sequence of
                    pairs (word, score)
        '''
        raise NotImplementedError

    def predict_word(self, context, prefix):
        results = self.predict_word_iter(context)
        filter_results = (
            (w[len(prefix):], s)
            for w, s in results
            if self.filter_xp.search(w) is not None
            and len(prefix) < len(w) and w.startswith(prefix))
        unique_results = common.unique_by(filter_results, lambda e: e[0])
        top_results = it.islice(unique_results, self.n_predictions)
        return list(top_results)


class ShellModel(Model):
    '''Defines a language model that proxies calls to a subprocess,
    over a Unix pipe.
    '''
    def __init__(self, cmd, opts):
        '''Open a pipe to the model.

        `cmd` -- `string` -- shell command to run in a subprocess

        `opts` -- `dict` -- arguments to send to the subprocess.
                  The key `"positional"` can refer to a list of positional
                  arguments.
                  The key `"verbose"` is used to control model verbosity,
                  as well as being sent to the subprocess.
                  All other keys are sent to the subprocess command as
                  `"--KEY VALUE"` (unless they already start with "-", in
                  which case just `"KEY VALUE"`).
        '''
        self.verbose = opts.get('verbose', False)
        cmd_positional = ' '.join(opts.get('positional', []))
        cmd_options = ' '.join(
            '%s %s' % (key if key.startswith('-') else ('--' + key),
                       shlex.quote(str(value)))
            for key, value in opts.items()
            if key != 'positional'
        )
        self.cmd = '%s %s %s' % (cmd, cmd_positional, cmd_options)
        self._debug('$ %s' % self.cmd)
        self.proc = subprocess.Popen(self.cmd, shell=True,
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)
        self.proc_in = open(self.proc.stdin.fileno(), 'w', encoding='utf-8')
        self.proc_out = open(self.proc.stdout.fileno(), 'r', encoding='utf-8')
        self._check_return_code()

    def __exit__(self, type, value, traceback):
        self._close()

    def _debug(self, line):
        if self.verbose:
            sys.stderr.write(line + '\n')

    def _close(self):
        '''Finish with the process & shut it down.'''
        self.proc.communicate()
        self._check_return_code()

    def _check_return_code(self):
        '''Check that our model process hasn't errored out.'''
        rc = self.proc.returncode
        if rc is not None and rc != 0:
            raise subprocess.CalledProcessError(rc, self.cmd)

    def _send_command(self, command):
        '''Issue a tab-delimited command to the model process.'''
        self._debug('#> %s' % (' <TAB> '.join(command)))
        self._check_return_code()
        self.proc_in.write(u'\t'.join(command) + u'\n')
        self.proc_in.flush()

    def predict(self, context, candidates):
        command = ['predict', context]
        if candidates is not None:
            command += candidates
        self._send_command(command)
        self._check_return_code()
        candidates_and_scores = [
            s for s in self.proc_out.readline().strip('\n').split('\t')
        ]
        self._debug('#< %s' % (' <TAB> '.join(candidates_and_scores)))
        pairs = [(candidates_and_scores[i*2],
                  float(candidates_and_scores[i*2+1]))
                 for i in range(len(candidates_and_scores)//2)]
        return pairs

    def train(self, text):
        self._send_command(['train', text])

    def clear(self):
        self._send_command(['clear'])
