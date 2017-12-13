# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import sys
import subprocess
import shlex
import regex
import itertools as it
from . import common

__doc__ = '''
Core LM Challenge APIs for LMC. The core 'Model' duck-API is as follows::

    # Get text completions following a context.
    # input: context - text before prediction (string)
    #        candidates - candidates to consider (list of strings, or None)
    # output: candidate_scores - to follow context (list of (string, float))
    candidate_scores = model.predict(context, candidates)

    # Adapt the model to the user, providing text that has been entered
    # input: text - text from the current user (string)
    model.train(text)

    # Reset all trained state in the model (i.e. we're about to start with
    # a new user)
    model.clear()

    # In addition, the model should be usable in Python's 'with' statement:
    model.__enter__()
    model.__exit__(type, value, traceback)
    '''


class Model:
    '''Base class for implementing the Model API for LM Challenge.
    Subclasses must at least implement
    ``candidates = predict(self, context, candidates)``
    to be a valid predictor.
    '''

    def predict(self, context, candidates):
        raise NotImplementedError

    def train(self, text):
        '''Default implementation: do nothing.'''
        pass

    def clear(self):
        '''Default implementation: do nothing.'''
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
        ``input_stream`` and ``output_stream``. This method does not
        return until ``input_stream`` is closed.
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

    Subclasses should provide:

        predict_word(context, prefix) -> results
        score_word(context, candidates) -> results
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
        """Predict a next word, or complete the current word.

        ``context`` - list of strings - the word tokens in the context

        ``prefix`` - string - the prefix of the word being typed (if empty,
                     returns next word predictions)

        ``return`` - a list of pairs (suffix, score)
        """
        raise NotImplementedError

    def score_word(self, context, candidates):
        """Score a set of candidates which follow a context.

        ``context`` - list of strings - the word tokens in the context

        ``candidates`` - set of strings - should return scores for each of
                         these, if possible

        ``return`` - a list of pairs (candidate, score)
        """
        raise NotImplementedError

    def train(self, text):
        return self.train_word(self._tokenizer.findall(text))

    def train_word(self, text):
        """Add this sequence of words to a user-adaptive model.
        Default implementation: do nothing.

        ``text`` - list of strings - word tokens to learn from
        """
        pass


class FilteringWordModel(WordModel):
    """Specialization of WordModel, which automatically filters prefixes
    & limits results.

    Subclasses must provide:

    score_word(context, candidates) -> results
    predict_word_iter(context) -> results (lazy)
    """
    def __init__(self, n_predictions, filter_pattern='.', **args):
        """Create a filtering word model.

        n_predictions -- how many predictions/completions to return (does not
                         apply when scoring candidates)

        filter_pattern -- a pattern to apply to filter results (does not apply
                          when scoring candidates) - a result will be allowed
                          if the pattern is matched anywhere in the string
        """
        super().__init__(**args)
        self.n_predictions = n_predictions
        self.filter_xp = regex.compile(filter_pattern)

    def predict_word_iter(self, context):
        """As per ``predict_word``, but should return a lazy generator/iterator
        of next words, which may include duplicates.

        context -- list of preceding tokens

        returns -- a list of pairs (word, score)
        """
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
        '''Open a pipe to the model.'''
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
