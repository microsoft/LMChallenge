# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import sys
import subprocess
import shlex

__doc__ = '''
Core LM Challenge APIs 'runner' for LMC. The core 'Model' duck-API is as
follows::

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


class BaseModel:
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

    def run_loop(self, input_stream=sys.stdin,
                 output_stream=sys.stdout, error_stream=sys.stderr):
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
                results = self.predict(context, candidates)
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


class ShellModel(BaseModel):
    '''Defines a language model that communicates over a Unix pipe'''

    def __init__(self, cmd, opts):
        '''Open a pipe to the model'''
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
