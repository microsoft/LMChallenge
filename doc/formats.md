# LM Challenge formats and APIs

  1. [Corpus format](#corpus-format)
  2. [Model API](#model-api)
  3. [Log formats](#log-formats)


## Corpus format

Two corpus formats are supported. The first, _plain text_ is designed basic 'flat' evaluation without user-specific training and modelling. The second, _marked-up_ format is designed to provide metadata about the text being evaluated - in particular, a User ID and timestamp, which allows LM Challenge to evaluate user adaptation on sequences of text.

In all cases, input data should be UTF-8 encoded.

### Plain text

The plain text format is simply a line-by-line text format with newline (`U+000A`) as the only delimiter (which should be used to separate paragraphs of potentially unrelated text).

For example:

    This is a line of text. Everything on this line is related.
    Now we have some more, unrelated text, OK.

### User marked-up text

The marked-up text format is based on [jsonlines](http://jsonlines.org/), which is a sequence of newline (`U+000A`)-separated JSON objects. Each line is of the following form (where every key except `text` is optional). The lines should be first grouped by `userId`, then ordered by `timestamp` for each user.

    {"userId": ID, "timestamp": NUMBER, "text": STRING}

For example:

    {"userId": "aaa", "timestamp": 100000, "text": "I'm happy today :-)"}
    {"userId": "aaa", "timestamp": 103000, "text": "Sad today :-("}
    {"userId": "bbb", "timestamp": 102000, "text": "Who do you think you are?"}


## Model API

A language model is an executable process that responds to commands from a _LM challenge game_ in a specific text format, usually comprising of a pre-trained model of the same language as the _test text_. Generally this is specified as a shell command with predefined arguments.

All text format APIs use UTF-8 encoding & only the newline (`U+000A`) & horizontal tab (`U+0009`) as delimiters (represented below as `<TAB>` and `<NEWLINE>`). Care must be taken to ensure streams are flushed at appropriate points (typically after each newline) in order to avoid deadlock.

#### `predict`

`predict` is used to predict the next string of characters given
context. The command specifies a context string (which is a prefix of
a line from the test data).
Optionally, the command also specifies a list of candidates that should
be considered for prediction (if not specified, the model is itself
responsible for generating valid candidates).
Candidates may be multiple words, but should all correspond to the same
amount of input (which should help make the resulting scores comparable).
The language model responds with a list of prediction strings for the
following characters, together with scores for each prediction.

 - Input is an untokenized context string (which may stop abruptly
   within a word, or may end in a space)
 - Output must be a list of next-string predictions (which may be words,
 characters, morphemes or phrases) with a score for each prediction.
 Format is `prediction<TAB>score<TAB>prediction<TAB>score` ....
 - The prediction should simply follow the characters of context (for
 example if the input is `"I like bi"` a prediction might be `"rds"`
 (as if completing the string `"I like birds"`), but not `"birds"`
 (which would be interpreted as suggesting `"I like bibirds"`).
 - Score is a number that's used to determine the ranking of your
 predictions; biggest score ranks first. The predictions need not be
 returned in rank order.
 - In general we make no further assumptions about what the score
 represents -- for example it could be a normalized predictive
 probability or log-probability, an unnormalized probability, the
 output of some non-probabilistic predictive model, or just the
 (reciprocal of the) predicted rank itself.
 - Some tools that operate on evaluation results may make additional
 assumptions about what model scores represent (e.g. that they are
 log-probabilities), but in these case the requirement will be
 documented.
 - If a specified candidate is unscorable in a model, it may be omitted
 from the results, in which case the treatment of that candidate is
 dependent on the evaluator.

For example:

    predict<TAB>I am your <NEWLINE>
	best<TAB>0.2<TAB>only<TAB>0.1<TAB>friend<TAB>0.08<TAB>Boss<TAB>0.05<NEWLINE>

    predict<TAB>I am your gu<NEWLINE>
	est<TAB>-1.23<TAB>ess<TAB>-2.51<TAB>y<TAB>-2.82<TAB>errilla<TAB>-6.33<NEWLINE>

    predict<TAB>I am your <TAB>guest<TAB>guerilla<NEWLINE>
    guest<TAB>-1.23<TAB>guerilla<TAB>-6.33<NEWLINE>

#### `train`

`train` allows a model the opportunity to learn from a line of input, after having been evaluated on it.

 - Input: an untokenized line of text
 - Output: none (it is an error to send back even a newline in response to this command)

For example:

    train<TAB>Hey Rebecca, did you see that lacrosster?<NEWLINE>

#### `clear`

`clear` instructs the model to forget everything it has learnt from previous `train` calls. For example, this will be called when the dataset is changing (e.g. evaluating on data from a different user).

 - Input: none
 - Output: none (it is an error to send back even a newline in response to this command)

For example:

    clear<NEWLINE>


## Log formats

All LM Challenge games share a common log schema (defined formally in `log.schema`, which has common metadata and optional payload data for each challenge. Logs should be stored as UTF-8 encoded [jsonlines](http://jsonlines.org/), optionally gzipped.

The required keys for a log event, which typically represents a single word or character from the data, are as follows:

    {"user": STRING,
     "character": NUMBER,
     "message": NUMBER,
     "token": NUMBER,
     "target": STRING}

 - `user` should be a unique identifier for that user (or `null`, if there is no user information)
 - `character` is the index of the start of the source data range, relative to the start of the message
 - `message` is the message index within a single user
 - `token` is the token index within a single message
 - `target` is a string from the source data, the text being modelled

LM Challenge logs should be sorted by `(user, message, token)` - i.e. all events for a single user should be contiguous, and message & token should be in ascending order for that user.

Note that LM Challenge logs contain the original source text, so should be subject to the same privacy constraints & data protection classification.

### `wc` logs

Log lines from `wc` contain an additional key `completions`, which records next-word-predictions and prefix completions for the target word.

    {"completions": [[STRING, STRING, ...],
                     [STRING, STRING, ...],
                     ...]}

Completions is a jagged 2D array of completions, such that `completions[i][j]` corresponds to the suffix predicted after typing `i` characters at prediction index `j`. For example, if the target is `"Hello"`, the completions array might be:

    [["Good", "Hi", "Are"],
     ["i", "ow", "e"],
     ["llo", "lp", "lpful"],
     ["lo", "p", "pful"],
     ["o", "enistic"]]

I.e. the second row corresponds to the predictions `["Hi", "How", "He"]` (or, the whole word is reconstructed using `target[:i] + completions[i][j]`).

If running `wc` in "fast" mode, only `completions[0][:]` is present - as this corresponds to zero characters of prefix, which is next-word-prediction.

### `we` & `ce` logs

Log lines from `we` or `ce` contain an additional key `logp`, which records the log probability of this word or character target, or `null` if the target is not in the language model vocabulary. It is the responsibility of the model/evaluator to ensure that `logp` is normalized over the vocabulary (therefore it should, in general, be negative).

### `wr` logs

Log lines from `wr` contain the additional keys `verbatim`, which records the most likely corruption and  `results`, which records corruption/correction candidates and scores (both from the language model, and the true error model).

    {"verbatim": STRING,
     "results": [[STRING, NUMBER, NUMBER|NULL],
                 ...]}

Each entry in results is an evaluated candidate, with a candidate string (which may be the same as the target or the verbatim), error model score and language model score. For example if the target is `"can"`:

    {"verbatim": "caj",
     "results": [["caj", 0.0, null],
                 ["can", -3.0, -2.5],
                 ["cab", -3.0, -2.8],
                 ["fab", -6.0, -3.2]]}

In this way, the list of results should include a candidate for the verbatim, and a candidate for the true target, as well as a number of other candidates, which are found by the evaluator to be likely given the corruption, and are included to confuse a language model, forcing it to disambiguate the true target.

After a error-LM mixture model has been fitted to the log, an additional element is appended to each array, containing the combined score from the combined model (which should not be null). This is the final sort order of candidates.
