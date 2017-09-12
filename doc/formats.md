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

Each LM Challenge game has a slightly different log format, although they are similar, both based upon UTF-8 encoded [jsonlines](http://jsonlines.org/).

Each format has the same top-level form, containing one entry for a group of same-timestamp lines from a single user (if users and timestamps were identified in the input).

     {"userId": ID, "timestamp": NUMBER, "trainingChars": NUMBER, ...}

Where all fields are optional (i.e. `userId` & `timestamp` in the absence of marked-up input, and `trainingChars` in the absence of dynamic learning).

_For data privacy and storage space reasons, there are various levels of detail for each format, which may be controlled using switches to the challenge game being run._

### `wp` logs

Log lines from `wp` contain an additional key `wordPredictions`, which maps to one of the following structures (depending on privacy and verbosity settings):

    1. [{"score": NUMBER, "rank": NUMBER, "target": STRING}...]
    2. [{"score": NUMBER, "rank": NUMBER, "targetChars": NUMBER}...]
    3. [{"score": NUMBER, "rank": NUMBER, "target": STRING, "predictions": [ [STRING, NUMBER]...]}...]

In every format, `rank` provides the fundamental analysis mechanism (if present, the correct word was predicted, if absent it was not). Format `1` provides a moderate level of information for analysis - including the true/correct word. Format `2` provides privacy for the test data by just containing the number of characters in the word. Format `3` provides the most debugging/analysis information - including all the predictions received from the model being tested.

For example, using format `1`, log lines might take the form (with line breaks added for readability, and the newline required by jsonlines replaced with `<NEWLINE>`):

    {"userId": "aaa", "timestamp": 10000, "trainingChars": 0,
     "wordPredictions": [{"score": -1, "rank": 2, "target": "Hi"},
                         {"target": "Amelia"},
                         {"score": -3, "rank": 14, "target": "yeah"}]}<NEWLINE>
    {"userId": "aaa", "timestamp": 20000, "trainingChars": 20,
     "wordPredictions": [{"target": "Gonna"},
                         {"score": -2, "rank": 9, "target": "go"}]}<NEWLINE>

### `tc` logs

Log lines from `tc` contain an additional key `textCompletions`, which maps to one of the following structures (depending on privacy settings):

    1. [{"score": NUMBER, "rank": NUMBER, "target": STRING}...]
    2. [{"score": NUMBER, "rank": NUMBER, "targetChars": NUMBER}...]

In every format, `rank` provides the fundamental analysis mechanism (if present, the target sequence was predicted, if absent it was typed character-by-character). Format `1` provides a full information for analysis - including the true/correct text. Format `2` provides privacy for the test data by just containing the number of characters in the text that was typed or predicted.

For example, using format `1`, log lines might take the form (with line breaks added for readability, and the newline required by jsonlines replaced with `<NEWLINE>`):

    {"userId": "aaa", "timestamp": 10000, "trainingChars": 0,
     "textCompletions": [{"target": "H"},
                         {"score": -3, "rank": 1, "target": "ello"},
                         {"target": " Amel"},
                         {"score": -3, "rank": 1, "target": "ia"}]}<NEWLINE>
    {"userId": "bbb", "timestamp": 9000, "trainingChars": 0,
     "textCompletions": [{"target": "Geeza, a"},
                         {"score": -5, "rank": 2, "target": "re"}]}<NEWLINE>

### `ic` logs

Log lines from `ic` contain an additional key `inputCorrections`, which maps to one of the following structures (depending on privacy settings):

    1. [{"score": [NUMBER, NUMBER|null], "target": STRING, "verbatim": STRING, "candidates": [ [STRING, NUMBER, NUMBER|null]...]}...]
    2. [{"score": [NUMBER, NUMBER|null], "targetChars": NUMBER, "verbatimMatch": BOOLEAN, "candidates": [ [NUMBER, NUMBER|null]...]}...]

Note that these formats don't provide quite as much high-level information as `wp` and `tc` - instead `ic` simply includes the scores emitted from the language model and the error evaluation model - it makes no attempt to combine these scores, or calculate ranks / hits. The reason for this is that there is no prior assumption for how to combine the two scores. The `score` for the target is given as `[error_score, language_score]`, and for candidates either format `1: [candidate_text, error_score, language_score]`, or format `2: [error_score, language_score]`, as per `score` (if privacy settings do not allow the text to be included). An evaluation script will likely first combine the pairs of scores using some scheme, before calculating the rank of the correct candidate under that scheme (this allows the evaluation script to try multiple schemes and select the best).

For example, using format `1`, log lines might take the form (with line breaks added for readability, and the newline required by jsonlines replaced with `<NEWLINE>`):

    {"userId": "aaa", "timestamp": 10000, "trainingChars": 0,
     "inputCorrections": [{"score": [-5, -2], "target": "Hi", "verbatim": "Ho", "candidates": [
                              ["Ho", 0, -9], ["Hop", -4, -4], ["Hi", -5, -2]
                          ]},
                          {"score": [-7, -10], "target": "Amelia", "verbatim": "Amlua", "candidates": [
                              ["Amelia", -7, -10], ["Amlua", 0, null],
                          ]}]}<NEWLINE>
