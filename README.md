# Language Model Challenge (LMChallenge)
[![Build Status](https://travis-ci.com/Microsoft/LMChallenge.svg?token=PsuQKRDL8Qs6yfLsqpTp&branch=master)](https://travis-ci.com/Microsoft/LMChallenge)

A set of tools to evaluate language models for typing.

This is a guide for users of LM Challenge. You may also want to see:

 - [data formats](doc/formats.md) for integrators
 - [dev notes](doc/dev.md) for developers wishing to extend LM Challenge

## What is LM Challenge for?

It is really quite hard to test language model performance. Some models output probabilities, others scores; some model words, others morphemes, characters or bytes. Vocabulary coverage varies. Comparing them in a fair way is tough... So in LM Challenge we have some very simple 'challenge games' that evaluate (and help compare) language models over a test corpus.

## Getting Started

We include support for building a Docker image that makes it easier to get started (which also documents [our dependencies](Dockerfile), in case you want to get set up natively).

```bash
docker build --rm -t lmchallenge .
```

## The Challenges

An _LM challenge game_ is a runnable Python module that evaluates one or more _language models_ on some task, over some _test text_.

The **challenge games** we have are:

 - `wp` - Word Prediction Challenge - a next-word-prediction task (generates Hit@N results)
 - `tc` - Text Completion Challenge - a text completion task (generates KSPC results)
 - `ic` - Input Correction Challenge - a correction task (generates accuracy results)

**Test text** is pure text data (as typed & understood by real actual humans!) LM Challenge does not define test text - we expect it to be provided. This is the other thing you need to decide on in order to evaluate a _language model_.

A **language model** is an executable process that responds to commands from a _LM challenge game_ in a specific text format, usually comprising of a pre-trained model of the same language as the _test text_.

### Word Prediction `wp`

The Word Prediction task scans through words in the test text, at each point querying the language model for next-word predictions.

The aim of the model is to predict the correct next word before other words (i.e. with as low a rank as possible). Statistics available from `wp` include `Hit@N` (ratio of correct predictions obtained with rank below `N`) and rank-weighted scores such as `SRR` (sum reciprocal rank - the sum total of `1/rank` over all words).

`wp` may be used as follows:

```bash
cat data.txt | lmc run "lm ..." wp > wp.log
lmc stats < wp.log
lmc pretty < wp.log
```

The first command creates a log file of the results of running the predictor over the test text. Subsequent commands provide ways of analysing those logs - aggregating summary stats and providing a colorful rendering of the behaviour of the predictor.

### Text Completion `tc`

The Text Completion task emulates a simple typist entering the text perfectly, and selecting predictions whenever possible. The typist queries the model for predictions at the current point. If a prediction below a certain rank matches the text, it is 'selected', and the process continues after the end of the selected prediction. If no match is found, the typist enters the next character from the test text, then repeats the process.

The aim of the model is to predict as much of the next word, part-word or sequence of words as possible. Statistics available from `tc` include `pcpc` (ratio of predictions to characters entered) and `kpc` (number of prediction selection or character typing events per character entered).

`tc` may be used as follows:

```bash
cat data.txt | lmc run "lm ..." tc > tc.log
lmc stats < tc.log
lmc pretty < tc.log
```

### Input Correction `ic`

The Input Correction task emulates a sloppy typist entering text, using the language model to correct input after it has been typed. This challenge requires a list of words to use as correction candidates for corrupted words (which should be a large set of valid words in the target language.) Text from the data source is first corrupted (as if by a sloppy typist). The corrupted text is fed into a search for nearby candidate words, which are scored according to the language model under evaluation. The evaluator measures corrected, un-corrected and mis-corrected results.

The aim of the model is to assign high score to the correct word, and low score to all other words. We evaluate this by mixing the score from the language model with an _input score_ for each word, then ranking based on that - it the top-ranked prediction is the correct word, this example was a success, otherwise it counts as a failure. The _input score_ is the log-probability of the particular corrupted text being produced from this word, in the same error model that was used to corrupt the true word. In order to be robust against different ranges of scores from language models, we must optimize the _input_ and _language model_ mixing parameters before counting statistics: this is done with `lmc ic-opt` (or can be done automatically in `lmc stats`). The JSON output of `lmc ic-opt` is used when running `lmc stats` or `lmc pretty` in order to fully specify the model under investigation.

Statistics available include `accuracy` (ratio of correct words after correction), `miscorrected` (proportion of errors introduced to correct words) and the combined metric `improvement` (ratio change in number of errors).

`ic` may be used as follows:

```bash
cat data.txt | lmc run "lm ..." ic words.txt > ic.log
lmc ic-opt < ic.log > ic.opt
lmc stats -a ic.opt < ic.log
lmc pretty -a ic.opt < ic.log
lmc page -a ic.opt < ic.log > ic.html
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
