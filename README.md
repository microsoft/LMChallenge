# Language Model Challenge (LM Challenge)

_A library & tools to evaluate predictive language models._ This is a guide for users of LM Challenge; you may also want to see:

 - [data formats](doc/formats.md) for integrators
 - [dev notes](doc/dev.md) for developers wishing to extend LM Challenge


## What is LM Challenge for?

It is hard to compare language model performance in general. Some models output probabilities, others scores; some model words, others morphemes, characters or bytes. Vocabulary coverage varies. Comparing them in a fair way is therefore difficult. So in LM Challenge we have some very simple 'challenge games' that evaluate (and help compare) language models over a test corpus.

LM Challenge is for researchers and engineers who wish to set a standard for fair comparison of very different language model architectures. It requires a little work to wrap your model in a standard API, but we believe this is often better than writing & testing evaluation tools afresh for each project/investigation.

Note: most of LM Challenge tools are word-based (although all can be applied to sub-word "character compositional" word models). Additionally, our assumption is that the language model is "forward contextual" - so it predicts a word or character based only on preceding words/characters.


## Getting Started

Install LM Challenge from the published Python package:

    pip3 install --user lmchallenge

(Or from this repository `python3 setup.py install --user`.)

**Setup:** LM Challenge needs a model to evaluate. We include an example ngram model implementation in `sample`. Download data & build models (this may take a couple of minutes):

    cd sample/
    ./prepare.sh

**Model REPL:** Now you can use the example script to evaluate a very basic ngram model (see [ngram.py](sample/ngram.py), which you may find useful if integrating your own prediction model). _Note that this command will not terminate, as it launches an interactive program:_

    python3 ngram.py words data/words.3gram

This starts an interactive program which can accept commands of a single word followed by a hard `TAB` character and any arguments, for example:

    > predict<TAB>
    =    0.0000    The    -1.0000    In    -2.0000...

This produces start-of-line predictions, each with an attached score. To query with word context, try the following (making sure you leave a trailing space at the end of the query, after "favourite"):

    > predict<TAB>My favourite 
    of    0.0000    song    -1.0000    the    -2.0000...

This provides next-word-prediction based on a context. There is more to the API (see [formats](doc/formats.md) for more details), but since you won't usually be using the API directly, let's move on to running LM Challenge over this model (so exit the predictor using `Ctrl+D`, back to your shell).

**Evaluation:** To run LM Challenge for this model, we'll pipe some text into `lmc run`, and save the result:

    mkdir out
    head -n 10 data/wiki.test.tokens | lmc run "python3 ngram.py words data/words.3gram" wc > out/w3.wc.log

The resulting log contains all of the original text, and can be queried using the `lmc` utilities. Note: `jq` here is optional, but a very convenient program for working with JSON.

    lmc stats out/w3.wc.log | jq .

You should see some statistics about the model - in particular `completion` & `prediction`. Now let's try comparing with a less powerful model:

    head -n 10 data/wiki.test.tokens | lmc run "python3 ngram.py words data/words.2gram" wc > out/w2.wc.log
    lmc stats out/*.wc.log | jq .

The aggregated level prediction and completion stats should be slightly different for the two models. But we can get a better picture from inspecting the logs in detail:

    lmc pretty out/w3.wc.log

This shows a pretty-printed dump of the data, according to how well the model performed on each token. We can also pretty-print the difference between two models:

    lmc diff out/w3.wc.log out/w2.wc.log

Filter the log for only capitalized words, and print summary statistics:

    lmc grep "^[A-Z][a-z]+$" out/w3.wc.log | lmc stats | jq .

You should notice that capitalized words are (in this small, statistically insignificant example), much harder to predict than words in general.

**Other challenges:** Other LM challenges can be run & inspected in a similar way, see `lmc run --help`.


## Running LM Challenge

LM Challenge is quite flexible - it can be used in a variety of ways:

 1. Command Line Interface
 2. Python API
 3. Log file format

### 1. Command Line Interface

This is the simplest way of using LM Challenge, and works if your model is implemented in any language supporting piped stdout/stdin. See the [Getting Started](#getting-started) guide above, and the CLI help:

    lmc --help
    lmc run --help

### 2. Python API

If your model runs in Python 3, and you wish to script evaluation in Python, you can use the API directly:

    import lmchallenge as lmc
    help(lmc)

Our documentation (as in `help(lmc)`) includes a tutorial for getting started with Python. We don't yet publish the HTML, but it has been tested with `pdoc`:

    $ pdoc --http
    # use your browser to view generated documentation

### 3. Log file format

If you require batching or distribution for sufficient evaluation speed, you can write the LM Challenge log files yourself. This means you can use LM Challenge to process & analyse the data, without imposing a particular execution model. To do this:

 1. Write JSONlines files that contain lmchallenge log data:
    - See [data formats](doc/formats.md) notes that describe the log format.
    - (Optionally) use the [JSON schema](lmchallenge/log.schema) that formally describes an acceptable log datum.
    - (Optionally) use the CLI `lmc validate` (or Python API `lmchallenge.validate.validate`) to check that your log conforms to the schema.
    - Note that log files can often be concatenated if they were generated in parallel.
 2. Use the lmchallenge tools to analyse the logs (everything except `lmc run`).


## The details

An _LM challenge game_ is a runnable Python module that evaluates one or more _language models_ on some task, over some _test text_.

The **challenge games** we have are:

 - `wc` - Word Completion Challenge - a Next Word Prediction / Completion task (generates Hit@N & completion ratios)
 - `we|ce` - Word|Character Entropy Challenges - a language probability distribution task (generates cross entropy given a defined vocabulary)
 - `wr` - Word Reranking Challenge - a correction task (generates accuracy)

**Test text** is pure text data (as typed & understood by real actual humans!) LM Challenge does not define test text - we expect it to be provided. This is the other thing you need to decide on in order to evaluate a _language model_.

A **language model** is an executable process that responds to commands from a _LM challenge game_ in a specific text format, usually comprising of a pre-trained model of the same language as the _test text_.

### Word Completion `wc`

The Word Completion task scans through words in the test text, at each point querying the language model for next-word predictions & word completions.

    cat DATA | lmc run "PREDICTOR" wc > LOG

The model should aim to predict the correct next word before other words (i.e. with as low a rank as possible), or failing that to predict it in the top two completions, given as short a typed prefix as possible. Statistics available from `wc` include:

 - next-word-prediction
   - `Hit@N` - ratio of correct predictions obtained with rank below `N`
   - `MRR` (Mean Reciprocal Rank) - the sum total of `1/rank` over all words
 - completion
   - `characters` - ratio of characters that were completed (e.g. if typing `"hello"`, and it is predicted after you type `"he"`, the ratio of completed characters would be `0.5`)
   - `tokens` - ratio of tokens that were completed before they were fully typed

Note that the flag `--next-word-only` may be used to speed up evaluation, by skipping all prefixes, only evaluating the model's next-word-prediction performance (so that completion stats are not generated).

### Word/Character Entropy `we|ce`

The Word/Character Entropy task produces stats that are analogous to the standard cross-entropy/perplexity measures used for evaluating language models. These evaluators scan through text, at each point querying the language model for a normalized log-probability for the current word.

    cat DATA | lmc run "PREDICTOR" we > LOG
    cat DATA | lmc run "PREDICTOR" ce > LOG

It is important to note that the entropy metric can only be compared between models that share a common vocabulary. If the vocabulary is different, the entropy task is different, and models should not be compared. Therefore, a model must generate a "fair" normalized log-probability over its vocabulary (and if a word is not in the vocabulary, to omit the score from the results). It should not merge "equivalence classes" of words (except by general agreement with every other model being evaluated). An example of this would be example normalizing capitalization to give "fish" the same score as "Fish", or giving many words an "out of vocabulary" score (such that, if you were to calculate `p("fish") + p("Fish") + p(everything else)` it would not sum to one). Simply ommiting any words that cannot be scored (e.g. OOV words) is safe, as this contributes to a special "entropy fingerprint", which checks that two models successfully scored the same set of words, and are therefore comparable under the entropy metric.

### Word Reranking `wr`

The Word Reranking task emulates a sloppy typist entering text, using the language model to correct input after it has been typed. This challenge requires a list of words to use as correction candidates for corrupted words (which should be a large set of valid words in the target language.) Text from the data source is first corrupted (as if by a sloppy typist). The corrupted text is fed into a search for nearby candidate words, which are scored according to the language model under evaluation. The evaluator measures corrected, un-corrected and mis-corrected results.

    cat DATA | lmc run "PREDICTOR" wr VOCAB > LOG

The aim of the model is to assign high score to the correct word, and low score to all other words. We evaluate this by mixing the score from the language model with an _input score_ for each word (using a minimum score for words that are not scored by the lanugage model), then ranking based on that. If the top-ranked prediction is the correct word, this example was a success, otherwise it counts as a failure. The _input score_ is the log-probability of the particular corrupted text being produced from this word, in the same error model that was used to corrupt the true word. In order to be robust against different ranges of scores from language models, we optimize the _input_ and _language model_ mixing parameters before counting statistics (this is done automatically, but requires the optional dependency `scipy`). The accuracy aggregate measures the maximum proportion of correct top predictions, using the optimum mixing proportions.


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
