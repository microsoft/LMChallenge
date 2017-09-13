# Developing

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

## Tests

These are currently very much incomplete (please help fix this), but they are run with:

    ./scripts/run build
    ./scripts/run -i prod test
    ./scripts/run flake

## Publishing

 1. check that you're happy with `version.txt`
 2. `python3 setup.py bdist_wheel upload -r internal`
 3. `git push origin HEAD:refs/tags/$(cat version.txt)`
 4. update, commit & push `version.txt`

## Documentation

You can build the documentation with the development Docker image.

    ./scripts/run doc

We publish documentation with a Docker-ed server:

    ./scripts/run doc
    docker build -t "lmchallenge-doc:$(cat version.txt)" doc
    docker run --name lmchallenge-doc -d "lmchallenge-doc:$(cat version.txt)"
    docker run -d -p 2873:80 --name lmchallenge-doc "lmchallenge-doc:$(cat version.txt)"

For now, we don't use an internal registry, but you can send things around manually:

    docker save lmchallenge-doc:3.0 | gzip -c > lmchallenge-doc.tar.gz
    scp lmchallenge-doc.tar.gz ...
    gzip -cd lmchallenge-doc.tar.gz | docker load

## Ideas

### `rank` command

`rank` is used to order a set of candidates given a preceding context. The command specifies a context string, and a set of candidate strings to evaluate. The language model responds with an ordered list of the same candidates. The model is not permitted to add candidates that are not in the specified set. Note that the candidates are not expected to complete the line of text (more text may follow).

 - Input: untokenized context string (which may stop abruptly within a word), unordered list of candidate strings to follow the context
 - Output: the same list of candidates, re-ordered according to the model's likelihood (so that the most likely candidate is first)

For example:

    rank<TAB>I am your <TAB>the<TAB>friend<TAB>unlikely<TAB>brother<NEWLINE>
	friend<TAB>brother<TAB>unlikely<TAB>the<NEWLINE>

    rank<TAB>I am you<TAB>'ve been<TAB>r brother<TAB> and you are me<TAB>.<NEWLINE>
	r brother<TAB>.<TAB> and you are me<TAB>'ve been<NEWLINE>

### `eval` command

`eval` is similar to `rank`, but there is no fixed context, instead the candidates specify entire lines of input that are ranked by the model on likelihood.

 - Input: list of (untokenized) candidate strings for a complete line of text
 - Output: the same list of candidates, re-ordered according to the model's likelihood

For example:

    eval<TAB>I are the best<TAB>Ian the best!<TAB>I am the best.<NEWLINE>
	I am the best.<TAB>Ian the best!<TAB>I are the best<NEWLINE>

### `corrupt` command

`corrupt` is for a slightly different class of models - these are corrupters that are able to generate artificial candidates for the text discrimination game. This may be provided by a language model or something simpler, but we include it here for simplicity. The command specifies an original 'proof' sentence, and the model responds with a number of sentences which are different from the original (but chosen so as to confuse a language model that is attempting to select the most likely).

 - Input: an untokenized line of text
 - Output: a list of corrupted versions of the input

For example:

    corrupt<TAB>What's up today with you?<NEWLINE>
	What's up with you today?<TAB>What'sup with you today?<TAB>what's up with you today?<NEWLINE>

### Text Discrimination Challenge

The game requires two types of agent: the _corrupter_ and the _language model_. A _corrupter's_ job is to take a line of text, and produce corrupt variations of it, to try to confuse the language models. A _language model's_ job is to discriminate between the real & corrupted text. The LM Discrimination Challenge framework passes text through one or more corrputers on to one or more language models, and records statistics on the success of each language model in selecting the original text.

If this all sounds quite drastic, don't worry - we're just trying to provide a baseline metric that makes only the most basic assumptions about, and requires only the bare minimum from, a language model. Other metrics such as perplexity, KSPC, etc. remain useful - this is just for those situations that we can't be sure that those metrics are fair.

Basic pipeline of text flow orchestrated by the Text Discrimination Challenge:

    data_source ====+----> corrupter ----+====> language_model -----+===> evaluator
                     \-------------------+-------------------------/

A more complex flow, with multiple (2\*) corrupters & (3\*) language models is:

                                              /=> language_model 2 ----+====> evaluator 2
                     /---> corrupter 1 -\    /==> language_model 1 ----+====> evaluator 1
    data_source ====+----> corrupter 0 --+==+===> language_model 0 ----+====> evaluator 0
                     \----------------------+-------------------------/

#### Example

For an example of how the Text Discrimination Challenge works, consider the case of a single corrupter and language model.

1. A line of input is read from the data source.

        The cat sat on a mat

2. This text is passed to the corrupter, which generates some corrupted candidates.

        $ corrupt <TAB> The cat sat on a mat
        The cat sat on a hat
        Thw cat sat on a mat
        The cat Sat on a mat

3. These candidates are sorted & passed to the language model to rank.

        $ eval <TAB> The cat Sat on a mat <TAB> The cat sat on a hat <TAB> The cat sat on a mat <TAB> Thw cat sat on a mat
        1. The cat sat on a hat
        2. The cat sat on a mat
        3. The cat Sat on a mat
        4. Thw cat sat on a mat

4. The evaluator uses the rank of the correct candidate (2nd) to update the rolling scores.
