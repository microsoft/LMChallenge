# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

echo "# Downloading data, and building ngram models..."
echo "#   (N.B. this should be run from the sample/ directory)"
echo

DATA="data-big"

mkdir -p ${DATA}

echo "# Downloading data"
wget -O "${DATA}/raw.tar.gz" "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
tar -xf "${DATA}/raw.tar.gz" -C "${DATA}" --strip-components=1

echo "# Creating test set"
cat ${DATA}/heldout-monolingual.tokenized.shuffled/news.en.heldout-* \
    > ${DATA}/test.txt

echo "# Creating test vocab"
cat ${DATA}/test.txt   \
    | tr ' ' '\n'      \
    | sort             \
    | uniq -c          \
    | sort -nr         \
    | awk '{print $2}' \
    | head -n 100000   \
    > ${DATA}/test.vocab.100k.txt

echo "# Setting memory limit at 3 GB"
ulimit -Sv 3000000

echo "# Counting word ngrams"
time cat ${DATA}/training-monolingual.tokenized.shuffled/news.en-0000*   \
    | env PYTHONPATH=.. python3 ngram_example.py sequence-words 3 --disk \
    | env PYTHONPATH=.. python3 ngram_example.py prune 3                 \
    > "${DATA}/words.ngram"

echo "# Counting character ngrams"
time cat ${DATA}/training-monolingual.tokenized.shuffled/news.en-0000* \
    | env PYTHONPATH=.. python3 ngram_example.py sequence-chars 5      \
    | env PYTHONPATH=.. python3 ngram_example.py prune 3               \
    > "${DATA}/chars.ngram"
