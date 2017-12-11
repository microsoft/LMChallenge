# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

echo "Downloading data, and building ngram models..."
echo "   (N.B. this should be run from the sample/ directory)"
echo

DATA="data"

mkdir -p ${DATA}

wget -O "${DATA}/raw.tar.gz" "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"

tar -xf "${DATA}/raw.tar.gz" -C "${DATA}" --strip-components=1

cat ${DATA}/training-monolingual.tokenized.shuffled/news.en-00001* \
    | env PYTHONPATH=.. python3 ngram_example.py sequence-words 3  \
    | env PYTHONPATH=.. python3 ngram_example.py prune 2           \
    | tee "${DATA}/words.ngram"                                    \
    | wc

cat ${DATA}/training-monolingual.tokenized.shuffled/news.en-00001* \
    | env PYTHONPATH=.. python3 ngram_example.py sequence-chars 5  \
    | env PYTHONPATH=.. python3 ngram_example.py prune 2           \
    | tee "${DATA}/chars.ngram"                                    \
    | wc

cat ${DATA}/heldout-monolingual.tokenized.shuffled/news.en.heldout-* \
    > ${DATA}/test.txt

cat ${DATA}/test.txt   \
    | tr ' ' '\n'      \
    | sort             \
    | uniq -c          \
    | sort -nr         \
    | awk '{print $2}' \
    | head -n 100000   \
    > ${DATA}/test.vocab.100k.txt
