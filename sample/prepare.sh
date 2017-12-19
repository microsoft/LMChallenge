# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

echo "# Downloading data, and building ngram models..."
echo "#   (N.B. this should be run from the sample/ directory)"
echo

DATA="data"

mkdir -p ${DATA}

echo "# Downloading data"
wget -O "${DATA}/raw.zip" "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
unzip -j "${DATA}/raw.zip" -d "${DATA}"

echo "# Creating correction vocab"
cat ${DATA}/*.tokens   \
    | tr ' ' '\n'      \
    | sort             \
    | uniq -c          \
    | sort -nr         \
    | awk '{print $2}' \
    > ${DATA}/vocab.txt

echo "# Counting word ngrams"
cat ${DATA}/wiki.train.tokens                             \
    | env PYTHONPATH=.. python3 ngram.py sequence-words 3 \
    | env PYTHONPATH=.. python3 ngram.py prune 3          \
    | tee "${DATA}/words.3gram"                           \
    | awk -F'\x1e' '{if (NF <= 3) print $0}'              \
    | tee "${DATA}/words.2gram"                           \
    | awk -F'\x1e' '{if (NF == 2) print $0}'              \
    > "${DATA}/words.1gram"

echo "# Counting character ngrams"
echo "#    (N.B. if this were serious, it should use the untokenized data)"
cat ${DATA}/wiki.train.tokens                              \
    | env PYTHONPATH=.. python3 ngram.py sequence-chars 5  \
    | env PYTHONPATH=.. python3 ngram.py prune 3           \
    > "${DATA}/chars.5gram"
