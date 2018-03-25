#!/bin/bash
set -ex 


# format data to pqa 
TASK=$1
#MODEL=use_sample_data
MODEL=sample_data

for d in train val
do
    python3 scripts/convert/format_DR.py data/${TASK}/sample_${d}.json data/${TASK}/DR_sample-${d}.json
done

for d in train val
do
    echo "process ${d}"
    python3 scripts/reader/preprocess.py  data/${TASK} data/${TASK} --split DR_sample-${d} --workers 20 --tokenizer spacy --standard 1
done

