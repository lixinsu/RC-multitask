#!/bin/bash

TASK=$1
MODEL_PATH=$2

set -ex


#python3 scripts/convert/format_DR.py data/${TASK}/tiny_test.json data/${TASK}/DR_test.json test
#
#python3 scripts/reader/predict.py \
#                        data/${TASK}/DR_test.json \
#                        --model ${MODEL_PATH} \
#                        --out-dir data/${TASK}/pred \
#                        --out-file tmp.preds \
#                        --pkl-file tmp.pkl \
#                        --standard  \
#                        --batch-size 128 \
#                        --tokenizer spacy

python3 scripts/reader/postprocess.py data/${TASK}/pred/tmp.preds data/${TASK}/pred/tmp.pkl data/${TASK}/pred/DR_test.pred

python3 scripts/reader/evaluate.py data/${TASK}/tiny_test.json data/${TASK}/pred/DR_test.pred
