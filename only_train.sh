#!/bin/bash
set -ex 


# format data to pqa 
TASK=$1
MODEL=multitask_sample_data

python3 scripts/reader/train.py  --num-epochs 40 \
                    --model-dir data/models/${TASK} \
                    --model-name ${MODEL} \
                    --data-dir data/${TASK} \
                    --train-file DR_sample-train-processed-spacy.txt \
                    --dev-file DR_sample-val-processed-spacy.txt \
                    --dev-json DR_sample-val.json \
                    --embed-dir data/embeddings \
                    --batch-size 32 \
                    --standard 1
