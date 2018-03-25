#!/bin/bash

#python3 scripts/reader/triggering_eval.py data/searchqa/pred/DR_test-use_sample_data.preds data/searchqa/DR_test.json
#python3 scripts/reader/triggering_eval.py data/quasart/pred/DR_test-use_sample_data.preds data/quasart/DR_test.json
#
#python3 scripts/reader/triggering_eval.py data/quasart/pred/DR_test-balance_sample_data.preds data/quasart/DR_test.json
#python3 scripts/reader/triggering_eval.py data/searchqa/pred/DR_test-balance_sample_data.preds data/searchqa/DR_test.json
python3 scripts/reader/triggering_eval.py data/searchqa/pred/tmp.preds data/searchqa/DR_test.json

