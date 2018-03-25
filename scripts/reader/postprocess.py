#!/usr/bin/env python
# coding: utf-8

import json
import sys
import pickle
from collections import defaultdict


def load_pred_file(pred_file):
    return json.loads(open(pred_file).read())


def load_origin_file(data_file):
    qids, examples = pickle.load(open(data_file, 'rb'))
    return qids, examples


def merge_answers(preds, qids, examples):
    qid2preds  = defaultdict(list)
    qid2pconfidences  = defaultdict(list)
    for qid in qids:
        qid2preds[qid.rsplit('-')[0]].append(preds[qid][1])
        qid2pconfidences[qid.rsplit('-')[0]].append(preds[qid][0][1])
    return qid2preds, qid2pconfidences


def generate_answer(qid2preds, qid2pconfidences):
    qid2final = {}
    for qid, preds in qid2preds.items():
        finalpred = ''
        max_score = -10000
        pconf = qid2pconfidences[qid]
        for i,pred in enumerate(preds):
            text,score = pred[0]
            score = float(score)
            p_score = pconf[i]
            #score = p_score * score
            if score > max_score and p_score > 0.5:
                max_score = score
                finalpred = text
        qid2final[qid] = finalpred
    return qid2final


def save_to_disk(qid2final, filename):
    fo = open(filename, 'w')
    for qid, info in sorted(list(qid2final.items())):
        fo.write("%s\t%s\n" % (qid, info))

if __name__ == '__main__':
    preds = load_pred_file(sys.argv[1])
    qids, examples = load_origin_file(sys.argv[2])
    qid2preds,qid2pconfidences = merge_answers(preds, qids, examples)
    qid2final = generate_answer(qid2preds,qid2pconfidences)
    save_to_disk(qid2final, sys.argv[3])
