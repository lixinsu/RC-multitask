#!/usr/bin/env python
# coding: utf-8

import sys
import json
from collections import defaultdict, Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def load_pred(pred_file):
    return json.loads(open(pred_file).read())


def load_origin(origin_file):
    data = []
    for line in open(origin_file):
        data.append(json.loads(line))
    return data

def calc_metrics(gt, pred):
    cmatrix = confusion_matrix(gt, pred)
    f1s  =  f1_score(gt, pred, average=None)
    recalls = recall_score(gt, pred, average=None)
    precisions = precision_score(gt, pred, average=None)
    return f1s, recalls, precisions, cmatrix

def eval_result(preds, origins):
    from sklearn import metrics
    qid2res = defaultdict(list)
    id2res = {}
    id2gt = {}
    qid2gt = defaultdict(list)
    for data in origins:
        qpid = data['qid']
        id2gt[qpid] = data['label']
        id2res[qpid] = preds[qpid][0]
        qid =qpid.split('-')[0]
        qid2gt[qid].append(data['label'])
        qid2res[qid].append(preds[qpid][0])

    # passage level metrics
    gts = []
    preds = []
    pred_scores = []
    for k in id2res:
        gts.append(id2gt[k])
        pred_score = id2res[k]
        pred_label = (1 if pred_score[1] > pred_score[0] else 0)
        preds.append(pred_label)
        pred_scores.append(pred_score[1])
    #print(preds)
    #print(gts)
    print('='*100)
    print('passage level')
    print('total %s' % len(preds))
    print('preditions %s pos' % (sum(preds),))
    print('groudtruth %s pos' % (sum(gts),))
    f1s, recalls, precisions, cmatrix = calc_metrics(gts, preds)
    print('f1', f1s)
    print('recall', recalls)
    print('precision', precisions)
    print('average precision score %s' % metrics.average_precision_score(gts, pred_scores, average=None))
    precision, recall, thresholds = metrics.precision_recall_curve(gts, pred_scores)
    print('AUC of PR %s ' % metrics.auc(recall, precision))
    fpr, tpr, thresholds = metrics.roc_curve(gts, pred_scores, pos_label=1)
    print('AUC of ROC %s ' % metrics.auc(fpr, tpr))
    print('ROC curve', roc_auc_score(gts, pred_scores))
    print('cmatrix')
    print(cmatrix)

    # question level metrics
    gts = []
    preds = []
    pred_scores = []
    for k in qid2res:
        gts.append(max(qid2gt[k]))
        max_score = max( [ x[1] for x in qid2res[k] ])
        pred_label = ( 1 if max_score > 0.5  else 0)
        preds.append(pred_label)
        pred_scores.append(max_score)
    print('='*100)
    print('question level')
    print('total %s' % len(preds))
    print('preditions %s pos' % (sum(preds),))
    print('groudtruth %s pos' % (sum(gts),))
    f1s, recalls, precisions, cmatrix = calc_metrics(gts, preds)
    print('f1', f1s)
    print('recall', recalls)
    print('precision', precisions)
    print('average precision score %s' % metrics.average_precision_score(gts, pred_scores, average=None))
    precision, recall, thresholds = metrics.precision_recall_curve(gts, pred_scores)
    print('AUC of PR %s ' % metrics.auc(recall,precision))
    fpr, tpr, thresholds = metrics.roc_curve(gts, pred_scores, pos_label=1)
    print('AUC of ROC %s ' % metrics.auc(fpr, tpr))
    print('ROC curve', roc_auc_score(gts, pred_scores))
    print('cmatrix')
    print(cmatrix)





if __name__ == "__main__":
    preds = load_pred(sys.argv[1])
    origins = load_origin(sys.argv[2])
    eval_result(preds, origins)



