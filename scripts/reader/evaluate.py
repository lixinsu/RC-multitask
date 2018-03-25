#!/usr/bin/env python
# coding: utf-8

import sys
import json
import time
import regex as re
import string
from collections import Counter
# ------------------------------------------------------------------------------
# Evaluation. Follows official evalutation script for v1.1 of the SQuAD dataset.
# ------------------------------------------------------------------------------


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def regex_match_score(prediction, pattern):
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE
        )
    except BaseException:
        logger.warn('Regular expression failed to compile: %s' % pattern)
        return False
    return compiled.match(prediction) is not None


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def load_answer(filename):
    qid2answer = {}
    with open(filename) as fi:
        for line in fi:
            data = json.loads(line)
            qid2answer[str(data['query_id'])] = data['answer']
    return qid2answer


def load_pred_answer(filename):
    qid2answer = {}
    with open(filename) as fi:
        for line in fi:
            data = line.strip('\n').split('\t')
            qid2answer[str(data[0])] = data[1]
    return qid2answer

if __name__ == '__main__':
    qid2answer = load_answer(sys.argv[1])
    pred_qid2answer = load_pred_answer(sys.argv[2])
    f1 =  []
    em = []
    for qid, answer in qid2answer.items():
        if qid not in pred_qid2answer:
            print('qid not in predictions, error')
            continue
        #print(pred_qid2answer[qid],'===', answer)
        if not pred_qid2answer[qid]:
            continue
        f1.append(f1_score(pred_qid2answer[qid] ,answer))
        em.append(exact_match_score(pred_qid2answer[qid] ,answer))
    print('total %s samples' % len(f1))
    print('f1 score: %s ' % (sum(f1)/len(f1)))
    print('em score %s ' % (sum(em)/len(em)))
