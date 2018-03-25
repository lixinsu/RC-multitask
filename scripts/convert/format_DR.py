#!/usr/bin/env python
# coding: utf-8

import json
import sys

def format_pqa(infile, outfile, mode='train'):
    print(outfile)
    fo = open(outfile, 'w')
    #debug = 1000
    with open(infile) as infp:
        for line in infp:
            #if debug == 0:
            #    break
            #debug -= 1
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            a = data['answer'].lower()
            q = data['query'].lower()
            qid = data['query_id']
            if len(a.strip()) == 0 or len(q)==0:
                print('error data format')
                continue
            for i, p in enumerate(data['passages']):
                p_text = p['passage_text'].lower()
                p_id = p['passage_id']
                label = 1 if a in p_text else 0
                if mode == 'train':
                    if len(p_text) > 0:
                        fo.write(json.dumps({'qid': "%s-%s" % (qid, p_id), 'passage': p_text, 'query': q, 'answers':[{'text':a, 'answer_start': p_text.find(a) if a in p_text else -1}], 'label': label}) + '\n')
                elif mode == 'test':
                    if len(p_text) > 0:
                        fo.write(json.dumps({'qid': "%s-%s" % (qid, p_id), 'passage': p_text, 'query': q, 'answers':[{'text':a, 'answer_start': -1}], 'label': label}) + '\n')

if __name__ == "__main__":
    if len(sys.argv) == 3:
        format_pqa(sys.argv[1], sys.argv[2])
    elif len(sys.argv)==4:
        format_pqa(sys.argv[1], sys.argv[2], mode=sys.argv[3])
