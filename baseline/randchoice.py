# coding:utf-8

import rouge
import os
import json
import random

test_dir = '../label_data/guardian_label/test/'

punc = [',','.','?','!',';',':']
def process_sent(s):
    # 去掉句子s中的标点，以及将字母转为小写
    for p in punc:
        s = s.replace(p, '')
    s =s.lower()
    return s

r1, r2, rl = 0, 0, 0
r = rouge.Rouge()
for fn in os.listdir(test_dir):
    f = open(test_dir+fn, 'r', encoding='utf-8')
    data = json.load(f)
    f.close()
    sents = []
    for d in data['documents']:
        for s in d['text']:
            sents.append(process_sent(s).split())
    ref = ' '.join(data['summary'])
    ref = process_sent(ref)
    sum_len = len(ref.split())

    hyp = ''
    sent_len = len(sents)
    idx = [_ for _ in range(sent_len)]
    random.shuffle(idx)
    cur_len = 0
    i = 0
    while i < sent_len:
        tmp_len = len(sents[idx[i]])
        if cur_len + tmp_len < sum_len:
            hyp += ' '.join(sents[idx[i]]) + ' '
            cur_len += tmp_len
        else:
            hyp += ' '.join(sents[idx[i]][:sum_len-cur_len])
            break
        i += 1

    rouge_score = r.get_scores(hyp, ref)[0]

    r1 += rouge_score['rouge-1']['r']
    r2 += rouge_score['rouge-2']['r']
    rl += rouge_score['rouge-l']['r']

tol = len(os.listdir(test_dir))

r1/=tol
r2/=tol
rl/=tol

print('r1=%.3f r2=%.3f r3=%.3f'%(r1,r2,rl))