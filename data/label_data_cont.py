# coding: utf-8

# 使用规则给数据自动生成连续标签，包括doc标签和sentence标签，有两种方法：
# method 0: 标签为doc、sent和summary的rouge-1-f值
# method 1: doc标记方法同method 0；sent标签为summary中与之最像的句子的rouge-1-f值

import sys
import re
import json
import os
import tqdm
import math

reload(sys)
sys.setdefaultencoding('utf-8')
data_set = ['guardian', 'bbc']
types = ['train', 'valid', 'test']
method = 0  # 0表示原来的方法，doc和sent都和整个summary比较；1表示新方法，doc和整个比较，sent和最相似的句子比较


def rouge_1_f(hyp, ref):
    hyp = re.sub(r'[^a-z]', ' ', hyp.lower()).strip().split()
    ref = re.sub(r'[^a-z]', ' ', ref.lower()).strip().split()
    if len(hyp) == 0 or len(ref) == 0:
        return .0
    ref_flag = [0 for _ in ref]
    hit = .0
    for w in hyp:
        for i in range(0, len(ref)):
            if w == ref[i] and ref_flag[i] == 0:
                hit += 1
                ref_flag[i] = 1
                break
    p = hit / len(hyp)
    r = hit / len(ref)
    if math.fabs(p + r) < 1e-10:
        f = .0
    else:
        f = 2 * p * r / (p + r)
    if f > 1:
        print(hyp, ref)
    return f


if __name__ == '__main__':
    for corpus in data_set:
        print('Label %s...' % corpus)
        out_dir = corpus + '_cont_%d' % method
        if os.path.exists(out_dir):
            os.system('rm -r %s' % out_dir)
        os.mkdir(out_dir)
        for t in types:
            cur_out_dir = os.path.join(out_dir, t)
            os.mkdir(cur_out_dir)
            data_dir = corpus + '/' + t + '/'
            files = os.listdir(data_dir)
            for fn in tqdm.tqdm(files):
                cur_path = os.path.join(data_dir, fn)
                with open(cur_path, 'r') as f:
                    data = json.load(f)
                    summary = str(' '.join(data["summary"]))
                    for i, doc in enumerate(data['documents']):
                        doc_content = str(' '.join(doc['text'])).strip()
                        doc_score = rouge_1_f(doc_content, summary)
                        data['documents'][i]['doc_label'] = doc_score
                        sent_label = []
                        for sent in doc['text']:
                            sent = str(sent).strip()
                            if method == 0:
                                sent_score = rouge_1_f(sent, summary)
                                sent_label.append(sent_score)
                            else:
                                sent_score = .0
                                for sum_sent in data["summary"]:
                                    sent_score = max(sent_score, rouge_1_f(sent, sum_sent))
                                sent_label.append(sent_score)
                        data['documents'][i]['sent_label'] = sent_label

                    output = open(os.path.join(cur_out_dir, fn), 'w')
                    data_string = json.dumps(data, indent=4)
                    output.write(data_string)
                    output.close()
