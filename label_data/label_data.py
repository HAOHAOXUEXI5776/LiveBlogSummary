# coding:utf-8

# 使用规则给数据自动生成标签，包括doc的标签和sentence的标签，
# 如果doc、sent和summary的rouge-1-f大于相应阈值，则标记为1，否则标记为0
# Data stats：
# (guardian)
# Docs per blog = 55.538869258 +- 33.6076574249
# Sents per doc = 6.45938179312 +- 6.22231904109
# Words per sent = 21.8179994484 +- 16.4682899869
# (bbc)
# Docs per blog = 78.7382140876 +- 98.0568910135
# Sents per doc = 4.52551685275 +- 7.59216156536
# Words per sent = 16.8257352541 +- 10.6111924907

import json
from rouge import Rouge
import sys
import os
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')
data_set = ['guardian', 'bbc']
types = ['train', 'valid', 'test']
doc_threshold = {'guardian':0.18, 'bbc':0.15}
sent_threshold = {'guardian':0.18, 'bbc':0.15}

if __name__ == '__main__':
    rouge = Rouge()
    for corpus in data_set:
        doc_num = []    # 记录每篇blog的docs_per_blog, sent_per_doc, words_per_sent
        sent_num = []
        word_num = []
        for t in types:
            data_dir = corpus + '_processed/' + t + '/'
            files = os.listdir(data_dir)
            for fn in files:
                cur_path = os.path.join(data_dir, fn)
                print cur_path
                with open(cur_path, 'r') as f:
                    data = json.load(f)
                    doc_num.append(len(data['documents']))
                    summary = str(' '.join(data["summary"]))
                    for i, doc in enumerate(data['documents']):
                        doc_content = str(' '.join(doc['text']))
                        doc_score = rouge.get_scores(doc_content, summary, avg=True)['rouge-1']['f']
                        if doc_score > doc_threshold[corpus]:
                            data['documents'][i]['doc_label'] = 1
                        else:
                            data['documents'][i]['doc_label'] = 0
                        sent_label = []
                        sent_num.append(len(doc['text']))
                        for sent in doc['text']:
                            sent = str(sent)
                            word_num.append(len(sent.split()))
                            sent_score = rouge.get_scores(sent, summary, avg=True)['rouge-1']['f']
                            if sent_score > sent_threshold[corpus]:
                                sent_label.append(1)
                            else:
                                sent_label.append(0)
                        data['documents'][i]['sent_label'] = sent_label

                    output = open('./' + corpus + '_label/' + t + '/' + fn, 'w')
                    data_string = json.dumps(data, indent=4)
                    output.write(data_string)
                    output.close()
        print corpus
        doc_num = np.array(doc_num)
        sent_num = np.array(sent_num)
        word_num = np.array(word_num)
        print 'Docs per blog =', doc_num.mean(), '+-', doc_num.std()
        print 'Sents per doc =', sent_num.mean(), '+-', sent_num.std()
        print 'Words per sent =', word_num.mean(), '+-', word_num.std()
