# coding:utf-8

# 使用PKUSUMSUM得到summary，并计算rouge值
#
# guardian(L):
# Coverage 0.149490343183 0.0159997516456 0.0856017314315
# Centroid 0.168149613246 0.0201327578085 0.0959103070393
# LexPageRank 0.171184467234 0.0194939809858 0.0962164539042
# TextRank 0.180521314667 0.0249420887843 0.102980172953
#
# bbc(L):
# Coverage 0.153488899326 0.0252654590562 0.0941556688018
# Centroid 0.132389392201 0.0176818767403 0.0808726507598
# LexPageRank 0.142839721633 0.0176441735951 0.0879295207592
# TextRank 0.166655855078 0.0271510870291 0.0999501744329


import os
import json
import sys
from tqdm import tqdm
from rouge import Rouge

reload(sys)
sys.setdefaultencoding('utf-8')
corpus = 'bbc'
jar_path = '/Users/liuhui/Desktop/Lab/Tools/PKUSUMSUM/PKUSUMSUM.jar'
# jar_path = '/home1/liuhui/PKUSUMSUM/PKUSUMSUM.jar'
data_dir = '../data/' + corpus + '_label/test/'
tmp_dir = './tmp/'
tmp_out = './out'
sum_len = 1  # 摘要长度是原摘要长度的几倍
methods = [0, 2, 4, 5]
methods_name = {0: 'Coverage', 1: 'Lead', 2: 'Centroid', 3: 'ILP', 4: 'LexPageRank', 5: 'TextRank', 6: 'Submodular'}

if __name__ == '__main__':
    rouge = Rouge()
    recall = []
    for i in range(0, 7):
        recall.append({'rouge-1': .0, 'rouge-2': .0, 'rouge-l': .0})
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    for fn in tqdm(os.listdir(data_dir)):
        f = open(os.path.join(data_dir, fn), 'r')
        blog = json.load(f)
        f.close()
        ref = str(' '.join(blog['summary']))
        sum_size = len(ref.strip().split()) * sum_len

        if os.path.exists(tmp_dir):  # 保证每次该文件夹都是空的
            for tf in os.listdir(tmp_dir):
                os.remove(tmp_dir + tf)

        for i, doc in enumerate(blog['documents']):
            tmp_f = open(tmp_dir + str(i), 'w')
            for sent in doc['text']:
                tmp_f.write(sent)
                tmp_f.write('\n')
            tmp_f.close()

        for m in methods:
            os.system('java -jar %s -T 2 -input %s -output %s -L 2 -n %d -m %d -stop stopword' % (
            jar_path, tmp_dir, tmp_out, 2 * sum_size, m))
            f = open(tmp_out, 'r')
            hyp = ' '.join(str(f.read()).strip().split()[:sum_size])
            f.close()
            score = rouge.get_scores(hyp, ref)[0]
            print methods_name[m], score['rouge-1']['r'], score['rouge-2']['r'], score['rouge-l']['r']
            recall[m]['rouge-1'] += score['rouge-1']['r']
            recall[m]['rouge-2'] += score['rouge-2']['r']
            recall[m]['rouge-l'] += score['rouge-l']['r']

    print('Final Results:')
    for m in methods:

        recall[m]['rouge-1'] /= len(os.listdir(data_dir))
        recall[m]['rouge-2'] /= len(os.listdir(data_dir))
        recall[m]['rouge-l'] /= len(os.listdir(data_dir))
        print methods_name[m], recall[m]['rouge-1'], recall[m]['rouge-2'], recall[m]['rouge-l']
