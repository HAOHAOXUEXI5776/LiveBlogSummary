# coding: utf-8

# 使用PKUSUMSUM得到summary，并计算rouge值

import os
import json
from rouge import Rouge

corpus = 'bbc'
jar_path = 'PKUSUMSUM/PKUSUMSUM.jar'
data_dir = '../label_data/'+corpus+'_label/test/'
tmp_dir = './tmp/'
tmp_out = './out.txt'
sum_len = 1 # 摘要长度是原摘要长度的几倍
methods = [0, 4, 5, 6]
method_name = {0: 'Coverage', 1: 'Lead', 2: 'Centroid', 3: 'ILP', 4: 'LexPageRank', 5: 'TextRank', 6: 'Submoudlar'}

if not os.path.exists(tmp_dir):
	os.mkdir(tmp_dir)


rouge = Rouge()
recall = []
for i in range(0, 7):
	recall.append({'rouge-1': .0, 'rouge-2': .0, 'rouge-l': .0})

for i, fn in enumerate(os.listdir(data_dir)):
	f = open(os.path.join(data_dir, fn), 'r')
	blog = json.load(f)
	f.close()
	ref = ' '.join(blog['summary'])
	sum_size = len(ref.strip().split())*sum_len

	for tf in os.listdir(tmp_dir):
		os.remove(tmp_dir + tf)

	# 根据一篇LiveBlog，构建多文档
	for j, doc in enumerate(blog['documents']):
		f = open(tmp_dir + str(j)+'.txt', 'w', encoding = 'utf-8')
		for sent in doc['text']:
			f.write(sent+'\n')
		f.close()

	for m in methods:
		os.system('java -jar %s -T 2 -input %s -output %s -L 2 -n %d -m %d -stop stopword' %
			(jar_path, tmp_dir, tmp_out, 2*sum_size, m))
		f = open(tmp_out, 'r', encoding = 'utf-8')
		hyp = ' '.join(f.read().strip().split()[:sum_size])
		f.close()
		score = rouge.get_scores(hyp, ref)[0]
		recall[m]['rouge-1'] += score['rouge-1']['r']
		recall[m]['rouge-2'] += score['rouge-2']['r']
		recall[m]['rouge-l'] += score['rouge-l']['r']

tol = len(os.listdir(data_dir))
for m in methods:
	recall[m]['rouge-1'] /= tol
	recall[m]['rouge-2'] /= tol
	recall[m]['rouge-l'] /= tol
	print(method_name[m], recall[m]['rouge-1'], recall[m]['rouge-2'], recall[m]['rouge-l'])



# !java -jar PKUSUMSUM/PKUSUMSUM.jar -T 2 -input ./tmp/ -output out.txt -L 2 -n 108 -m 0 -stop n