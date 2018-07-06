#coding :utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
import rouge
import utils
from model import Module1


embed_path = "word2vec/embedding.npz"
voca_path =  "word2vec/word2id.json"
train_dir = "label_data/guardian_label/train/"
val_dir = "label_data/guardian_label/valid/"
test_dir = "label_data/guardian_label/test/"
checkpoint_dir = "checkpoint/"
batch_size = 1
learning_rate = 1e-3
max_norm = 1.0
sent_trunc = 20
doc_trunc = 10
use_cuda = torch.cuda.is_available()


def validate(net, vocab, val_iter, criterion):
    net.eval()
    loss, p1, r1, f1 = 0, 0, 0, 0
    batch_num = 0
    r = rouge.Rouge()
    r.metrics = ['rouge-1']

    punc = [',','.','?','!',';',':']
    def process_sent(s):
        # 去掉句子s中的标点，以及将字母转为小写
        for p in punc:
            s = s.replace(p, '')
        s =s.lower()
        return s

    def get_sent(documents, doc_lens, offset):
        i = 0
        cur_off = 0
        while True:
            if doc_lens[i] + cur_off > offset:
                break
            else:
                cur_off += doc_lens[i]
                i += 1
        return documents[i]['text'][offset-cur_off]

    for batch in val_iter:
        features, targets, doc_nums, doc_lens = vocab.make_features(batch, sent_trunc=sent_trunc,
                                                                       doc_trunc=doc_trunc)
        gold_summary = ' '.join(batch['summary'][0]) 
        documents = batch['documents'][0]

        features, targets = Variable(features), Variable(targets.float())
        if use_cuda:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features, doc_nums, doc_lens)
        loss += criterion(probs, targets).item()

        # 根据probs各分量的大小，选择得分最高的那部分的下标对应的句子作为摘要句
        doc_nums = doc_nums[0]
        probs = probs[doc_nums:] #去掉开头的文档标签预测
        if use_cuda:
            probs = probs.cpu()
        probs = list(probs.detach().numpy())
        sorted_index = list(np.argsort(probs)) #probs顺序排序后对应的下标
        sorted_index.reverse()

        gold_summary = process_sent(gold_summary)
        ref_len =len(gold_summary.split()) # 参考摘要中的单词数
        hyps_summary = ""
        cur_len = 0
        for index in sorted_index:
            tmp = get_sent(documents, doc_lens, index)
            tmp = process_sent(tmp).split()
            tmp_len = len(tmp)
            if cur_len + tmp_len > ref_len:
                hyps_summary += ' '.join(tmp[:ref_len-cur_len])
                break
            else:
                hyps_summary += ' '.join(tmp)+' '
                cur_len += tmp_len

        rouge_1 = r.get_scores(hyps_summary, gold_summary)[0]['rouge-1']
        p1 += rouge_1['p']
        r1 += rouge_1['r']
        f1 += rouge_1['f']

        batch_num += 1

    net.train()
    return loss / batch_num, p1 / batch_num, r1 / batch_num, f1 / batch_num


def train():
    embed = torch.Tensor(np.load(embed_path)['embedding'])
    with open(voca_path) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    train_data = []
    for fn in os.listdir(train_dir):
        f = open(train_dir + fn, 'r', encoding="utf-8")
        train_data.append(json.load(f))
        f.close()
    train_dataset = utils.Dataset(train_data)

    val_data = []
    for fn in os.listdir(val_dir):
        f = open(val_dir + fn, 'r', encoding="utf-8")
        val_data.append(json.load(f))
        f.close()
    val_dataset = utils.Dataset(val_data)

    embed_num = embed.size(0)
    embed_dim = embed.size(1)

    net = Module1(embed_num, embed_dim, doc_trunc, embed)
    if use_cuda:
        net.cuda()

    def my_collate(batch):
        return {key: [d[key] for d in batch] for key in batch[0]}

    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=my_collate)

    val_iter = DataLoader(dataset=val_dataset,
                          batch_size=1,
                          shuffle=False,
                          collate_fn=my_collate)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(0, 10):
        for i, batch in enumerate(train_iter):
            features, targets, doc_nums, doc_lens = vocab.make_features(batch, sent_trunc=sent_trunc,
                                                                           doc_trunc=doc_trunc)
            features, targets = Variable(features), Variable(targets.float())
            if use_cuda:
                features = features.cuda()
                targets = targets.cuda()
            probs = net(features, doc_nums, doc_lens)
            loss = criterion(probs, targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(net.parameters(), max_norm)
            optimizer.step()

            # 在eval上计算loss，rouge-1的p、r、f值

            print("EPOCH %d: BATCH_ID=%d loss=%f time=%s"%(epoch, i, loss, time.ctime()))
            if i%50 == 0:
                eval_loss, p1, r1, f1 = validate(net, vocab, val_iter, criterion)
                print("EVAL----EPOCH %d: BATCH_ID=%d loss=%f p1=%f r1=%f f1=%f time=%s" % (epoch, i, eval_loss, p1, r1, f1, time.ctime()))
        net.save('checkpoint', epoch)


def test():
    embed = torch.Tensor(np.load(embed_path)['embedding'])
    with open(voca_path) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    test_data = []
    for fn in os.listdir(test_dir):
        f = open(test_dir + fn, 'r', encoding="utf-8")
        test_data.append(json.load(f))
        f.close()
    test_dataset = utils.Dataset(test_data)

    embed_num = embed.size(0)
    embed_dim = embed.size(1)

    net = Module1(embed_num, embed_dim, doc_trunc, embed)
    if use_cuda:
        net.cuda()

    def my_collate(batch):
        return {key: [d[key] for d in batch] for key in batch[0]}

    test_iter = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=True,
                            collate_fn=my_collate)
    criterion = nn.BCELoss()

    for i in range(10):
        net.load(checkpoint_dir, i)
        if use_cuda:
            net.cuda()
        loss, p1, r1, f1 = validate(net, vocab, test_iter, criterion)
        print('loss=%f, r1=%f'%(loss, r1))


if __name__ == '__main__':
    # test()
    train()




