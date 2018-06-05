import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import utils
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import time
import rouge

embed_path = "word2vec/embedding.npz"
voca_path =  "word2vec/word2id.json"
train_dir = "label_data/guardian_label/train/"
val_dir = "label_data/guardian_label/valid/"
batch_size = 1
learning_rate = 1e-3
max_norm = 1.0
sent_trunc = 20
doc_trunc = 10

use_cuda = torch.cuda.is_available()

class MyModule(nn.Module):
    def __init__(self, embed_num, embed_dim, doc_trunc, embed = None):
        super(MyModule, self).__init__()

        V = embed_num #单词表的大小
        D = embed_dim #词向量长度
        self.H = 200
        self.doc_trunc = doc_trunc

        self.embed = nn.Embedding(V, D, padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        # 输入batch_size个live blog，每个blog由doc_num个doc组成，一个doc有若sent_num个sent组成
        # 一个sent有word_num个word组成
        self.word_RNN = nn.GRU(
            input_size = D,
            hidden_size = self.H,
            batch_first = True,
            bidirectional = True
        )

        self.sent_RNN = nn.GRU(
            input_size = 2*self.H,
            hidden_size = self.H,
            batch_first = True,
            bidirectional = True
        )

        # 此时，每个文档被一个1*2H维的向量所表示

        self.doc_pre = nn.Linear(2*self.H, 1)

        self.sent_pre = nn.Linear(2*self.H, self.doc_trunc)

    def max_pool1d(self, x, sent_lens):
        out = []
        for index, t in enumerate(x):
            if sent_lens[index] == 0:
                if use_cuda:
                    out.append(torch.zeros(1, 2*self.H, 1).cuda())
                else:
                    out.append(torch.zeros(1, 2 * self.H, 1))
            else:
                try:
                    tmpsize = t.shape
                    t = t[:sent_lens[index], :]
                except:
                    print(index, sent_lens[index], tmpsize)
                    exit()
                t = torch.t(t).unsqueeze(0)
                out.append(F.avg_pool1d(t, t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def forward(self, x, doc_nums, doc_lens):
        # x: total_sent_num* word_num
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        x = self.embed(x)
        x = self.word_RNN(x)[0] #total_sent_num*word_num*(2*H)
        x = self.max_pool1d(x, sent_lens)#total_sent_num*(2*H)

        x = x.view(-1, doc_trunc, 2*self.H)
        x = self.sent_RNN(x)[0]
        x = self.max_pool1d(x, doc_lens) #total_doc_num*(2*H)

        doc_pro = self.doc_pre(x)
        target_pre = doc_pro.view(-1)

        sent_pro = self.sent_pre(x)
        for i, cur_doc in enumerate(sent_pro):
            try:
                target_pre = torch.cat((target_pre, cur_doc[:doc_lens[i]]))
            except:
                print(i, doc_lens[i], cur_doc.shape, target_pre.shape, cur_doc[:doc_lens[i]].shape)
                exit()
        return F.sigmoid(target_pre) #一维tensor，前部分是文档的预测，后部分是所有句子（不含padding）的预测

    def save(self, dir):
        checkpoint = self.state_dict()
        torch.save(checkpoint, dir+'/para.pt')

    def load(self, dir):
        data = torch.load(dir+'/para.pt')
        self.load_state_dict(data)
        return self



embed = torch.Tensor(np.load(embed_path)['embedding'])
with open(voca_path) as f:
    word2id = json.load(f)
vocab = utils.Vocab(embed, word2id)

train_data = []
for fn in os.listdir(train_dir):
    f = open(train_dir+fn, 'r', encoding = "utf-8")
    train_data.append(json.load(f))
    f.close()
train_dataset = utils.Dataset(train_data)

val_data = []
for fn in os.listdir(val_dir):
    f = open(val_dir+fn, 'r', encoding = "utf-8")
    val_data.append(json.load(f))
    f.close()
val_dataset = utils.Dataset(val_data)

embed_num = embed.size(0)
embed_dim = embed.size(1)

net = MyModule(embed_num, embed_dim, doc_trunc, embed)
if use_cuda:
    net.cuda()

def my_collate(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}

train_iter = DataLoader(dataset = train_dataset,
                        batch_size = batch_size,
                        shuffle = True,
                        collate_fn=my_collate)

val_iter = DataLoader(dataset = val_dataset,
                        batch_size = 1,
                        shuffle = False,
                      collate_fn=my_collate)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

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

    for batch in val_iter:
        features, targets, sents_content, gold_summary, doc_nums, doc_lens = vocab.make_features(batch, sent_trunc=sent_trunc,
                                                                       doc_trunc=doc_trunc)
        gold_summary = gold_summary[0] #返回的gold_summary是列表，而batch_size为1，故取第一个便是了

        features, targets = Variable(features), Variable(targets.float())
        if use_cuda:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features, doc_nums, doc_lens)
        loss += criterion(probs, targets)

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
            tmp = sents_content[index]
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
    return loss/batch_num, p1/batch_num, r1/batch_num, f1/batch_num


for epoch in range(0, 10):
    for i, batch in enumerate(train_iter):
        features, targets, _1, _2, doc_nums, doc_lens = vocab.make_features(batch, sent_trunc=sent_trunc,
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
            print("EVAL----EPOCH %d: BATCH_ID=%d loss=%f p1=%f r1=%f f1=%f time=%s" % (epoch, i, loss, p1, r1, f1, time.ctime()))









