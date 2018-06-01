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

embed_path = "word2vec/embedding.npz"
voca_path =  "word2vec/word2id.json"
train_dir = "label_data/guardian_label/train/"
val_dir = "label_data/guardian_label/valid/"
batch_size = 32
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
                        batch_size = batch_size,
                        shuffle = False,
                      collate_fn=my_collate)

critieion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)

for epoch in range(0, 10):
    for i, batch in enumerate(train_iter):
        features, targets, _, doc_nums, doc_lens = vocab.make_features(batch, sent_trunc=sent_trunc, doc_trunc=doc_trunc)
        features, targets = Variable(features), Variable(targets.float())
        if use_cuda:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features, doc_nums, doc_lens)
        loss = critieion(probs, targets)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(net.parameters(), max_norm)
        optimizer.step()

        print("EPOCH %d: BATCH_ID=%d loss=%f time=%s"%(epoch, i, loss, time.ctime()))










