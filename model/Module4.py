# coding: utf-8

# 用于和Module2对比，从而验证层次式网络结构的有效性：
# 在Module4中，忽略doc这一层，所有sent池化后得到整个文档的表示，然后直接预测各sent的得分

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


class Module4(nn.Module):
    def __init__(self, args, embed=None):
        super(Module4, self).__init__()
        self.model_name = 'Module4'
        self.args = args
        V = args.embed_num  # 单词表的大小
        D = args.embed_dim  # 词向量长度
        self.H = args.hidden_size  # 隐藏状态维数

        self.embed = nn.Embedding(V, D, padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        self.word_RNN = nn.GRU(
            input_size=D,
            hidden_size=self.H,
            batch_first=True,
            bidirectional=True
        )

        self.sent_RNN = nn.GRU(
            input_size=2 * self.H,
            hidden_size=self.H,
            batch_first=True,
            bidirectional=True
        )

        # 预测sent标签时，考虑sent内容，与整个blog相关性，sent位置，bias
        self.sent_content = nn.Linear(2 * self.H, 1, bias=False)
        self.sent_salience = nn.Bilinear(2 * self.H, 2 * self.H, 1, bias=False)
        self.sent_pos_embed = nn.Embedding(self.args.doc_trunc, self.args.pos_dim)
        self.sent_pos = nn.Linear(self.args.pos_dim, 1, bias=False)
        self.sent_bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

    def max_pool1d(self, x, seq_lens):
        out = []
        for index, t in enumerate(x):
            if seq_lens[index] == 0:
                if use_cuda:
                    out.append(torch.zeros(1, 2 * self.H, 1).cuda())
                else:
                    out.append(torch.zeros(1, 2 * self.H, 1))
            else:
                t = t[:seq_lens[index], :]
                t = torch.t(t).unsqueeze(0)
                out.append(F.max_pool1d(t, t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def forward(self, x, doc_nums, doc_lens):
        # x: total_sent_num * word_num
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        x = self.embed(x)  # total_sent_num * word_num * D
        x = self.word_RNN(x)[0]  # total_sent_num * word_num * (2*H)
        sent_vec = self.max_pool1d(x, sent_lens)  # total_sent_num * (2*H)

        blog_lens = []
        start = 0
        for doc_num in doc_nums:
            blog_lens.append(np.array(doc_lens[start: start + doc_num]).sum())
            start = start + doc_num

        x = self.padding(sent_vec, blog_lens, self.args.doc_trunc * self.args.blog_trunc)
        x = self.sent_RNN(x)[0]
        blog_vec = self.max_pool1d(x, doc_lens)  # batch_size * (2*H)

        probs = []
        # 预测sent标签
        sent_idx = 0
        start = 0
        for i in range(0, len(doc_nums)):
            end = start + doc_nums[i]
            for j in range(start, end):
                context = blog_vec[i]
                # next_sent_idx = sent_idx + doc_lens[j]
                for k in range(0, doc_lens[j]):
                    sent_content = self.sent_content(sent_vec[sent_idx])
                    sent_salience = self.sent_salience(sent_vec[sent_idx], context)
                    sent_index = torch.LongTensor([[k]])
                    if use_cuda:
                        sent_index = sent_index.cuda()
                    sent_pos = self.sent_pos(self.sent_pos_embed(sent_index).squeeze(0))
                    sent_pre = sent_content + sent_salience + sent_pos + self.sent_bias
                    probs.append(sent_pre)
                    sent_idx += 1
                # sent_idx = next_sent_idx
            start = end
        return torch.cat(probs).squeeze()  # 一维tensor，前部分是文档的预测，后部分是所有句子（不含padding）的预测

    # 对于一个序列进行padding，不足的补上全零向量
    def padding(self, vec, seq_lens, trunc):
        pad_dim = vec.size(1)
        result = []
        start = 0
        for seq_len in seq_lens:
            stop = start + seq_len
            valid = vec[start:stop]
            start = stop
            pad = Variable(torch.zeros(trunc - seq_len, pad_dim))
            if use_cuda:
                pad = pad.cuda()
            result.append(torch.cat([valid, pad]).unsqueeze(0))
        result = torch.cat(result, dim=0)
        return result

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)
