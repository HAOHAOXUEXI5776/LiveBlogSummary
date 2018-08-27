# coding:utf-8

# 几乎与Module2一模一样，删除了doc label预测层（Module4是直接删除了doc表示层）
# 在预测句子标签时也去掉了和doc label有关的一项

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


class Module5(nn.Module):
    def __init__(self, args, embed=None):
        super(Module5, self).__init__()
        self.model_name = 'Module5'
        self.args = args
        V = args.embed_num  # 单词表的大小
        D = args.embed_dim  # 词向量长度
        self.H = args.hidden_size  # 隐藏状态维数

        self.embed = nn.Embedding(V, D, padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        # 输入batch_size个live blog，每个live blog由doc_num个doc组成，一个doc有若sent_num个sent组成
        # 一个sent有word_num个word组成
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

        self.doc_RNN = nn.GRU(
            input_size=2 * self.H,
            hidden_size=self.H,
            batch_first=True,
            bidirectional=True
        )

        # 预测sent标签时，考虑sent内容，与所在doc及blog相关性，sent位置，doc标签，bias
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

        x = self.padding(sent_vec, doc_lens, self.args.doc_trunc)  # total_doc_num * doc_trunc * (2*H)
        x = self.sent_RNN(x)[0]  # total_doc_num * doc_trunc * (2*H)
        doc_vec = self.max_pool1d(x, doc_lens)  # total_doc_num * (2*H)

        x = self.padding(doc_vec, doc_nums, self.args.blog_trunc)  # batch_size * blog_trunc * (2*H)
        x = self.doc_RNN(x)[0]  # batch_size * blog_trunc * (2*H)
        blog_vec = self.max_pool1d(x, doc_nums)  # batch_size * (2*H)

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
