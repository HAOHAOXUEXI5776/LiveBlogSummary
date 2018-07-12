# coding:utf-8

# 更完整的层次式网络结构，word => sent => doc => live blog，预测doc标签时，使用doc向量和live blog向量，
# 预测sent标签时，使用sent向量、doc向量和live blog向量。
# 注：Module1的网络结构，word => sent => doc，预测doc标签时，只使用doc向量，预测sent标签时，也只使用doc向量

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


class Module2(nn.Module):
    def __init__(self, args, embed=None):
        super(Module2, self).__init__()
        self.model_name = 'Module2'
        self.args = args
        V = args.embed_num  # 单词表的大小
        D = args.embed_dim  # 词向量长度
        self.H = args.hidden_size  # 隐藏状态维数

        self.embed = nn.Embedding(V, D, padding_idx=0)
        if embed is not None:
            self.embed.weight.data.copy_(embed)

        # 输入batch_size个live blog，每个blog由doc_num个doc组成，一个doc有若sent_num个sent组成
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

        # 预测doc标签时，考虑doc向量、blog向量
        self.doc_pre = nn.Bilinear(2 * self.H, 2 * self.H, 1, bias=False)

        # 预测sent标签时，考虑sent向量、doc向量、blog向量（后两个拼接成一个4*H的向量）
        self.sent_pre = nn.Bilinear(2 * self.H, 4 * self.H, 1, bias=False)

    def avg_pool1d(self, x, seq_lens):
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
                out.append(F.avg_pool1d(t, t.size(2)))

        out = torch.cat(out).squeeze(2)
        return out

    def forward(self, x, doc_nums, doc_lens):
        # x: total_sent_num * word_num
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        x = self.embed(x)  # total_sent_num * word_num * D
        x = self.word_RNN(x)[0]  # total_sent_num * word_num * (2*H)
        sent_vec = self.avg_pool1d(x, sent_lens)  # total_sent_num * (2*H)

        x = sent_vec.view(-1, self.args.doc_trunc, 2 * self.H)  # total_doc_num * doc_trunc * (2*H)
        x = self.sent_RNN(x)[0]  # total_doc_num * doc_trunc * (2*H)
        doc_vec = self.avg_pool1d(x, doc_lens)  # total_doc_num * (2*H)

        x = self.pad_blog(doc_vec, doc_nums)  # batch_size * blog_trunc * (2*H)
        x = self.doc_RNN(x)[0]  # batch_size * blog_trunc * (2*H)
        blog_vec = self.avg_pool1d(x, doc_nums)  # batch_size * (2*H)

        # 预测doc标签
        probs = []
        start = 0
        for i in range(0, len(doc_nums)):
            end = start + doc_nums[i]
            valid = doc_vec[start:end]
            start = end
            for doc in valid:
                probs.append(self.doc_pre(doc, blog_vec[i]))

        # 预测sent标签
        sent_idx = 0
        start = 0
        for i in range(0, len(doc_nums)):
            end = start + doc_nums[i]
            for j in range(start, end):
                context = torch.cat((blog_vec[i], doc_vec[j]))
                next_sent_idx = sent_idx + self.args.doc_trunc
                for k in range(0, doc_lens[j]):
                    probs.append(self.sent_pre(sent_vec[sent_idx], context))
                    sent_idx += 1
                sent_idx = next_sent_idx
            start = end

        return F.sigmoid(torch.cat(probs).squeeze())  # 一维tensor，前部分是文档的预测，后部分是所有句子（不含padding）的预测

    def pad_blog(self, doc_vec, blog_lens):
        pad_dim = doc_vec.size(1)
        doc_trunc = self.args.blog_trunc
        result = []
        start = 0
        for blog_len in blog_lens:
            stop = start + blog_len
            valid = doc_vec[start:stop]  # (blog_len,2*H)
            start = stop
            if blog_len >= doc_trunc:
                result.append(valid[:doc_trunc].unsqueeze(0))
            else:
                pad = Variable(torch.zeros(doc_trunc - blog_len, pad_dim))
                if use_cuda:
                    pad = pad.cuda()
                result.append(torch.cat([valid, pad]).unsqueeze(0))  # (1,doc_trunc,2*H)
        result = torch.cat(result, dim=0)  # (B,doc_trunc,2*H)
        return result

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)

    def load(self, dir):
        if self.args.use_cuda:
            data = torch.load(dir)['model']
        else:
            data = torch.load(dir, map_location=lambda storage, loc: storage)['model']
        self.load_state_dict(data)
        if self.args.use_cuda:
            return self.cuda()
        else:
            return self
