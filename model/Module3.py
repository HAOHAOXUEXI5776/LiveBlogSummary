# coding: utf-8

# 网络结构和Module2一样，只不过现在使用Attention从下一层隐藏状态得到上一层的向量，而非平均池化或最大池化

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, query, context):
        batch_size = query.size(0)
        in_len = context.size(1)
        # (batch, query_len, dim) * (batch, in_len, dim) -> (batch, query_len, in_len)
        attn = torch.bmm(query, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn_scores = F.softmax(attn.view(-1, in_len), dim=1).view(batch_size, -1, in_len)
        # (batch, query_len, in_len) * (batch, in_len, dim) -> (batch, query_len, dim)
        attn_out = torch.bmm(attn_scores, context)

        return attn_out, attn_scores


class Module3(nn.Module):
    def __init__(self, args, embed=None):
        super(Module3, self).__init__()
        self.model_name = 'Module3'
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

        self.attn = Attention()
        self.word_query = nn.Parameter(torch.randn(1, 1, 2 * self.H))
        self.sent_query = nn.Parameter(torch.randn(1, 1, 2 * self.H))
        self.doc_query = nn.Parameter(torch.randn(1, 1, 2 * self.H))

        # 预测doc标签时，考虑doc内容，与blog相关性，doc位置，bias
        self.doc_content = nn.Linear(2 * self.H, 1, bias=False)
        self.doc_salience = nn.Bilinear(2 * self.H, 2 * self.H, 1, bias=False)
        self.doc_pos_embed = nn.Embedding(self.args.blog_trunc, self.args.pos_dim)
        self.doc_pos = nn.Linear(self.args.pos_dim, 1, bias=False)
        self.doc_bias = nn.Parameter(torch.FloatTensor(1).uniform_(-1.0, 1.0))

        # 预测sent标签时，考虑sent内容，与所在doc及blog相关性，sent位置，doc标签，bias
        self.sent_content = nn.Linear(2 * self.H, 1, bias=False)
        self.sent_salience = nn.Bilinear(2 * self.H, 4 * self.H, 1, bias=False)
        self.sent_pos_embed = nn.Embedding(self.args.doc_trunc, self.args.pos_dim)
        self.sent_pos = nn.Linear(self.args.pos_dim, 1, bias=False)
        self.sent_doc_label = nn.Linear(1, 1, bias=False)
        self.sent_bias = nn.Parameter(torch.FloatTensor(1).uniform_(-1.0, 1.0))

    def forward(self, x, doc_nums, doc_lens):  # x: total_sent_num * word_num

        # word => sent
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        for i, l in enumerate(sent_lens):  # 有些句子长度为0，将其变为"UNK"（长度为0在attention中没法计算）
            if l == 0:
                x[i][0] = 1
        word_mask = torch.ones_like(x) - torch.sign(x)
        x = self.embed(x)  # total_sent_num * word_num * D
        x = self.word_RNN(x)[0]  # total_sent_num * word_num * (2*H)
        word_mask = word_mask.data.type(torch.ByteTensor).view(x.size(0), 1, x.size(1))
        if use_cuda:
            word_mask = word_mask.cuda()
        query = self.word_query.expand(x.size(0), -1, -1).contiguous()
        self.attn.set_mask(word_mask)
        sent_vec = self.attn(query, x)[0].squeeze(1)  # total_sent_num * (2*H)

        # sent => doc
        x = self.padding(sent_vec, doc_lens, self.args.doc_trunc)  # total_doc_num * doc_trunc * (2*H)
        x = self.sent_RNN(x)[0]  # total_doc_num * doc_trunc * (2*H)
        sent_mask = torch.ones(x.size(0), self.args.doc_trunc)
        for i in range(x.size(0)):
            for j in range(doc_lens[i]):
                sent_mask[i][j] = 0
        sent_mask = sent_mask.type(torch.ByteTensor).view(x.size(0), 1, self.args.doc_trunc)
        if use_cuda:
            sent_mask = sent_mask.cuda()
        query = self.sent_query.expand(x.size(0), -1, -1).contiguous()
        self.attn.set_mask(sent_mask)
        doc_vec = self.attn(query, x)[0].squeeze(1)  # total_doc_num * (2*H)

        # doc => blog
        x = self.padding(doc_vec, doc_nums, self.args.blog_trunc)  # batch_size * blog_trunc * (2*H)
        x = self.doc_RNN(x)[0]  # batch_size * blog_trunc * (2*H)
        doc_mask = torch.ones(x.size(0), self.args.blog_trunc)
        for i in range(x.size(0)):
            for j in range(doc_nums[i]):
                doc_mask[i][j] = 0
        doc_mask = doc_mask.type(torch.ByteTensor).view(x.size(0), 1, self.args.blog_trunc)
        if use_cuda:
            doc_mask = doc_mask.cuda()
        query = self.doc_query.expand(x.size(0), -1, -1).contiguous()
        self.attn.set_mask(doc_mask)
        blog_vec = self.attn(query, x)[0].squeeze(1)  # batch_size * (2*H)

        # 预测doc标签
        probs = []
        start = 0
        for i in range(0, len(doc_nums)):
            end = start + doc_nums[i]
            valid = doc_vec[start:end]
            start = end
            for j, doc in enumerate(valid):
                doc_content = self.doc_content(doc)
                doc_salience = self.doc_salience(doc, blog_vec[i])
                doc_index = torch.LongTensor([[j]])
                if use_cuda:
                    doc_index = doc_index.cuda()
                doc_pos = self.doc_pos(self.doc_pos_embed(doc_index).squeeze(0))
                doc_pre = F.sigmoid(doc_content + doc_salience + doc_pos + self.doc_bias)
                probs.append(doc_pre)

        # 预测sent标签
        sent_idx = 0
        start = 0
        for i in range(0, len(doc_nums)):
            end = start + doc_nums[i]
            for j in range(start, end):
                context = torch.cat((blog_vec[i], doc_vec[j]))
                # next_sent_idx = sent_idx + doc_lens[j]
                for k in range(0, doc_lens[j]):
                    sent_content = self.sent_content(sent_vec[sent_idx])
                    sent_salience = self.sent_salience(sent_vec[sent_idx], context)
                    sent_index = torch.LongTensor([[k]])
                    if use_cuda:
                        sent_index = sent_index.cuda()
                    sent_pos = self.sent_pos(self.sent_pos_embed(sent_index).squeeze(0))
                    sent_doc_label = self.sent_doc_label(probs[i])
                    sent_pre = F.sigmoid(sent_content + sent_salience + sent_pos + sent_doc_label + self.sent_bias)
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
