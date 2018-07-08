# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class Module1(nn.Module):
    def __init__(self, args, embed=None):
        super(Module1, self).__init__()
        self.model_name = 'Module1'
        self.args = args
        V = args.embed_num  # 单词表的大小
        D = args.embed_dim  # 词向量长度
        self.H = args.hidden_size  # 隐藏状态维数
        self.doc_trunc = args.doc_trunc

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

        # 此时，每个文档被一个1*2H维的向量所表示

        self.doc_pre = nn.Linear(2 * self.H, 1)

        self.sent_pre = nn.Linear(2 * self.H, self.doc_trunc)

    def max_pool1d(self, x, sent_lens):
        out = []
        for index, t in enumerate(x):
            if sent_lens[index] == 0:
                if use_cuda:
                    out.append(torch.zeros(1, 2 * self.H, 1).cuda())
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
        print('mark_begin')
        print(x)
        sent_lens = torch.sum(torch.sign(x), dim=1).data
        print(sent_lens)
        print('mark_end')
        x = self.embed(x)
        x = self.word_RNN(x)[0]  # total_sent_num*word_num*(2*H)
        x = self.max_pool1d(x, sent_lens)  # total_sent_num*(2*H)

        x = x.view(-1, self.args.doc_trunc, 2 * self.H)
        x = self.sent_RNN(x)[0]
        x = self.max_pool1d(x, doc_lens)  # total_doc_num*(2*H)

        doc_pro = self.doc_pre(x)
        target_pre = doc_pro.view(-1)

        sent_pro = self.sent_pre(x)
        for i, cur_doc in enumerate(sent_pro):
            try:
                target_pre = torch.cat((target_pre, cur_doc[:doc_lens[i]]))
            except:
                print(i, doc_lens[i], cur_doc.shape, target_pre.shape, cur_doc[:doc_lens[i]].shape)
                exit()
        return F.sigmoid(target_pre)  # 一维tensor，前部分是文档的预测，后部分是所有句子（不含padding）的预测

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'args': self.args}
        torch.save(checkpoint, dir)

    def load(self, dir):
        if self.args.device is not None:
            data = torch.load(dir)['model']
        else:
            data = torch.load(dir, map_location=lambda storage, loc: storage)['model']
        self.load_state_dict(data)
        if self.args.device is not None:
            return self.cuda()
        else:
            return self
