import torch

class Vocab():
    def __init__(self,embed,word2id):
        self.embed = embed
        self.word2id = word2id
        self.id2word = {v:k for k,v in word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'
    
    def __len__(self):
        return len(self.word2id)

    def i2w(self,idx):
        return self.id2word[idx]
    def w2i(self,w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX

    def make_features(self, batch, sent_trunc = 20, doc_trunc = 10):
        # sent_trunc: 每个句子的词数整到sent_trunc
        # doc_trunc： 每个文档的句子数整到doc_trunc

        # summarys, titles = [], []
        # for s, t in zip(batch["summary"], batch["title"]):
        #     summarys.append(''.join(s))
        #     titles.append(t)
        doc_nums = [] # 每个liveblog含有多少文档
        doc_targets = [] # 文档的标签，长度为sum(doc_nums)，不固定
        for d in batch["documents"]:
            doc_nums.append(len(d))
            for td in d:
                target = 1 if td["is_key_event"] else 0
                doc_targets.append(target)

        sents = [] #存储所有句子（一维、含padding添加的句子
        sents_target = [] # 存储所有句子的标签（一维、不含因padding添加的句子
        # sents_content = [] # 存储所有的句子内容，与sents_target等长，便于之后计算rouge值
        doc_lens = [] # 存储每篇文档包含的句子数（不含padding添加的句子
        for d in batch["documents"]:
            for td in d:
                cur_sent_num = len(td["text"])
                if(cur_sent_num > doc_trunc):
                    sents.extend(td["text"][:doc_trunc])
                    sents_target.extend(td["sent_label"][:doc_trunc])
                    # sents_content.extend(td["text"][:doc_trunc])
                    doc_lens.append(doc_trunc)
                else:
                    sents.extend(td["text"]+(doc_trunc-cur_sent_num)*[""])
                    sents_target.extend(td["sent_label"])
                    # sents_content.extend(td["text"])
                    doc_lens.append(cur_sent_num)

        # 将每个句子的单词数固定到sent_trunc
        for i, sent in enumerate(sents):
            sent = sent.split()
            cur_sent_len = len(sent)
            if cur_sent_len > sent_trunc:
                sent = sent[:sent_trunc]
            else:
                sent += (sent_trunc-cur_sent_len)*[self.PAD_TOKEN]
            sent = [self.w2i(_) for _ in sent]
            sents[i] = sent

        sents = torch.LongTensor(sents)
        targets = doc_targets+sents_target
        targets = torch.LongTensor(targets)

        return sents, targets, doc_nums, doc_lens






    # def make_featureds(self,batch,sent_trunc=50,doc_trunc=100,split_token='\n'):
    #     sents_list,targets,doc_lens = [],[],[]
    #     # trunc document
    #     for doc,label in zip(batch['doc'],batch['labels']):
    #         sents = doc.split(split_token)
    #         labels = label.split(split_token)
    #         labels = [int(l) for l in labels]
    #         max_sent_num = min(doc_trunc,len(sents))
    #         sents = sents[:max_sent_num]
    #         labels = labels[:max_sent_num]
    #         sents_list += sents
    #         targets += labels
    #         doc_lens.append(len(sents))
    #     # trunc or pad sent
    #     max_sent_len = 0
    #     batch_sents = []
    #     for sent in sents_list:
    #         words = sent.split()
    #         if len(words) > sent_trunc:
    #             words = words[:sent_trunc]
    #         max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
    #         batch_sents.append(words)
    #
    #     features = []
    #     for sent in batch_sents:
    #         feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len-len(sent))]
    #         features.append(feature)
    #
    #     features = torch.LongTensor(features)
    #     targets = torch.LongTensor(targets)
    #     summaries = batch['summaries']
    #
    #     return features,targets,summaries,doc_lens
