import os
import sys
import json
sys.path.append('../')
from Module1 import Module1
import utils


from graphviz import Digraph
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot




# 构造net的输入
doc_trunc = 10
sent_trunc = 20
embed_path = "../word2vec/embedding.npz"
voca_path = "../word2vec/word2id.json"
embed = torch.Tensor(np.load(embed_path)['embedding'])
with open(voca_path) as f:
    word2id = json.load(f)
vocab = utils.Vocab(embed, word2id)

data_dir = "../label_data/guardian_label/valid/"
data = []
for fn in os.listdir(data_dir):
    f = open(data_dir + fn, 'r', encoding="utf-8")
    data.append(json.load(f))
    f.close()
dataset = utils.Dataset(data)

def my_collate(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}

data_iter = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=True,
                        collate_fn=my_collate)

for i, batch in enumerate(data_iter):
    break

features, targets, doc_nums, doc_lens = vocab.make_features(batch, sent_trunc=sent_trunc,
                                                                           doc_trunc=doc_trunc)
features, targets = Variable(features), Variable(targets.float())



net = Module1(embed_num = embed.size(0), embed_dim = embed.size(1), doc_trunc = doc_trunc, embed = embed)
y = net(features, doc_nums, doc_lens)
g = make_dot(y)
g.render('here', view = False)