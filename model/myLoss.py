# coding: utf-8

# 自定义loss函数，在训练中不断调整Doc label和Sent label的占比，从而使训练效果达到最好

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class myLoss(nn.Module):
    def __init__(self, beta=0.0):
        super(myLoss, self).__init__()
        self.beta = beta
        # self.beta = nn.Parameter(torch.Tensor([beta]))

    def forward(self, predict, target, doc_num):
        pre_1 = predict[:doc_num]
        tar_1 = target[:doc_num]
        pre_2 = predict[doc_num:]
        tar_2 = target[doc_num:]
        loss_1 = F.mse_loss(pre_1, tar_1)
        loss_2 = F.mse_loss(pre_2, tar_2)
        norm = self.beta + 1.0
        loss = (self.beta * loss_1 + loss_2) / norm
        return loss
