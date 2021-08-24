#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from torch import nn
from torch.nn import functional as F
import torch


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        """
        dice_loss
        
        :param predict: 模型输出(b*c*h*w)
        :param target: 目标（b*h*w)
        :return: dice_loss
        """
        batch_size = predict.size(0)
        num_class = predict.size(1)
        probability = F.softmax(predict, dim=1)  # 转成概率形式

        # 转one-hot
        target_one_hot = F.one_hot(target, num_classes=num_class).permute((0, 3, 1, 2))
        loss = 0.0
        for i in range(num_class):
            p = probability[:, i, ...]
            gt = target_one_hot[:, i, ...]
            dice_coff = (2 * torch.sum(p * gt) + self.smooth) / (torch.sum(p) + torch.sum(gt) + self.smooth)
            loss += dice_coff

        return 1 - loss / (num_class * batch_size)
