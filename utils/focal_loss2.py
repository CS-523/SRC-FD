#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Traceless'
https://www.freesion.com/article/32691013519/
"""
import copy

import torch
from torch import nn


class focal_loss2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss2, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print("Focal_loss alpha = {}, gamma = {}".format(alpha, gamma))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {}, gamma = {}".format(alpha, gamma))

            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        self.all_loss = None  # 用来记录所有样本的损失值

    def forward(self, preds_softmax, labels):
        """
        focal_loss损失计算
        https://blog.csdn.net/weixin_42445581/article/details/105909907 为什么总是出现nan值
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        self.alpha = self.alpha.to(preds_softmax.device)
        min_pro = torch.tensor([10e-10, 10e-10]).to(preds_softmax.device)
        preds_softmax = preds_softmax.view(-1, preds_softmax.size(-1))  #
        preds_softmax = torch.maximum(preds_softmax, min_pro)
        preds_logsoft = torch.log(preds_softmax)
        # https://zhuanlan.zhihu.com/p/462008911
        preds_softmax = preds_softmax.gather(1, labels.view(-1,
                                                            1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )----得到pt
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))  # 得到log(pt)
        alpha = self.alpha.gather(0, labels.view(-1))  # 得到at
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        self.all_loss = copy.deepcopy(loss.detach())

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss





