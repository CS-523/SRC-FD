#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Traceless'
__mtime__ = '2024/5/12'
"""
import torch
import copy
import numpy as np
import math

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg

def get_dev_model(ori_model_state, updated_model_state):

    dev_model = copy.deepcopy(ori_model_state)
    for key, _ in list(dev_model.items()):
        dev_model[key] = updated_model_state[key] - ori_model_state[key]
    return dev_model

def add_noise_adp(y, epsilon_user, similarity=None, method='adp'):
    if method == 'sadp':
        assert similarity is not None, print('similarity cannot be None!!')
        epsilon_user = epsilon_user - math.log(1-similarity)
    # 获取每一层参数中的最小值和最大值
    min_weight = torch.min(y)
    max_weight = torch.max(y)
    # 每一层权重或偏置的变化范围 [c-r,c+r]
    center = (max_weight - min_weight) / 2  # + torch.zeros_like(y)
    radius = max_weight - center  # + torch.zeros_like(y)
    # 参数与 center 之间的距离 μ
    miu = y - center

    # 伯努利采样概率 Pr[u=1]
    # Pr = ((y - center) * (torch.exp(epsilon) - 1) + radius * (torch.exp(epsilon) + 1)) / 2 * radius * (torch.exp(epsilon) + 1)
    # 伯努利变量
    u = torch.bernoulli(torch.rand(y.shape))
    u_bool = u.type(torch.bool)
    # 自适应扰动
    epislon = math.exp(epsilon_user)
    epislon_1 = (epislon + 1) / (epislon - 1)
    # print('epislon_1 is {}'.format(epislon_1))
    y_test = copy.deepcopy(y)
    y_test[u_bool] = center + miu[u_bool] * epislon_1
    y_test[~u_bool] = center - miu[~u_bool] * epislon_1
    # for i in range(len(y_test)):
    #     if u[i] == 1.:
    #         y_test[i] = center + miu[i] * epislon_1
    #     elif u[i] == 0.:
    #         y_test[i] = center - miu[i] * epislon_1
    # 矩阵运算---更改成矩阵运算

    #
    # test_div = torch.sum(torch.abs(y_test - y))
    # print('test_div: {}'.format(test_div))
    return y_test

def sadp(w, epsilon_user, similar):
    # 记录各客户端本地模型每一层的参数个数
    for key, _ in list(w.items()):
        ###################################################### 一行一行地处理
        # 获取当前层的参数张量的维度
        dim = w[key].ndim

        # 获取当前层的行数 rows 、每一行所拥有的参数个数 N
        if 1 < dim:
            rows = w[key].shape[0]
        elif 1 == dim:
            rows = 1
        # 记录 N
        # 一行一行地处理
        if 1 < rows:
            # 堆叠每一行的 y
            y_matrix = None
            for row in range(rows):
                y_ori = w[key][row]
                '''Step-2.3 : 自适应参数扰动'''
                # epsilon 张量化
                # epsilon = epsilon_user + torch.zeros_like(y)
                y = add_noise_adp(y_ori, epsilon_user=epsilon_user, similarity=similar, method='sadp')
                if y_matrix is None:
                    y_matrix = copy.deepcopy(y)
                else:
                    y_matrix = torch.vstack((y_matrix, y))
            # 更新每一层扰动后的参数
            w[key] = y_matrix  # (rows, M)
        elif 1 == rows:
            y_ori = w[key]
            y_noise = add_noise_adp(y_ori, epsilon_user=epsilon_user, similarity=similar, method='sadp')
            # 更新每一层扰动后的参数
            w[key] = y_noise
        return w

def agg(w, dev_model_state):
    w_model = copy.deepcopy(w)
    for key, _ in list(w_model.items()):
        w_model[key] = dev_model_state[key] + w_model[key]
    return w_model


def adp(w, epsilon_user):
    # 记录各客户端本地模型每一层的参数个数
    for key,_ in list(w.items()):
        ###################################################### 一行一行地处理
        # 获取当前层的参数张量的维度
        dim = w[key].ndim

        # 获取当前层的行数 rows 、每一行所拥有的参数个数 N
        if 1 < dim:
            rows = w[key].shape[0]

        elif 1 == dim:
            rows = 1

        # 记录 N
        # 一行一行地处理
        if 1 < rows:
            # 堆叠每一行的 y
            y_matrix = None
            for row in range(rows):
                y_ori = w[key][row]
                '''Step-2.3 : 自适应参数扰动'''
                # epsilon 张量化
                # epsilon = epsilon_user + torch.zeros_like(y)
                y = add_noise_adp(y_ori, epsilon_user=epsilon_user)
                if y_matrix is None:
                    y_matrix = copy.deepcopy(y)
                else:
                    y_matrix = torch.vstack((y_matrix, y))
            # 更新每一层扰动后的参数
            w[key] = y_matrix  # (rows, M)
        elif 1 == rows:
            y_ori = w[key]
            y_noise = add_noise_adp(y_ori, epsilon_user=epsilon_user)
            # 更新每一层扰动后的参数
            w[key] = y_noise
        return w


def adp_cnn():
    # 这个是专门用来处理卷积神经网络，暂时不使用了
    pass
def radp():
    pass
