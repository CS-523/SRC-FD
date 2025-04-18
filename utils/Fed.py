#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import time
import math

import crypten
import crypten.mpc as mpc
import numpy as np
import torch
from crypten.encoder import FixedPointEncoder

from adp import get_dev_model, agg, sadp
from .recipe import total_size


def Privacy_account(args, threshold_epochs):
    # args中需要用到的参数有args.num_items_train，args.delta，args.privacy_budget
    q_s = 1  # 在本实验中默认使用所有的客户端
    clipthr = 20  # 一般是默认值
    # q_s = args.num_Chosenusers/args.num_users
    # num_items_train是参与者数据集的数据量大小
    delta_s = 2 * clipthr / args.num_items_train  # 三角s
    noise_scale = delta_s * np.sqrt(2 * q_s * threshold_epochs * np.log(1 / args.delta)) / args.privacy_budget
    return noise_scale

def add_security(model, privacy_method, device, dataset_sizes):
    if privacy_method == 'FedAVG':
        tmp_model = copy.deepcopy(model)
        # raise NotImplementedError
    elif 'DP' in privacy_method:
        epsilon = eval(privacy_method.split(':')[-1])
        tmp_model = copy.deepcopy(model)
        tmp_model_state = tmp_model.state_dict()
        # tmp_model_state_noise, _ = FedHFL_fla_opt([tmp_model_state], a=epsilon)

        noise_scale = Privacy_account_new(num_items_train=dataset_sizes, delta=0.01, privacy_budget=epsilon,
                                          threshold_epochs=20)
        tmp_model_state_noise = noise_add(noise_scale, [tmp_model_state], device=device)

        tmp_model.load_state_dict(tmp_model_state_noise[0])
    elif 'PLC' in privacy_method:
        alpha = eval(privacy_method.split(':')[-1])
        tmp_model = copy.deepcopy(model)
        tmp_model_state = tmp_model.state_dict()
        # tmp_model_state_noise = model_add_noise(noise=1 * (1 + alpha) * (1 + np.random.random()), w=tmp_model_state)
        tmp_model_state_noise = model_add_noise_alpha(alpha, w=tmp_model_state)
        tmp_model.load_state_dict(tmp_model_state_noise)

    return tmp_model

# def add_noise_to_gradients_dp_model(model, budget):
#     for name, p in model.named_parameters():
#         if p.grad is not None:
#             sentivity = torch.max(p.grad) - torch.min(p.grad)
#             noise_scale = (sentivity / budget).cpu().numpy() + 1e-6
#             noise = np.random.laplace(0, noise_scale, p.grad.size())
#             noise = torch.from_numpy(noise).float().to(p.grad.device)
#             # 添加噪声值
#             p.grad.add_(noise)


def add_noise_to_gradients_dp_model(model, model_back, ori_budget):
    model_back_state = model_back.state_dict()
    model_state = model.state_dict()
    bias_grad_dict = {}

    for name, p in model.named_parameters():
        bias_grad_dict[name] = p - model_back_state[name]  # 存储原始信息
    for name, p in bias_grad_dict.items():
        # p = model_para[name]
        sentivity = torch.max(p) - torch.min(p)
        noise_scale = (sentivity / ori_budget).detach().cpu().numpy() + 1e-6
        noise = np.random.laplace(0, noise_scale, p.size())
        noise = torch.from_numpy(noise).float().to(p.device)

        model_state[name] = model_state[name] + noise
        # model_state[name] = p + noise
    error_sum = 0
    for key, dy_dx in model_state.items():
        error = torch.sqrt(torch.sum((model_back_state[key] - dy_dx) ** 2))
        error_sum = error_sum + error
    print('error_sum: {}'.format(error_sum))
    return model_state


    # for name, p in model.named_parameters():
    #     if p.grad is not None:
    #         sentivity = torch.max(p.grad) - torch.min(p.grad)
    #         noise_scale = (sentivity / budget).cpu().numpy() + 1e-6
    #         noise = np.random.laplace(0, noise_scale, p.grad.size())
    #         noise = torch.from_numpy(noise).float().to(p.grad.device)
    #         # 添加噪声值
    #         p.grad.add_(noise)

# def add_noise_to_gradients_sentry_model(model, budget, new_budget):
#     # model_para = model.state_dict()
#     # for name in model_para.keys():
#     bias_grad_dict = {}
#     for name, p in model.named_parameters():
#         if 'bias' in name:
#             bias_grad_dict[name] = p.grad  # 存储原始信息
#     for name, p in model.named_parameters():
#         # p = model_para[name]
#         if p.grad is not None:
#
#             if 'weight' in name:
#                 bias_key = name.replace('weight', 'bias')
#                 num_bias = bias_grad_dict[bias_key].size()[0]
#                 sentivity = torch.abs(bias_grad_dict[bias_key])
#                 sentivity = sentivity.repeat(p.shape[1], 1).T
#                 if torch.mean(sentivity) == 0:
#                     print(bias_grad_dict[bias_key])
#                     # print(new_budget, num_bias)
#                 # noise = torch.randn_like(p.grad) * noise_scale
#                 noise_scale = (sentivity / new_budget).cpu().numpy() + 1e-6
#                 noise = np.random.laplace(0, noise_scale, p.grad.size())
#                 noise = torch.from_numpy(noise).float().to(p.grad.device)
#
#             else:
#                 sentivity = torch.max(p.grad)-torch.min(p.grad)
#                 noise_scale = (sentivity / budget).cpu().numpy() + 1e-6
#                 noise = np.random.laplace(0, noise_scale, p.grad.size())
#                 noise = torch.from_numpy(noise).float().to(p.grad.device)
#             # 添加噪声值
#             p.grad.add_(noise)

def add_noise_to_gradients_sentry_model(model, model_back, budget, new_budget):
    # 不使用梯度，而是使用的是两个Model的差值
    # model_para = model.state_dict()
    # for name in model_para.keys():
    model_back_state = model_back.state_dict()
    model_state = model.state_dict()
    bias_grad_dict = {}
    for name, p in model.named_parameters():
        bias_grad_dict[name] = p - model_back_state[name]  # 存储原始信息
    for name, p in model_state.items():
        # p = model_para[name]
        if p is not None:

            if 'weight' in name:
                bias_key = name.replace('weight', 'bias')
                num_bias = bias_grad_dict[bias_key].size()[0]
                sentivity = torch.abs(bias_grad_dict[bias_key])
                sentivity = sentivity.repeat(p.shape[1], 1).T
                if torch.mean(sentivity) == 0:
                    print(bias_grad_dict[bias_key])
                    # print(new_budget, num_bias)
                # noise = torch.randn_like(p.grad) * noise_scale
                noise_scale = (sentivity / new_budget).detach().cpu().numpy() + 1e-6
                noise = np.random.laplace(0, noise_scale, p.size())
                noise = torch.from_numpy(noise).float().to(p.device)

            else:
                data_p = bias_grad_dict[name]
                sentivity = torch.max(data_p)-torch.min(data_p)
                noise_scale = (sentivity / budget).detach().cpu().numpy() + 1e-6
                noise = np.random.laplace(0, noise_scale, data_p.size())
                noise = torch.from_numpy(noise).float().to(data_p.device)
            # 添加噪声值
            # p.add_(noise)
            model_state[name] = p + noise

    return model_state

# def add_security_sentry(model, privacy_method_budget):
#     epsilon = eval(privacy_method_budget.split(':')[-1])
#     privacy_method = privacy_method_budget.split(':')[0]
#
#     if privacy_method == 'FedAVG':
#         tmp_model = copy.deepcopy(model)
#         # raise NotImplementedError
#     elif 'SENTRY' == privacy_method:
#         tmp_model = copy.deepcopy(model)
#         add_noise_to_gradients_sentry_model(tmp_model, budget=epsilon, new_budget=epsilon*3)
#
#     elif 'CAFL' == privacy_method:
#         # alpha = eval(privacy_method.split(':')[-1])
#         tmp_model = copy.deepcopy(model)
#         tmp_model_state = tmp_model.state_dict()
#         # tmp_model_state_noise = model_add_noise(noise=1 * (1 + alpha) * (1 + np.random.random()), w=tmp_model_state)
#         tmp_model_state_noise = add_noise_to_gradients_cafl_model(tmp_model_state, ori_budget=epsilon)
#         tmp_model.load_state_dict(tmp_model_state_noise)
#
#     elif 'LDP-FL' == privacy_method:
#         # alpha = eval(privacy_method.split(':')[-1])
#         tmp_model = copy.deepcopy(model)
#         tmp_model_state = tmp_model.state_dict()
#         # tmp_model_state_noise = model_add_noise(noise=1 * (1 + alpha) * (1 + np.random.random()), w=tmp_model_state)
#         tmp_model_state_noise = add_noise_to_gradients_cafl_model(tmp_model_state, ori_budget=epsilon)
#         tmp_model.load_state_dict(tmp_model_state_noise)
#
#     elif 'DP' == privacy_method:
#         tmp_model = copy.deepcopy(model)
#         add_noise_to_gradients_dp_model(tmp_model, budget=epsilon)
#     else:
#         raise NotImplementedError
#
#
#     return tmp_model

def add_security_numpy(model, privacy_method, device, dataset_sizes):
    if privacy_method == 'FedAVG':
        tmp_model = copy.deepcopy(model)
        # raise NotImplementedError
    elif 'DP' in privacy_method:
        epsilon = eval(privacy_method.split(':')[-1])
        tmp_model = copy.deepcopy(model)
        # tmp_model_state = tmp_model.state_dict()
        # tmp_model_state_noise, _ = FedHFL_fla_opt([tmp_model_state], a=epsilon)

        noise_scale = Privacy_account_new(num_items_train=dataset_sizes, delta=0.01, privacy_budget=epsilon,
                                          threshold_epochs=20)
        tmp_model_state_noise = noise_add_list(noise_scale, [tmp_model], device=device)

        tmp_model = tmp_model_state_noise[0]
    elif 'PLC' in privacy_method:
        alpha = eval(privacy_method.split(':')[-1])
        tmp_model = copy.deepcopy(model)
        # tmp_model_state = tmp_model.state_dict()
        tmp_model_state_noise = model_add_noise_list(alpha, w=tmp_model)
        tmp_model = tmp_model_state_noise
    else:
        raise NotImplementedError

    return tmp_model

def Privacy_account_new(num_items_train, delta, privacy_budget, threshold_epochs):
    # args中需要用到的参数有args.num_items_train，args.delta，args.privacy_budget
    q_s = 1  # 在本实验中默认使用所有的客户端
    clipthr = 20  # 一般是默认值
    # q_s = args.num_Chosenusers/args.num_users
    # num_items_train是参与者数据集的数据量大小
    delta_s = 2 * clipthr / num_items_train  # 三角s
    noise_scale = delta_s * np.sqrt(2 * q_s * threshold_epochs * np.log(1 / delta)) / privacy_budget
    return noise_scale

def noise_add(noise_scale, w, device):
    w_noise = copy.deepcopy(w)
    if isinstance(w[0], np.ndarray) == True:
        noise = np.random.normal(0, noise_scale, w[0].size())
        w_noise = w_noise + noise
    else:
        for k in range(len(w)):
            for i in w[k].keys():
                noise = np.random.normal(0, noise_scale, w[k][i].size())
                noise = torch.from_numpy(noise).float().to(device)
                w_noise[k][i] = w_noise[k][i] + noise

    return w_noise

def noise_add_list(noise_scale, w, device):
    w_noise = copy.deepcopy(w)
    if isinstance(w[0], np.ndarray) == True:
        noise = np.random.normal(0, noise_scale, w[0].size())
        w_noise = w_noise + noise
    else:
        for k in range(len(w)):
            for i in range(len(w[k])):
                noise = np.random.normal(0, noise_scale, w[k][i].size())
                noise = torch.from_numpy(noise).float().to(device)
                w_noise[k][i] = w_noise[k][i] + noise

    return w_noise

def model_add_noise(noise, w):
    w_noise = copy.deepcopy(w)
    for i in w.keys():
        w_noise[i] = w_noise[i] + noise
    return w_noise

def model_add_noise_alpha(alpha, w):
    w_noise = copy.deepcopy(w)
    # noise = (1 + alpha) * (1+np.random.random())
    noise = 0.5

    for i in w.keys():
        tmp_tensor_fla = torch.flatten(w_noise[i])
        tmp_tensor_num = tmp_tensor_fla.nelement()
        tmp_tensor_num: int

        sel_num = max(int(tmp_tensor_num * alpha), 1)
        tmp_encrypt_idxs = np.random.choice(a=np.arange(tmp_tensor_num), size=sel_num, replace=False)
        tmp_encrypt_idxs = np.sort(tmp_encrypt_idxs)
        tmp_encrypt_value = np.random.random(size=sel_num)
        tmp_encrypt_value = np.round(tmp_encrypt_value, 3)

        for tmp_idx, ind_noise in zip(tmp_encrypt_idxs, tmp_encrypt_value):
            tmp_tensor_fla[tmp_idx] = tmp_tensor_fla[tmp_idx] + ind_noise  # 注意是减去的哦

        tmp_new_tensor = torch.reshape(tmp_tensor_fla, shape=w_noise[i].shape)
        # tmp_new_tensor = torch.add(tmp_new_tensor, tmp_sum_noise)  # 聚合噪声值
        tmp_new_tensor = torch.add(tmp_new_tensor, noise)
        w_noise[i] = copy.deepcopy(tmp_new_tensor)
    return w_noise

def model_add_noise_list(alpha, w):
    w_noise = copy.deepcopy(w)
    noise = 1 * (1 + alpha) * (10 + np.random.random())
    for i in range(len(w_noise)):

        tmp_tensor_fla = torch.flatten(w_noise[i])
        tmp_tensor_num = tmp_tensor_fla.nelement()
        tmp_tensor_num: int

        sel_num =  max(int(tmp_tensor_num * alpha), 1)
        tmp_encrypt_idxs = np.random.choice(a=np.arange(tmp_tensor_num), size=sel_num, replace=False)
        tmp_encrypt_idxs = np.sort(tmp_encrypt_idxs)
        tmp_encrypt_value = np.random.random(size=sel_num) * 20
        tmp_encrypt_value = np.round(tmp_encrypt_value, 3)

        for tmp_idx, ind_noise in zip(tmp_encrypt_idxs, tmp_encrypt_value):
            tmp_tensor_fla[tmp_idx] = tmp_tensor_fla[tmp_idx] + ind_noise  # 注意是减去的哦

        tmp_new_tensor = torch.reshape(tmp_tensor_fla, shape=w_noise[i].shape)
        # tmp_new_tensor = torch.add(tmp_new_tensor, tmp_sum_noise)  # 聚合噪声值
        tmp_new_tensor = torch.add(tmp_new_tensor, noise)
        w_noise[i] = copy.deepcopy(tmp_new_tensor)

    return w_noise

def noise_add_laplace(budget, w, device, global_model):
    # 1. 更改那个敏感度为两个模型参数梯度的差值的模；
    # 2. 需要接收可变的隐私预算变量

    w_noise = copy.deepcopy(w)
    if isinstance(w[0], np.ndarray) == True:
        # sentivity = np.linalg.norm(w[0]-global_model)
        sentivity = torch.dist(w[0].float(), global_model.float())
        noise_scale = sentivity/budget
        noise = np.random.laplace(0, noise_scale.item(), w[0].size())
        w_noise = w_noise + noise

    else:
        for k in range(len(w)):
            if isinstance(budget, list):
                new_budget = budget[k]
            else:
                new_budget = budget
            for i in w[k].keys():
                # sentivity = np.linalg.norm(w[k][i] - global_model[i])
                sentivity2 = torch.dist(w[k][i].float(), global_model[i].float())
                gradients_value = w[k][i] - global_model[i]
                sentivity = torch.max(gradients_value.float()) - torch.min(gradients_value.float())
                noise_scale = (sentivity / new_budget).item()
                # noise_scale = 2 / new_budget
                noise = np.random.laplace(0, noise_scale, w[k][i].size())
                noise = torch.from_numpy(noise).float().to(device)
                w_noise[k][i] = w_noise[k][i] + noise
                print('sentivity2:{}, sentivity: {}, noise_scale, {}, noise_mean:{}'.format(sentivity2, sentivity, noise_scale, new_budget, torch.mean(noise)))
    return w_noise

# def calc_budget(num_batchs, max_error):

    # num_batchs = len(self.ldr_train)
    # assert self.tmp_error_sum > 0, print('error {}'.format(self.tmp_error_sum))
    # budget = max_error*num_batchs*self.args.local_ep/self.tmp_error_sum * self.budget
    # assert budget > 0, print('budget:{}, self.max_error:{}, num_batchs:{}, self.args.local_ep:{}, self.tmp_error_sum:{}, '
    #                          'self.budget:{}'.format(budget, self.max_error, num_batchs, self.args.local_ep, self.tmp_error_sum, self.budget))

    # return budget
# 含有named_parameters版本的add noise
def add_noise_to_gradients(model, budget, ori_budget):
    # model_para = model.state_dict()
    # for name in model_para.keys():
    bias_grad_dict = {}
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_grad_dict[name] = p.grad  # 存储原始信息
    for name, p in model.named_parameters():
        # p = model_para[name]
        if p.grad is not None:

            if 'weight' in name:
                bias_key = name.replace('weight', 'bias')
                num_bias = bias_grad_dict[bias_key].size()[0]
                sentivity = torch.abs(bias_grad_dict[bias_key])
                sentivity = sentivity.repeat(p.shape[1], 1).T
                if torch.mean(sentivity) == 0:
                    print(bias_grad_dict[bias_key])
                    # print(new_budget, num_bias)
                # noise = torch.randn_like(p.grad) * noise_scale
                # noise_scale = (sentivity / self.budget).cpu().numpy() + 1e-6
                noise_scale = (sentivity / ori_budget).cpu().numpy() + 1e-6
                noise = np.random.laplace(0, noise_scale, p.grad.size())
                noise = torch.from_numpy(noise).float().to(p.grad.device)

            else:
                sentivity = torch.max(p.grad)-torch.min(p.grad)
                noise_scale = (sentivity / budget).cpu().numpy() + 1e-6
                noise = np.random.laplace(0, noise_scale, p.grad.size())
                noise = torch.from_numpy(noise).float().to(p.grad.device)
            # 添加噪声值
            p.grad.add_(noise)

def add_noise_cafl(y, epsilon_user):
    # if method == 'sadp':
    #     assert similarity is not None, print('similarity cannot be None!!')
    #     epsilon_user = epsilon_user - math.log(1-similarity)
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
    epislon_0 = (epislon - 1) / (epislon + 1)
    # print('epislon_1 is {}'.format(epislon_1))
    y_test = y.clone()
    y_test[u_bool] = miu[u_bool] * epislon_1 - miu[u_bool]
    y_test[~u_bool] = miu[~u_bool] - miu[~u_bool] * epislon_0
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

# 不含有named_parameters版本的add noise
def add_noise_to_gradients_dy_dx(original_grident, budget, ori_budget):
    # model_para = model.state_dict()
    # for name in model_para.keys():
    bias_grad_dict = {}
    # for name, p in model.named_parameters():
    for key, dy_dx in original_grident.items():
        # dy_dx =
        ind = eval(key)
        if ind%2==0:
            cnt = ind/2
            bias_grad_dict[cnt] = dy_dx  # 存储原始信息
    # for name, p in model.named_parameters():
    for key, dy_dx in original_grident.items():
        ind = eval(key)

        # p = model_para[name]
        if dy_dx is not None:

            if ind%2 == 1:
                # bias_key = name.replace('weight', 'bias')
                # num_bias = bias_grad_dict[bias_key].size()[0]

                tmp_cnt = (ind+1)/2
                sentivity = torch.abs(bias_grad_dict[tmp_cnt])
                sentivity = sentivity.repeat(dy_dx.shape[1], 1).T
                if torch.mean(sentivity) == 0:
                    print(bias_grad_dict[tmp_cnt])
                    # print(new_budget, num_bias)
                # noise = torch.randn_like(p.grad) * noise_scale
                # noise_scale = (sentivity / self.budget).cpu().numpy() + 1e-6
                noise_scale = (sentivity / ori_budget).cpu().numpy() + 1e-6
                noise = np.random.laplace(0, noise_scale, dy_dx.size())
                noise = torch.from_numpy(noise).float().to(dy_dx.device)

            else:
                sentivity = torch.max(dy_dx)-torch.min(dy_dx)
                noise_scale = (sentivity / budget).cpu().numpy() + 1e-6
                noise = np.random.laplace(0, noise_scale,dy_dx.size())
                noise = torch.from_numpy(noise).float().to(dy_dx.device)
            # 添加噪声值
            dy_dx.add_(noise)

def add_noise_to_gradients_dy_dx_cafl(original_grident, ori_budget):
    grident_back = copy.deepcopy(original_grident)
    error_sum = 0
    for key, dy_dx in original_grident.items():
        dy_dx = add_noise_cafl(dy_dx, ori_budget)
        original_grident[key] = dy_dx

    for key, dy_dx in original_grident.items():
        error = torch.sqrt(torch.sum((grident_back[key] - dy_dx) ** 2))
        error_sum = error_sum + error
    print('error_sum: {}'.format(error_sum))

# def add_noise_to_gradients_cafl_model(original_grident, ori_budget):
#     grident_back = copy.deepcopy(original_grident)
#     error_sum = 0
#     for key, dy_dx in original_grident.items():
#         dy_dx = add_noise_cafl(dy_dx, ori_budget)
#         original_grident[key] = dy_dx
#
#     for key, dy_dx in original_grident.items():
#         error = torch.sqrt(torch.sum((grident_back[key] - dy_dx) ** 2))
#         error_sum = error_sum + error
#     print('error_sum: {}'.format(error_sum))
#     model, model_back, budget,
def add_noise_to_gradients_cafl_model(model, model_back, ori_budget):
    # grident_back = model.state_dict()

    model_back_state = model_back.state_dict()
    model_state = model.state_dict()
    bias_grad_dict = {}
    for name, p in model.named_parameters():
        bias_grad_dict[name] = p - model_back_state[name]  # 存储原始信息
    for name, p in bias_grad_dict.items():
        # p = model_para[name]
        dy_dx = add_noise_cafl(p, ori_budget)
        model_state[name] = model_state[name] + dy_dx
        # model_state[name] = p + noise
    error_sum = 0
    for key, dy_dx in model_state.items():
        error = torch.sqrt(torch.sum((model_back_state[key] - dy_dx) ** 2))
        error_sum = error_sum + error
    print('error_sum: {}'.format(error_sum))
    return model_state

def add_noise_to_gradients_dy_dx_opacus(original_grident, ori_budget):
    grident_back = copy.deepcopy(original_grident)
    error_sum = 0
    for key, dy_dx in original_grident.items():
        min_weight = torch.min(dy_dx)
        max_weight = torch.max(dy_dx)
        sentivity = (max_weight - min_weight).item()
        size =dy_dx.size()
        noise_tmp = torch.from_numpy(np.random.laplace(0, sentivity/ori_budget, size)).to(dy_dx.device)

        original_grident[key] = dy_dx + noise_tmp

    for key, dy_dx in original_grident.items():
        error = torch.sqrt(torch.sum((grident_back[key] - dy_dx) ** 2))
        error_sum = error_sum + error
    print('error_sum: {}'.format(error_sum))

def noise_add_laplace_sentry(budget, w, device, global_model, data_num = 1):
    # 1. 更改那个敏感度为两个模型参数梯度的差值的模；
    # 2. 需要接收可变的隐私预算变量

    w_noise = copy.deepcopy(w)
    if isinstance(w[0], np.ndarray) == True:
        # sentivity = np.linalg.norm(w[0]-global_model)
        sentivity = torch.dist(w[0].float(), global_model.float())
        noise_scale = sentivity/budget
        noise = np.random.laplace(0, noise_scale.item(), w[0].size())
        w_noise = w_noise + noise

    else:
        for k in range(len(w)):
            if isinstance(budget, list):
                new_budget = budget[k]
            else:
                new_budget = budget
            for i in w[k].keys():
                if 'weight' in i:

                # sentivity = np.linalg.norm(w[k][i] - global_model[i])
                # sentivity2 = torch.dist(w[k][i].float(), global_model[i].float())
                    bias_key = i.replace('weight', 'bias')
                    # gradients_value = w[k][i] - global_model[i]
                    num_bias = w[k][bias_key].size()[0]
                    sentivity = torch.abs(w[k][bias_key]-global_model[bias_key])/data_num

                    sentivity = sentivity.repeat(w[k][i].shape[1], 1).T
                    if torch.mean(sentivity) == 0:
                        print(w[k][bias_key])
                        print(new_budget, num_bias)
                    noise_scale = (sentivity * num_bias / new_budget).cpu().numpy()+1e-6
                    # noise_scale = (sentivity / new_budget).item()
                    # noise_scale = 0.1 / new_budget
                    noise = np.random.laplace(0, noise_scale, w[k][i].size())
                    noise = torch.from_numpy(noise).float().to(device)
                    w_noise[k][i] = w_noise[k][i] + noise
                    # print('sentivity2:{}, sentivity: {}, noise_scale, {}, noise_mean:{}'.format(sentivity2, sentivity, noise_scale, new_budget, torch.mean(noise)))
                    print('sentivity: {}, noise_scale, {}, noise_mean:{}'.format(torch.mean(sentivity), np.mean(noise_scale), new_budget, torch.mean(noise)))

    return w_noise

def sadp_noise_add(similarity, w, ori_model_state, epsilon):
    w_noise = copy.deepcopy(w)
    for k in range(len(w)):
        model = w_noise[k]
        updated_model_state = copy.deepcopy(model)
        dev_model_state = get_dev_model(ori_model_state=ori_model_state, updated_model_state=updated_model_state)
        dev_model_state_noise = sadp(w=copy.deepcopy(dev_model_state), epsilon_user=epsilon, similar=similarity)
        w_noise[k] = agg(w=ori_model_state, dev_model_state=dev_model_state_noise)
        # model.load_state_dict(updated_model_state_noise)
    return w_noise


def get_avg_error(global_true, global_fed):
    # 求聚合后的模型与fed avg模型的差值
    w_avg = copy.deepcopy(global_true)
    erro_list = []
    num_para = 0
    for k in w_avg.keys():
        # gradients
        error_tensor = torch.abs(w_avg[k] - global_fed[k])
        error_sum = torch.sum(error_tensor).item()
        erro_list.append(error_sum)
        num_para += torch.numel(error_tensor)

    return sum(erro_list)/num_para


def get_memory_two_phase(w, comm_parts, unit='Mb'):
    # total_mem_con = 0
    total_parts = len(w)
    tmp_para = copy.deepcopy(w[0])
    # transfer_dict = {}

    tmp_mem = get_memory_para(tmp_para)
    total_mem_con = tmp_mem * (total_parts * comm_parts + (comm_parts - 1) * comm_parts + total_parts)

    # 需要将all_encrypt_tensor和encrypt_sum转换为list和dict才能计算变量大小和数量
    # total_mem_con += total_size(all_encrypt_tensor, verbose=True)
    # total_mem_con += total_size(encrypt_sum[0], verbose=True) * (comm_parts-1)  # 因为需要将本地聚合后的模型分享给其它committee member
    if unit == 'Mb':
        total_mem_con = total_mem_con / 1024 / 1024
    return total_mem_con


def get_memory_hfl_fla(all_encrypt_tensor, all_noise_shares, encrypt_new_w, unit='Mb'):
    # 解析出来第一个，然后直接相乘即可
    total_parts = len(all_encrypt_tensor)
    noise_mem = []
    base_noise_mem = []
    weight_mem = []
    for part in range(total_parts):
        for com_part in range(total_parts - 1):
            tmp_para = all_encrypt_tensor[part][com_part]
            tmp_para[0] = tmp_para[0].tolist()
            tmp_para[1] = tmp_para[1].tolist()
            tmp_mem = total_size(tmp_para)
            noise_mem.append(tmp_mem)
            # 求基础噪声所占的数据大小
        base_tmp_mem = get_memory_base_noise(noise_share=all_noise_shares[part])
        base_noise_mem.append(base_tmp_mem)
        weight_tmp_mem = get_memory_para(encrypt_new_w[part])
        weight_mem.append(weight_tmp_mem)
    total_mem = sum(noise_mem) + sum(base_noise_mem) + 2 * sum(weight_mem)
    if unit == 'Mb':
        total_mem = total_mem / 1024 / 1024
    return total_mem


def get_memory_hfl(all_encrypt_tensor, all_noise_shares, encrypt_new_w, unit='Mb'):
    # 解析出来第一个，然后直接相乘即可
    total_parts = len(all_encrypt_tensor)
    noise_mem = []
    base_noise_mem = []
    weight_mem = []
    for part in range(total_parts):
        for com_part in range(total_parts - 1):
            tmp_para = all_encrypt_tensor[part][com_part]
            tmp_mem = get_memory_noise(tmp_para)
            noise_mem.append(tmp_mem)
            # 求基础噪声所占的数据大小
        base_tmp_mem = get_memory_base_noise(noise_share=all_noise_shares[part])
        base_noise_mem.append(base_tmp_mem)
        weight_tmp_mem = get_memory_para(encrypt_new_w[part])
        weight_mem.append(weight_tmp_mem)
    total_mem = sum(noise_mem) + sum(base_noise_mem) + 2 * sum(weight_mem)
    if unit == 'Mb':
        total_mem = total_mem / 1024 / 1024
    return total_mem


def get_memory_para(w):
    # 先将参数装换为dict和list组合的结构，方便与内存同济
    tmp_para = copy.deepcopy(w)
    transfer_dict = {}

    for k in tmp_para.keys():
        if tmp_para[k].device == 'cpu':
            transfer_dict[k] = tmp_para[k].numpy().tolist()
        else:
            transfer_dict[k] = tmp_para[k].cpu().numpy().tolist()
    tmp_mem = total_size(transfer_dict, verbose=False)
    return tmp_mem


def get_memory_base_noise(noise_share):
    tmp_noise_share = noise_share.tolist()
    tmp_mem = total_size(noise_share, verbose=False)
    return tmp_mem


def get_memory_noise(w_noise):
    # 先将参数装换为dict和list组合的结构，方便与内存同济
    tmp_para = copy.deepcopy(w_noise)
    transfer_dict = {}
    for k in tmp_para.keys():
        transfer_dict[k] = []
        transfer_dict[k].append(tmp_para[k][0].tolist())
        transfer_dict[k].append(tmp_para[k][1].tolist())

    tmp_mem = total_size(transfer_dict, verbose=False)
    return tmp_mem


def FedAvg(w):
    time_start = time.time()
    mem_consumer = {}
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    time_end = time.time()
    w_mem = get_memory_para(w_avg) * len(w) / 1024 / 1024
    mem_consumer['total_mem_cons'] = w_mem
    mem_consumer['total_time_cons'] = time_end - time_start
    return w_avg, mem_consumer


def get_all_data(w):
    # 就是将所有的权重数据按照一定的顺序取出来，组成一维的数据--方便加密和聚合
    w_tmp = copy.deepcopy(w)
    res = []
    for i in range(0, len(w)):
        tmp_par = []
        for k in w[i].keys():
            tmp_par.append(w[i][k])


def return_weight():
    # 依据 get_all_data，按顺序将数据还原成权重
    pass


def generate_arithmetic(local_data, word_size):
    @mpc.run_multiprocess(world_size=word_size)
    def examine_arithmetic_shares(local_data):
        x_enc = crypten.cryptensor(local_data, ptype=crypten.mpc.arithmetic)
        # print(x_enc.share)
        # rank = comm.get().get_rank()
        # if rank == 0:
        #     print('current rank is {}'.format(rank))
        # crypten.print(f"\nRank {rank}:\n {x_enc}\n", in_order=True)
        return_value = x_enc.share.numpy()
        return return_value

    res_value = examine_arithmetic_shares(local_data)
    return res_value


def generate_arithmetic_noise(local_data, word_size):
    # 通过噪声的方式来生成安全共享---local data的Word_size个安全共享
    data_shape = local_data.shape
    res_value = []
    final_share = copy.deepcopy(local_data.numpy())
    for com_part in range(word_size - 1):
        tmp_share = 2 * np.random.random(size=data_shape) - 1
        tmp_share = np.round(tmp_share, decimals=3)
        final_share = final_share - tmp_share
        res_value.append(np.array(tmp_share))
    res_value.append(np.array(final_share))

    # 用来测试生成的安全共享是否符合要求
    # test_tensor = copy.deepcopy(res_value[0])
    # for i in range(1, len(res_value)):
    #     test_tensor = test_tensor + res_value[i]
    # error_values = torch.abs(local_data - torch.from_numpy(np.array(test_tensor))
    # error_sum = torch.sum(error_values)

    return res_value


def FedTwoPhrase(w_locals, c=2):  # 指的是选择的个数或者比例
    ## 工作流程
    # 生成c个安全共享，并发送给各个committee成员--计算总的数据流
    # committee 接收安全共享，使用数组
    # committee 对接收到的安全共享求和
    # 发送给其它committee member---计算总的数据流
    # 对所有的共享数据进行求和，然后再求平均，进而得到最终的参数
    time_start = time.time()
    w = copy.deepcopy(w_locals)
    total_mem_con = 0
    mem_consumer = {}
    encoder = FixedPointEncoder()
    total_parts = len(w)
    if c < 1:
        comm_parts = max(int(total_parts * c), 3)  # 这里最少是3方
    else:
        comm_parts = int(c)

    assert comm_parts > 1, print('the setting of c is not appropriate')
    names = list(str(i) for i in range(comm_parts))  # committee 客户端的名称
    crypten.init()
    # rank = crypten.communicator.get().get_rank()  # 本机的rank
    all_encrypt_tensor = []  # 里面的每一个元素表示对应committee member的安全共享
    # 生成c个安全共享，并发送给各个committee成员--计算总的数据流 --- encrypt_tensor = w[:comm_parts]
    for part in range(total_parts):
        local_data = w[part]  # 表示第part个用户的参数信息---
        encrypt_tensor = w[:comm_parts]  # 里面的每一个元素表示对应committee member的安全共享
        local_data: dict
        # encrypt_local_data = []  # 里面的每一个元素表示对应committee member的安全共享
        for k in local_data.keys():
            tmp_tensor = local_data[k]
            comm_parts_encrypts = generate_arithmetic(local_data=tmp_tensor, word_size=comm_parts)
            for com_part in range(comm_parts):
                # 得到第 com_part个的加密参数
                tmp_encrypt = torch.from_numpy(comm_parts_encrypts[com_part])  # 第com_part个加密的数据

                encrypt_tensor[com_part][k] = copy.deepcopy(tmp_encrypt)  # 对应位置的数据改为加密的参数
            # # 验证代码---start
            # tmp_tensor2 = encrypt_tensor[0][k]
            # for com_part in range(1, comm_parts):
            #     tmp_tensor2 += encrypt_tensor[com_part][k]
            #
            # # de_tmp_tensor = tmp_tensor2.get_plain_text()  # 必须的是cryptensor类型的才可以直接使用get_plain_text
            # de_tmp_tensor = encoder.decode(tensor=tmp_tensor2)
            # # 输出总的误差值
            # error_mpc = torch.sum(torch.abs(tmp_tensor - de_tmp_tensor))
            # print('error_mpc:')
            # print(error_mpc)
            # # 验证代码---end
        all_encrypt_tensor.append(encrypt_tensor)  # 里面一共有total_parts个元素，每个元素代表本地数据针对所有comm_parts的加密数据
    # committee 接收安全共享，使用数组---比较难以理解诶
    receive_encrypt_tensor = []
    for com_part in range(comm_parts):
        tmp_encrypt_comm = []
        for part in range(total_parts):
            tmp_encrypt_comm.append(
                all_encrypt_tensor[part][com_part])  # all_encrypt_tensor[part][com_part]代表成员part的数据针对所有comm_part的加密数据
        receive_encrypt_tensor.append(tmp_encrypt_comm)
    # committee 对接收到的安全共享求和---直接相加会溢出吗？--这个只是用来测试数据量的代码的，可以使用crypten来计算，就可以避免溢出了的--得到各个committee member中的参数
    encrypt_sum = []
    for com_part in range(comm_parts):
        received_data = receive_encrypt_tensor[com_part]
        w_sum = copy.deepcopy(received_data[0])
        for k in w_sum.keys():
            for part in range(1, total_parts):
                w_sum[k] += received_data[part][k]
        encrypt_sum.append(w_sum)

    # 发送给其它committee member 得到 encrypt_sum ---计算总的数据流--w_avg即为所求的模型聚合值
    w_avg = copy.deepcopy(encrypt_sum[0])
    for k in w_avg.keys():
        for com_part in range(1, comm_parts):
            w_avg[k] += encrypt_sum[com_part][k]
        w_avg[k] = encoder.decode(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], len(w))
    time_end = time.time()

    # 自己总结一下总的消耗就行
    total_mem_con = get_memory_two_phase(w, comm_parts)
    mem_consumer['total_mem_cons'] = total_mem_con
    mem_consumer['total_time_cons'] = time_end - time_start
    return w_avg, mem_consumer


def FedTwoPhraseNoise(w_locals, c=2):  # 指的是选择的个数或者比例
    ## 工作流程
    # 生成c个安全共享，并发送给各个committee成员--计算总的数据流
    # committee 接收安全共享，使用数组
    # committee 对接收到的安全共享求和
    # 发送给其它committee member---计算总的数据流
    # 对所有的共享数据进行求和，然后再求平均，进而得到最终的参数
    time_start = time.time()
    torch.set_printoptions(precision=16)
    mem_consumer = {}
    w = copy.deepcopy(w_locals)
    total_parts = len(w)
    if c < 1:
        comm_parts = max(int(total_parts * c), 3)  # 这里最少是3方
    else:
        comm_parts = int(c)

    assert comm_parts > 1, print('the setting of c is not appropriate')
    names = list(str(i) for i in range(comm_parts))  # committee 客户端的名称
    # rank = crypten.communicator.get().get_rank()  # 本机的rank
    all_encrypt_tensor = []  # 里面的每一个元素表示对应committee member的安全共享
    # 生成c个安全共享，并发送给各个committee成员--计算总的数据流 --- encrypt_tensor = w[:comm_parts]
    for part in range(total_parts):
        local_data = w[part]  # 表示第part个用户的参数信息---
        encrypt_tensor = w[:comm_parts]  # 里面的每一个元素表示对应committee member的安全共享
        local_data: dict
        # encrypt_local_data = []  # 里面的每一个元素表示对应committee member的安全共享
        for k in local_data.keys():
            tmp_tensor = local_data[k]
            comm_parts_encrypts = generate_arithmetic_noise(local_data=tmp_tensor, word_size=comm_parts)
            for com_part in range(comm_parts):
                # 得到第 com_part个的加密参数
                tmp_encrypt = torch.from_numpy(comm_parts_encrypts[com_part])  # 第com_part个加密的数据

                encrypt_tensor[com_part][k] = copy.deepcopy(tmp_encrypt)  # 对应位置的数据改为加密的参数
            # # 验证代码---start
            # tmp_tensor2 = encrypt_tensor[0][k]
            # for com_part in range(1, comm_parts):
            #     tmp_tensor2 += encrypt_tensor[com_part][k]
            #
            # # de_tmp_tensor = tmp_tensor2.get_plain_text()  # 必须的是cryptensor类型的才可以直接使用get_plain_text
            # de_tmp_tensor = encoder.decode(tensor=tmp_tensor2)
            # # 输出总的误差值
            # error_mpc = torch.sum(torch.abs(tmp_tensor - de_tmp_tensor))
            # print('error_mpc:')
            # print(error_mpc)
            # # 验证代码---end
        all_encrypt_tensor.append(encrypt_tensor)  # 里面一共有total_parts个元素，每个元素代表本地数据针对所有comm_parts的加密数据
    # committee 接收安全共享，使用数组---比较难以理解诶
    receive_encrypt_tensor = []
    for com_part in range(comm_parts):
        tmp_encrypt_comm = []
        for part in range(total_parts):
            tmp_encrypt_comm.append(
                all_encrypt_tensor[part][com_part])  # all_encrypt_tensor[part][com_part]代表成员part的数据针对所有comm_part的加密数据
        receive_encrypt_tensor.append(tmp_encrypt_comm)
    # committee 对接收到的安全共享求和---直接相加会溢出吗？--这个只是用来测试数据量的代码的，可以使用crypten来计算，就可以避免溢出了的--得到各个committee member中的参数
    encrypt_sum = []
    for com_part in range(comm_parts):
        received_data = receive_encrypt_tensor[com_part]
        w_sum = copy.deepcopy(received_data[0])
        for k in w_sum.keys():
            for part in range(1, total_parts):
                w_sum[k] += received_data[part][k]
        encrypt_sum.append(copy.deepcopy(w_sum))

    # 发送给其它committee member 得到 encrypt_sum ---计算总的数据流--w_avg即为所求的模型聚合值
    for com_part in range(comm_parts):
        w_avg = copy.deepcopy(encrypt_sum[0])
        for k in w_avg.keys():
            for com_part in range(1, comm_parts):
                w_avg[k] += encrypt_sum[com_part][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    time_end = time.time()
    # 自己总结一下总的消耗就行---为什么会与实际值差别那么大呢？
    total_mem_con = get_memory_two_phase(w, comm_parts)
    mem_consumer['total_mem_cons'] = total_mem_con
    mem_consumer['total_time_cons'] = time_end - time_start
    return w_avg, mem_consumer


def FedMpc(w):
    ## 工作流程
    # 生成len(w)个安全共享，并发送给其它成员--计算总的数据流
    # 接收安全共享，使用数组
    # 对接收到的安全共享求和
    # 发送给其它committee member---计算总的数据流
    # 对所有的共享数据进行求和，然后再求平均，进而得到最终的参数
    # w_avg, mem_consumer = FedTwoPhrase(w, c=len(w))
    w_avg, mem_consumer = FedTwoPhraseNoise(w, c=len(w))

    return w_avg, mem_consumer


def get_para_num(w):
    # 得到模型参数的数据总量
    total = sum([param.nelement() for param in w.parameters()])
    return total


def FedHFL(w_locals, a=0.02):  # a指的是加密参数的比例
    ## 工作流程
    # 对所有参数添加噪声（后期会消掉的）
    # 生成len(w)个安全共享，并发送给其它成员--计算总的数据流
    # 接收安全共享，使用数组
    # 对接收到的安全共享求和
    # 发送给其它committee member---计算总的数据流
    # 对所有的共享数据进行求和，然后再求平均，进而得到最终的参数
    # 得到所有参数的个数，然后对应选择固定数量的索引值
    time_start = time.time()
    torch.set_printoptions(precision=16)
    total_mem_con = 0
    mem_consumer = {}
    w = copy.deepcopy(w_locals)
    total_parts = len(w)  # 所有参与者的数量
    assert a < 1, print('the setting for a is not appropriate')
    # 先加噪声，再根据a 选择参数
    noise_list = np.random.random(total_parts)
    noise_list = np.around(noise_list, 3)
    # rank = crypten.communicator.get().get_rank()  # 本机的rank
    all_encrypt_tensor = []  # 里面的每一个元素表示对应committee member的安全共享
    new_w = copy.deepcopy(w)  # 里面的每一个元素表示对应committee member的安全共享
    all_noise_shares = []
    for part in range(total_parts):
        tmp_noise = noise_list[part]
        local_data = w[part]  # 表示第part个用户的参数信息---
        encrypt_tensor = copy.deepcopy(w[:total_parts - 1])  # 里面的每一个元素表示对应committee member的安全共享
        local_data: dict  # 注意：如果为了提高精度，可以选择只保留其中几位小数
        tmp_noise_shares = -1 + 2 * np.random.random(size=(total_parts - 2))
        tmp_noise_shares = np.around(tmp_noise_shares, 3)
        tmp_final_noise = 0 - (np.sum(tmp_noise_shares) + tmp_noise)
        tmp_noise_shares: np.array
        tmp_noise_shares = np.append(tmp_noise_shares, tmp_final_noise)  # 这个相当于基础随机数的安全共享内容
        all_noise_shares.append(tmp_noise_shares)
        # encrypt_local_data = []  # 里面的每一个元素表示对应committee member的安全共享
        for k in local_data.keys():
            tmp_tensor = copy.deepcopy(local_data[k])
            tmp_tensor = torch.add(tmp_tensor, tmp_noise)  # 添加噪声---这是基础噪声
            # comm_parts_encrypts = generate_arithmetic(local_data=tmp_tensor, word_size=comm_parts)
            tmp_tensor: torch.Tensor
            tmp_tensor_fla = torch.flatten(tmp_tensor)
            for com_part in range(total_parts - 1):  # 因为只需要安全共享给其它total_parts-1个clients
                # 根据a选择需要加密的参数
                sel_num = max(int(tmp_tensor_fla.shape[-1] * a), 1)  # 最少选择一个参数进行加密
                # 记录要加密的key和平整化后的索引值，以及加密后的数值--就是随机数
                encrypt_idxs = np.random.choice(a=np.arange(tmp_tensor_fla.shape[-1]), size=sel_num, replace=False)
                # encrypt_idxs = np.random.choice(low=0, high=tmp_tensor_fla.shape[-1], size=sel_num)   # 平整化后的索引值---注意，这样做会有重复索引被选择的哦
                encrypt_value = np.random.random(size=sel_num)
                encrypt_value = np.around(encrypt_value, 3)
                # 原始数据减去对应的随机数
                for ind, ind_noise in zip(encrypt_idxs, encrypt_value):
                    tmp_tensor_fla[ind] = tmp_tensor_fla[ind] - ind_noise
                # 需要保存的哦
                encrypt_tensor[com_part][k] = [encrypt_idxs, encrypt_value]
            tmp_new_tensor = torch.reshape(tmp_tensor_fla, shape=tmp_tensor.shape)
            new_w[part][k] = copy.deepcopy(tmp_new_tensor)  # 对应位置的数据改为加密后的参数---代表自己本地的数据，所以一共有total_parts个数据
            # # 验证代码---start--暂时未完成
            # tmp_tensor2 = encrypt_tensor[0][k]
            # for com_part in range(1, comm_parts):
            #     tmp_tensor2 += encrypt_tensor[com_part][k]
            #
            # # de_tmp_tensor = tmp_tensor2.get_plain_text()  # 必须的是cryptensor类型的才可以直接使用get_plain_text
            # de_tmp_tensor = encoder.decode(tensor=tmp_tensor2)
            # # 输出总的误差值
            # error_mpc = torch.sum(torch.abs(tmp_tensor - de_tmp_tensor))
            # print('error_mpc:')
            # print(error_mpc)
            # # 验证代码---end
        all_encrypt_tensor.append(encrypt_tensor)  # 里面一共有total_parts个元素，每个元素代表本地数据针对所有comm_parts的加密数据
    # committee 接收安全共享，使用数组---比较难以理解诶
    receive_encrypt_tensor = []  # 将对应的随机数发送给其他人
    receive_encrypt_noise = []  # 将基础随机数发送给其他人
    indicator_index = [0] * total_parts
    for part in range(total_parts):  # 每一个part都会发送给其它人发来的共享（total_parts-1）
        tmp_encrypt_comm = []
        receive_tmp_noises = []
        for com_part in range(total_parts):  # 因为总共有total_parts-1发来安全共享
            if part == com_part:  ## 说明要越过当前client，因为不能自己给自己发数据嘛，以后的索引值要减去1，
                continue  # 不能接受自己本身的数据嘛
            else:
                tmp_encrypt_comm.append(  # com_part是有total parts个的，而发来的共享是有 (total_parts-1)的
                    all_encrypt_tensor[com_part][
                        indicator_index[com_part]])  # all_encrypt_tensor[part][com_part]代表成员part的数据针对所有comm_part的加密数据

                receive_tmp_noises.append(all_noise_shares[com_part][indicator_index[com_part]])
                indicator_index[com_part] += 1
        receive_encrypt_tensor.append(tmp_encrypt_comm)  # total_parts * (total_parts-1)
        receive_encrypt_noise.append(receive_tmp_noises)

    # 接收端对噪声求和
    sum_list = np.sum(np.array(receive_encrypt_noise), axis=1)
    # committee 对接收到的安全共享求和---直接相加会溢出吗---因为是小于1的随机数，是不会溢出的
    encrypt_new_w = copy.deepcopy(new_w)
    for com_part in range(total_parts):
        received_data = receive_encrypt_tensor[com_part]  # 接收到其他人发来的加密数据---一组随机数
        new_local_data = encrypt_new_w[com_part]  # 使用接收到的噪声对当前数据集进行加密
        tmp_sum_noise = sum_list[com_part]
        for k in new_local_data.keys():
            tmp_tensor = new_local_data[k]
            tmp_tensor_fla = torch.flatten(tmp_tensor)
            for part in range(0, total_parts - 1):
                tmp_encrypt_idxs = received_data[part][k][0]
                tmp_encrypt_value = received_data[part][k][1]
                for tmp_idx, ind_noise in zip(tmp_encrypt_idxs, tmp_encrypt_value):
                    tmp_tensor_fla[tmp_idx] = tmp_tensor_fla[tmp_idx] + ind_noise
            tmp_new_tensor = torch.reshape(tmp_tensor_fla, shape=tmp_tensor.shape)
            tmp_new_tensor = torch.add(tmp_new_tensor, tmp_sum_noise)  # 聚合噪声值
            encrypt_new_w[com_part][k] = copy.deepcopy(tmp_new_tensor)  # 对应位置的数据改为加密后的参数
    # 发送给服务器 得到 encrypt_sum ---计算总的数据流--w_avg即为所求的模型聚合值
    w_avg = copy.deepcopy(encrypt_new_w[0])
    for k in w_avg.keys():
        for com_part in range(1, total_parts):
            w_avg[k] += encrypt_new_w[com_part][k]
        w_avg[k] = w_avg[k] / total_parts
    time_end = time.time()
    # 总的消耗
    total_mem_con = get_memory_hfl(all_encrypt_tensor, all_noise_shares, encrypt_new_w)
    mem_consumer['total_mem_cons'] = total_mem_con
    mem_consumer['total_time_cons'] = time_end - time_start
    return w_avg, mem_consumer


def FedHFL_fla(w_locals, a=0.02):  # a指的是加密参数的比例
    ## 工作流程
    # 对所有参数添加噪声（后期会消掉的）
    # 生成len(w)个安全共享，并发送给其它成员--计算总的数据流
    # 接收安全共享，使用数组
    # 对接收到的安全共享求和
    # 发送给其它committee member---计算总的数据流
    # 对所有的共享数据进行求和，然后再求平均，进而得到最终的参数
    # Note: 展开所有的参数值，然后选择对应比例的索引
    # 这部分内容没必要总是计算的
    # 代码冗余的地方，就是在本地执行了两次更改权重值的操作，使得执行时间大幅增加--------注意
    para_num = sum([w_locals[0][k].nelement() for k in w_locals[0].keys()])
    sel_num = max(int(para_num * a), 1)  # 最少选择一个参数进行加密
    time_start = time.time()
    mem_consumer = {}
    w = copy.deepcopy(w_locals)
    total_parts = len(w)  # 所有参与者的数量
    assert a < 1, print('the setting for a is not appropriate')
    # 先加噪声，再根据a 选择参数
    noise_list = np.random.random(total_parts)
    noise_list = np.around(noise_list, 3)
    # rank = crypten.communicator.get().get_rank()  # 本机的rank
    all_encrypt_tensor = []  # 里面的每一个元素表示对应committee member的安全共享
    new_w = copy.deepcopy(w)  # 里面的每一个元素表示对应committee member的安全共享
    all_noise_shares = []

    for part in range(total_parts):
        local_data = w[part]  # 表示第part个用户的参数信息---
        encrypt_tensor = []  # 里面的每一个元素表示对应committee member的安全共享
        local_data: dict  # 注意：如果为了提高精度，可以选择只保留其中几位小数
        tmp_noise_shares = -1 + 2 * np.random.random(size=(total_parts - 2))
        tmp_noise_shares = np.around(tmp_noise_shares, 3)
        tmp_final_noise = 0 - (np.sum(tmp_noise_shares) + noise_list[part])
        tmp_noise_shares: np.array
        tmp_noise_shares = np.append(tmp_noise_shares, tmp_final_noise)  # 这个相当于基础随机数的安全共享内容
        all_noise_shares.append(tmp_noise_shares)
        # encrypt_local_data = []  # 里面的每一个元素表示对应committee member的安全共享
        # 生成安全索引和对应的噪声值

        for com_part in range(total_parts - 1):
            tmp_encrypt_idxs = np.random.choice(a=np.arange(para_num), size=sel_num, replace=False)
            tmp_encrypt_idxs = np.sort(tmp_encrypt_idxs)

            tmp_encrypt_value = np.random.random(size=sel_num)
            tmp_encrypt_value = np.round(tmp_encrypt_value, 3)
            encrypt_tensor.append([tmp_encrypt_idxs, tmp_encrypt_value])
        # 保存对应的索引以及噪声值
        indicator_list = [0] * (total_parts - 1)
        for k in local_data.keys():
            tmp_tensor = copy.deepcopy(local_data[k])
            tmp_tensor = torch.add(tmp_tensor, noise_list[part])  # 添加噪声---这是基础噪声
            # comm_parts_encrypts = generate_arithmetic(local_data=tmp_tensor, word_size=comm_parts)
            tmp_tensor: torch.Tensor
            tmp_tensor_num = tmp_tensor.nelement()
            tmp_tensor_fla = torch.flatten(tmp_tensor)
            for com_part in range(total_parts - 1):  # 因为只需要安全共享给其它total_parts-1个clients
                # 根据a选择需要加密的参数
                indicator_list[
                    com_part] += tmp_tensor_num  # https://blog.csdn.net/lollows/article/details/115916294 必须使用括号
                sel_idx = (encrypt_tensor[com_part][0] < indicator_list[com_part]) & (
                            encrypt_tensor[com_part][0] > indicator_list[com_part] - tmp_tensor_num)
                encrypt_idxs = encrypt_tensor[com_part][0][sel_idx]
                # 记录要加密的key和平整化后的索引值，以及加密后的数值--就是随机数
                # encrypt_idxs = np.random.choice(low=0, high=tmp_tensor_fla.shape[-1], size=sel_num)   # 平整化后的索引值---注意，这样做会有重复索引被选择的哦
                encrypt_value = encrypt_tensor[com_part][1][sel_idx]
                # 原始数据减去对应的随机数
                for ind, ind_noise in zip(encrypt_idxs, encrypt_value):
                    ind = ind - indicator_list[com_part] + tmp_tensor_num
                    tmp_tensor_fla[ind] = tmp_tensor_fla[ind] - ind_noise
            tmp_new_tensor = torch.reshape(tmp_tensor_fla, shape=tmp_tensor.shape)
            new_w[part][k] = copy.deepcopy(tmp_new_tensor)  # 对应位置的数据改为加密后的参数---代表自己本地的数据，所以一共有total_parts个数据

            # # 验证代码---start--暂时未完成
            # tmp_tensor2 = encrypt_tensor[0][k]
            # for com_part in range(1, comm_parts):
            #     tmp_tensor2 += encrypt_tensor[com_part][k]
            #
            # # de_tmp_tensor = tmp_tensor2.get_plain_text()  # 必须的是cryptensor类型的才可以直接使用get_plain_text
            # de_tmp_tensor = encoder.decode(tensor=tmp_tensor2)
            # # 输出总的误差值
            # error_mpc = torch.sum(torch.abs(tmp_tensor - de_tmp_tensor))
            # print('error_mpc:')
            # print(error_mpc)
            # # 验证代码---end
        all_encrypt_tensor.append(encrypt_tensor)  # 里面一共有total_parts个元素，每个元素代表本地数据针对所有comm_parts的加密数据
    # committee 接收安全共享，使用数组---比较难以理解诶
    receive_encrypt_tensor = []  # 将对应的随机数发送给其他人
    receive_encrypt_noise = []  # 将基础随机数发送给其他人
    indicator_index = [0] * total_parts
    for part in range(total_parts):  # 每一个part都会发送给其它人发来的共享（total_parts-1）
        tmp_encrypt_comm = []
        receive_tmp_noises = []
        for com_part in range(total_parts):  # 因为总共有total_parts-1发来安全共享
            if part == com_part:  ## 说明要越过当前client，因为不能自己给自己发数据嘛，以后的索引值要减去1，
                continue  # 不能接受自己本身的数据嘛
            else:
                tmp_encrypt_comm.append(  # com_part是有total parts个的，而发来的共享是有 (total_parts-1)的
                    all_encrypt_tensor[com_part][
                        indicator_index[com_part]])  # all_encrypt_tensor[part][com_part]代表成员part的数据针对所有comm_part的加密数据

                receive_tmp_noises.append(all_noise_shares[com_part][indicator_index[com_part]])
                indicator_index[com_part] += 1
        receive_encrypt_tensor.append(tmp_encrypt_comm)  # total_parts * (total_parts-1)
        receive_encrypt_noise.append(receive_tmp_noises)

    # 接收端对噪声求和
    sum_list = np.sum(np.array(receive_encrypt_noise), axis=1)
    # committee 对接收到的安全共享求和---直接相加会溢出吗---因为是小于1的随机数，是不会溢出的---可以使用增量式的加法
    encrypt_new_w = copy.deepcopy(new_w)
    for com_part in range(total_parts):
        received_data = receive_encrypt_tensor[com_part]  # 接收到其他人发来的加密数据---一组随机数
        new_local_data = encrypt_new_w[com_part]  # 使用接收到的噪声对当前数据集进行加密
        tmp_sum_noise = sum_list[com_part]
        indicator_list = [0] * (total_parts - 1)
        for k in new_local_data.keys():
            tmp_tensor = new_local_data[k]
            tmp_tensor_fla = torch.flatten(tmp_tensor)
            tmp_tensor_num = tmp_tensor.nelement()
            tmp_tensor_num: int
            for part in range(0, total_parts - 1):
                indicator_list[part] += tmp_tensor_num  # 注意不能使用and哦，两个都是list的话可以使用and，如果是numpy，必须使用&
                sel_idx = (received_data[part][0] < indicator_list[part]) & (
                            received_data[part][0] > indicator_list[part] - tmp_tensor_num)
                # sel_idx = [part_encrypt_data[com_part] < indicator_list[com_part]]
                tmp_encrypt_idxs = received_data[part][0][sel_idx]
                tmp_encrypt_value = received_data[part][1][sel_idx]
                for tmp_idx, ind_noise in zip(tmp_encrypt_idxs, tmp_encrypt_value):
                    tmp_idx = tmp_idx - indicator_list[part] + tmp_tensor_num
                    tmp_tensor_fla[tmp_idx] = tmp_tensor_fla[tmp_idx] + ind_noise
            tmp_new_tensor = torch.reshape(tmp_tensor_fla, shape=tmp_tensor.shape)
            tmp_new_tensor = torch.add(tmp_new_tensor, tmp_sum_noise)  # 聚合噪声值
            encrypt_new_w[com_part][k] = copy.deepcopy(tmp_new_tensor)  # 对应位置的数据改为加密后的参数
    # 发送给服务器 得到 encrypt_sum ---计算总的数据流--w_avg即为所求的模型聚合值
    w_avg = copy.deepcopy(encrypt_new_w[0])
    for k in w_avg.keys():
        for com_part in range(1, total_parts):
            w_avg[k] += encrypt_new_w[com_part][k]
        w_avg[k] = w_avg[k] / total_parts
    time_end = time.time()
    # 总的消耗
    total_mem_con = get_memory_hfl_fla(all_encrypt_tensor, all_noise_shares, encrypt_new_w)
    mem_consumer['total_mem_cons'] = total_mem_con
    mem_consumer['total_time_cons'] = time_end - time_start
    return w_avg, mem_consumer


def FedHFL_fla_opt(w_locals, a=0.02):  # a指的是加密参数的比例
    ## 工作流程
    # 对所有参数添加噪声（后期会消掉的）
    # 生成len(w)个安全共享，并发送给其它成员--计算总的数据流
    # 接收安全共享，使用数组
    # 对接收到的安全共享求和
    # 发送给其它committee member---计算总的数据流
    # 对所有的共享数据进行求和，然后再求平均，进而得到最终的参数
    # Note: 展开所有的参数值，然后选择对应比例的索引
    # 这部分内容没必要总是计算的
    # 代码冗余的地方，就是在本地执行了两次更改权重值的操作，使得执行时间大幅增加 -------- 注意
    para_num = sum([w_locals[0][k].nelement() for k in w_locals[0].keys()])
    sel_num = max(int(para_num * a), 1)  # 最少选择一个参数进行加密
    time_start = time.time()
    mem_consumer = {}
    total_parts = len(w_locals)  # 所有参与者的数量
    total_num = sel_num * (total_parts - 1)
    assert a < 1, print('the setting for a is not appropriate')
    # 先加噪声，再根据a 选择参数
    noise_list = np.random.random(total_parts)
    noise_list = np.around(noise_list, 3)
    # rank = crypten.communicator.get().get_rank()  # 本机的rank
    all_encrypt_tensor = []  # 里面的每一个元素表示对应committee member的安全共享
    # new_w = copy.deepcopy(w)  # 里面的每一个元素表示对应committee member的安全共享
    all_noise_shares = []
    for part in range(total_parts):
        # local_data = w[part]  # 表示第part个用户的参数信息---
        encrypt_tensor = []  # 里面的每一个元素表示对应committee member的安全共享
        local_data: dict  # 注意：如果为了提高精度，可以选择只保留其中几位小数
        tmp_noise_shares = -1 + 2 * np.random.random(size=(total_parts - 2))
        tmp_noise_shares = np.around(tmp_noise_shares, 3)
        tmp_final_noise = 0 - (np.sum(tmp_noise_shares) + noise_list[part])
        tmp_noise_shares: np.array
        tmp_noise_shares = np.append(tmp_noise_shares, tmp_final_noise)  # 这个相当于基础随机数的安全共享内容
        all_noise_shares.append(tmp_noise_shares)
        # encrypt_local_data = []  # 里面的每一个元素表示对应committee member的安全共享
        # 生成安全索引和对应的噪声值
        tmp_encrypt_idxs_all = np.random.choice(a=np.arange(para_num), size=total_num, replace=False)
        tmp_encrypt_value_all = np.random.random(size=total_num)
        tmp_encrypt_value_all = np.round(tmp_encrypt_value_all, 3)
        for com_part in range(total_parts - 1):
            tmp_encrypt_idxs = tmp_encrypt_idxs_all[com_part * sel_num: com_part * (sel_num + 1)]
            # tmp_encrypt_idxs = np.sort(tmp_encrypt_idxs)

            tmp_encrypt_value = tmp_encrypt_value_all[com_part * sel_num: com_part * (sel_num + 1)]
            # tmp_encrypt_value = np.round(tmp_encrypt_value, 3)
            encrypt_tensor.append([tmp_encrypt_idxs, tmp_encrypt_value])
        all_encrypt_tensor.append(encrypt_tensor)  # 里面一共有total_parts个元素，每个元素代表本地数据针对所有comm_parts的加密数据
    # committee 接收安全共享，使用数组---比较难以理解诶
    receive_encrypt_tensor = []  # 将对应的随机数发送给其他人
    receive_encrypt_noise = []  # 将基础随机数发送给其他人
    indicator_index = [0] * total_parts
    for part in range(total_parts):  # 每一个part都会发送给其它人发来的共享（total_parts-1）
        tmp_encrypt_comm = []
        receive_tmp_noises = []
        for com_part in range(total_parts):  # 因为总共有total_parts-1发来安全共享
            if part == com_part:  ## 说明要越过当前client，因为不能自己给自己发数据嘛，以后的索引值要减去1，
                continue  # 不能接受自己本身的数据嘛
            else:
                tmp_encrypt_comm.append(  # com_part是有total parts个的，而发来的共享是有 (total_parts-1)的
                    all_encrypt_tensor[com_part][
                        indicator_index[com_part]])  # all_encrypt_tensor[part][com_part]代表成员part的数据针对所有comm_part的加密数据

                receive_tmp_noises.append(all_noise_shares[com_part][indicator_index[com_part]])
                indicator_index[com_part] += 1
        receive_encrypt_tensor.append(tmp_encrypt_comm)  # total_parts * (total_parts-1)
        receive_encrypt_noise.append(receive_tmp_noises)

    # 接收端对噪声求和
    sum_list = np.sum(np.array(receive_encrypt_noise), axis=1)
    # committee 对接收到的安全共享求和---直接相加会溢出吗---因为是小于1的随机数，是不会溢出的---可以使用增量式的加法
    encrypt_new_w = copy.deepcopy(w_locals)
    for com_part in range(total_parts):
        received_data = receive_encrypt_tensor[com_part]  # 接收到其他人发来的加密数据---一组随机数
        new_local_data = encrypt_new_w[com_part]  # 使用接收到的噪声对当前数据集进行加密
        tmp_sum_noise = sum_list[com_part] + noise_list[com_part]  # 基础噪声
        indicator_list = [0] * (total_parts - 1)
        encrypt_tensor = all_encrypt_tensor[com_part]

        for k in new_local_data.keys():
            tmp_tensor = new_local_data[k]
            tmp_tensor_fla = torch.flatten(tmp_tensor)
            tmp_tensor_num = tmp_tensor.nelement()
            tmp_tensor_num: int
            for part in range(0, total_parts - 1):
                indicator_list[part] += tmp_tensor_num  # 注意不能使用and哦，两个都是list的话可以使用and，如果是numpy，必须使用&
                sel_idx = (received_data[part][0] < indicator_list[part]) & (
                            received_data[part][0] > indicator_list[part] - tmp_tensor_num)
                # sel_idx = [part_encrypt_data[com_part] < indicator_list[com_part]]
                tmp_encrypt_idxs = received_data[part][0][sel_idx]
                tmp_encrypt_value = received_data[part][1][sel_idx]
                for tmp_idx, ind_noise in zip(tmp_encrypt_idxs, tmp_encrypt_value):
                    tmp_idx = tmp_idx - indicator_list[part] + tmp_tensor_num
                    tmp_tensor_fla[tmp_idx] = tmp_tensor_fla[tmp_idx] + ind_noise
                sel_idx = (encrypt_tensor[part][0] < indicator_list[part]) & (
                            encrypt_tensor[part][0] > indicator_list[part] - tmp_tensor_num)
                tmp_encrypt_idxs = encrypt_tensor[part][0][sel_idx]
                tmp_encrypt_value = encrypt_tensor[part][1][sel_idx]
                for tmp_idx, ind_noise in zip(tmp_encrypt_idxs, tmp_encrypt_value):
                    tmp_idx = tmp_idx - indicator_list[part] + tmp_tensor_num
                    tmp_tensor_fla[tmp_idx] = tmp_tensor_fla[tmp_idx] - ind_noise  # 注意是减去的哦

            tmp_new_tensor = torch.reshape(tmp_tensor_fla, shape=tmp_tensor.shape)
            tmp_new_tensor = torch.add(tmp_new_tensor, tmp_sum_noise)  # 聚合噪声值
            encrypt_new_w[com_part][k] = copy.deepcopy(tmp_new_tensor)  # 对应位置的数据改为加密后的参数
    # 发送给服务器 得到 encrypt_sum ---计算总的数据流--w_avg即为所求的模型聚合值
    w_avg = copy.deepcopy(encrypt_new_w[0])
    for k in w_avg.keys():
        for com_part in range(1, total_parts):
            w_avg[k] += encrypt_new_w[com_part][k]
        w_avg[k] = w_avg[k] / total_parts
    time_end = time.time()
    # 总的消耗
    total_mem_con = get_memory_hfl_fla(all_encrypt_tensor, all_noise_shares, encrypt_new_w)
    mem_consumer['total_mem_cons'] = total_mem_con
    mem_consumer['total_time_cons'] = time_end - time_start
    return w_avg, mem_consumer


def FedHFL_fla_opt_for_privacy(w_locals, a=0.02):  # a指的是加密参数的比例
    ## 工作流程
    # 对所有参数添加噪声（后期会消掉的）
    # 生成len(w)个安全共享，并发送给其它成员--计算总的数据流
    # 接收安全共享，使用数组
    # 对接收到的安全共享求和
    # 发送给其它committee member---计算总的数据流
    # 对所有的共享数据进行求和，然后再求平均，进而得到最终的参数
    # Note: 展开所有的参数值，然后选择对应比例的索引
    # 这部分内容没必要总是计算的
    # 代码冗余的地方，就是在本地执行了两次更改权重值的操作，使得执行时间大幅增加--------注意
    para_num = sum([w_locals[0][k].nelement() for k in w_locals[0].keys()])
    sel_num = max(int(para_num * a), 1)  # 最少选择一个参数进行加密
    time_start = time.time()
    mem_consumer = {}
    total_parts = len(w_locals)  # 所有参与者的数量
    assert a < 1, print('the setting for a is not appropriate')
    # 先加噪声，再根据a 选择参数
    noise_list = np.random.random(total_parts)
    noise_list = np.around(noise_list, 3)
    # rank = crypten.communicator.get().get_rank()  # 本机的rank
    all_encrypt_tensor = []  # 里面的每一个元素表示对应committee member的安全共享
    # new_w = copy.deepcopy(w)  # 里面的每一个元素表示对应committee member的安全共享
    all_noise_shares = []

    for part in range(total_parts):
        # local_data = w[part]  # 表示第part个用户的参数信息---
        encrypt_tensor = []  # 里面的每一个元素表示对应committee member的安全共享
        local_data: dict  # 注意：如果为了提高精度，可以选择只保留其中几位小数
        tmp_noise_shares = -1 + 2 * np.random.random(size=(total_parts - 2))
        tmp_noise_shares = np.around(tmp_noise_shares, 3)
        tmp_final_noise = 0 - (np.sum(tmp_noise_shares) + noise_list[part])
        tmp_noise_shares: np.array
        tmp_noise_shares = np.append(tmp_noise_shares, tmp_final_noise)  # 这个相当于基础随机数的安全共享内容
        all_noise_shares.append(tmp_noise_shares)
        # encrypt_local_data = []  # 里面的每一个元素表示对应committee member的安全共享
        # 生成安全索引和对应的噪声值

        for com_part in range(total_parts - 1):
            tmp_encrypt_idxs = np.random.choice(a=np.arange(para_num), size=sel_num, replace=False)
            tmp_encrypt_idxs = np.sort(tmp_encrypt_idxs)

            tmp_encrypt_value = np.random.random(size=sel_num)
            tmp_encrypt_value = np.round(tmp_encrypt_value, 3)
            encrypt_tensor.append([tmp_encrypt_idxs, tmp_encrypt_value])
        all_encrypt_tensor.append(encrypt_tensor)  # 里面一共有total_parts个元素，每个元素代表本地数据针对所有comm_parts的加密数据
    # committee 接收安全共享，使用数组---比较难以理解诶
    receive_encrypt_tensor = []  # 将对应的随机数发送给其他人
    receive_encrypt_noise = []  # 将基础随机数发送给其他人
    indicator_index = [0] * total_parts
    for part in range(total_parts):  # 每一个part都会发送给其它人发来的共享（total_parts-1）
        tmp_encrypt_comm = []
        receive_tmp_noises = []
        for com_part in range(total_parts):  # 因为总共有total_parts-1发来安全共享
            if part == com_part:  ## 说明要越过当前client，因为不能自己给自己发数据嘛，以后的索引值要减去1，
                continue  # 不能接受自己本身的数据嘛
            else:
                tmp_encrypt_comm.append(  # com_part是有total parts个的，而发来的共享是有 (total_parts-1)的
                    all_encrypt_tensor[com_part][
                        indicator_index[com_part]])  # all_encrypt_tensor[part][com_part]代表成员part的数据针对所有comm_part的加密数据

                receive_tmp_noises.append(all_noise_shares[com_part][indicator_index[com_part]])
                indicator_index[com_part] += 1
        receive_encrypt_tensor.append(tmp_encrypt_comm)  # total_parts * (total_parts-1)
        receive_encrypt_noise.append(receive_tmp_noises)

    # 接收端对噪声求和
    sum_list = np.sum(np.array(receive_encrypt_noise), axis=1)
    # committee 对接收到的安全共享求和---直接相加会溢出吗---因为是小于1的随机数，是不会溢出的---可以使用增量式的加法
    encrypt_new_w = copy.deepcopy(w_locals)
    for com_part in range(total_parts):
        received_data = receive_encrypt_tensor[com_part]  # 接收到其他人发来的加密数据---一组随机数
        new_local_data = encrypt_new_w[com_part]  # 使用接收到的噪声对当前数据集进行加密
        tmp_sum_noise = sum_list[com_part] + noise_list[com_part]  # 基础噪声
        indicator_list = [0] * (total_parts - 1)
        encrypt_tensor = all_encrypt_tensor[com_part]

        for k in new_local_data.keys():
            tmp_tensor = new_local_data[k]
            tmp_tensor_fla = torch.flatten(tmp_tensor)
            tmp_tensor_num = tmp_tensor.nelement()
            tmp_tensor_num: int
            for part in range(0, total_parts - 1):
                indicator_list[part] += tmp_tensor_num  # 注意不能使用and哦，两个都是list的话可以使用and，如果是numpy，必须使用&
                sel_idx = (received_data[part][0] < indicator_list[part]) & (
                            received_data[part][0] > indicator_list[part] - tmp_tensor_num)
                # sel_idx = [part_encrypt_data[com_part] < indicator_list[com_part]]
                tmp_encrypt_idxs = received_data[part][0][sel_idx]
                tmp_encrypt_value = received_data[part][1][sel_idx]
                for tmp_idx, ind_noise in zip(tmp_encrypt_idxs, tmp_encrypt_value):
                    tmp_idx = tmp_idx - indicator_list[part] + tmp_tensor_num
                    tmp_tensor_fla[tmp_idx] = tmp_tensor_fla[tmp_idx] + ind_noise
                sel_idx = (encrypt_tensor[part][0] < indicator_list[part]) & (
                            encrypt_tensor[part][0] > indicator_list[part] - tmp_tensor_num)
                tmp_encrypt_idxs = encrypt_tensor[part][0][sel_idx]
                tmp_encrypt_value = encrypt_tensor[part][1][sel_idx]
                for tmp_idx, ind_noise in zip(tmp_encrypt_idxs, tmp_encrypt_value):
                    tmp_idx = tmp_idx - indicator_list[part] + tmp_tensor_num
                    tmp_tensor_fla[tmp_idx] = tmp_tensor_fla[tmp_idx] - ind_noise  # 注意是减去的哦

            tmp_new_tensor = torch.reshape(tmp_tensor_fla, shape=tmp_tensor.shape)
            tmp_new_tensor = torch.add(tmp_new_tensor, tmp_sum_noise)  # 聚合噪声值
            encrypt_new_w[com_part][k] = copy.deepcopy(tmp_new_tensor)  # 对应位置的数据改为加密后的参数
    # 发送给服务器 得到 encrypt_sum ---计算总的数据流--w_avg即为所求的模型聚合值---只需要加密后的参数信息即可
    # w_avg = copy.deepcopy(encrypt_new_w[0])
    # for k in w_avg.keys():
    #     for com_part in range(1, total_parts):
    #         w_avg[k] += encrypt_new_w[com_part][k]
    #     w_avg[k] = w_avg[k]/total_parts
    # time_end = time.time()
    # # 总的消耗
    # total_mem_con = get_memory_hfl_fla(all_encrypt_tensor, all_noise_shares, encrypt_new_w)
    # mem_consumer['total_mem_cons'] = total_mem_con
    # mem_consumer['total_time_cons'] = time_end - time_start
    return encrypt_new_w
# 先给出一个随着客户端数量增加---massage memory 使用对比--two phrase和VHFL（包括简单和复杂网络结构）
# 再给出一个随着客户端数量增加---massage size 使用对比--- 比不过的应该就是建立通信的次数，一个是n**2，另外一个是n*m
# 再给出一个随着客户端数量增加---性能 对比 与差分隐私还有原始的fedavg--- （同下一个）先做这一个，需要证明自己所加入的隐私是有效的（对性能的要高一些），而不是随便加的
# 再给出一个安全性对比，就是使用DLG对暴露的数据进行攻击，然后查看其隐私性能----先测试自己所提方法的性能与隐私方面的测试（与加入高斯隐私等相对比，而不是那些复杂的隐私方法）
#

# https://github.com/Xtra-Computing/PrivML   联邦学习-梯度提升树
