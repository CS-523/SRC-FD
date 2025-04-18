#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Traceless'
"""
import copy
import numpy as np
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from utils.focal_loss2 import focal_loss2
import time
# from Fed import get_avg_error
from utils.Fed import get_avg_error

class DatasetSplit(Dataset):
    def __init__(self, x, label, idxs):
        self.dataset = x
        self.label = label
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]], self.label[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, x=None, label=None, idxs=None,  focal_mode=True):
        self.args = args

        # self.loss_func = focal_loss2(alpha=0.16, gamma=3)  # 需要自己来根据数据集设置的超参数
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(x=x, label=label, idxs=idxs), batch_size=self.args.local_bs, shuffle=True)
        if focal_mode:
            # criterion = focal_loss2(alpha=alpha_list[run_num], gamma=2)
            # criterion = focal_loss2(alpha=0.16, gamma=int(gamma_list[run_num]))
            self.loss_func = focal_loss2(alpha=0.16, gamma=3)
            # criterion = focal_loss2(alpha=0.012)
            loss_name = 'focal_loss'
        else:
            self.loss_func = nn.CrossEntropyLoss()
            loss_name = 'cross_loss'
        self.tmp_error_sum = 0
        self.data_round = len(self.ldr_train) *self.args.local_ep

    def train(self, net):
        net.train()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        # torch.nn.functional.cross_entropy()
        optimizer = torch.optim.Adam(lr=self.args.lr, params=net.parameters())

        epoch_loss = []
        net_back = copy.deepcopy(net)
        self.tmp_error_sum = 1e-6
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.ldr_train):
                x, labels = x.to(self.args.device).float(), labels.to(self.args.device).float()
                optimizer.zero_grad()
                log_probs = net(x)
                loss = self.loss_func(log_probs, labels.long())
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(x), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                # 计算本轮更新的error
                # tmp_error = 0
                tmp_error = get_avg_error(global_true=net_back.state_dict(), global_fed=net.state_dict())
                # if tmp_error <1e-6:
                #     print('tmp_error: {}'.format(tmp_error))
                #     for name, parms in net.named_parameters():
                #         print('-->name:', name)
                #         print('-->para:', parms)
                #         print('-->grad_requirs:', parms.requires_grad)
                #         print('-->grad_value:', parms.grad)
                #         print("===")
                net_back = copy.deepcopy(net)
                # print('tmp_error: {}'.format(tmp_error))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    
class LocalUpdate_sentry(object):
    def __init__(self, args, x=None, label=None, idxs=None, budget=None, focal_mode=True, max_error=0,
                 ablation_sen=True, ablation_budget=True, bias_rate =0.5):
        self.args = args

        # self.loss_func = focal_loss2(alpha=0.16, gamma=3)  # 需要自己来根据数据集设置的超参数
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(x=x, label=label, idxs=idxs), batch_size=self.args.local_bs, shuffle=True)
        self.max_error = max_error
        self.budget = budget
        self.sentivity = None
        self.weight_rate = (1-bias_rate)
        if focal_mode:
            # criterion = focal_loss2(alpha=alpha_list[run_num], gamma=2)
            # criterion = focal_loss2(alpha=0.16, gamma=int(gamma_list[run_num]))
            self.loss_func = focal_loss2(alpha=0.16, gamma=3)
            # criterion = focal_loss2(alpha=0.012)
            loss_name = 'focal_loss'
        else:
            self.loss_func = nn.CrossEntropyLoss()
            loss_name = 'cross_loss'
        self.tmp_error_sum = 0
        self.data_round = len(self.ldr_train) *self.args.local_ep
        self.ablation_sen = ablation_sen
        self.ablation_budget = ablation_budget


    def add_noise_to_gradients(self, model, budget):
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
                    sentivity1 = torch.abs(bias_grad_dict[bias_key])
                    # sentivity = torch.maximum(sentivity1 , sentivity2* self.weight_rate)
                    sentivity = sentivity1.repeat(p.shape[1], 1).T
                    if torch.mean(sentivity) == 0:
                        print('bias_grad_dict[bias_key]:{}'.format(bias_grad_dict[bias_key]))
                        # print(new_budget, num_bias)
                    # noise = torch.randn_like(p.grad) * noise_scale
                    noise_scale1 = (sentivity / self.budget).cpu().numpy() + 1e-6
                    sentivity2 = torch.max(p.grad) - torch.min(p.grad)
                    noise_scale2 = (sentivity2 / budget).cpu().numpy() + 1e-6
                    noise_scale = np.maximum(noise_scale1, noise_scale2*self.weight_rate)
                    noise = np.random.laplace(0, noise_scale, p.grad.size())
                    noise = torch.from_numpy(noise).float().to(p.grad.device)
                else:
                    sentivity = torch.max(p.grad)-torch.min(p.grad)
                    noise_scale = (sentivity / budget).cpu().numpy() + 1e-6
                    noise = np.random.laplace(0, noise_scale, p.grad.size())
                    noise = torch.from_numpy(noise).float().to(p.grad.device)
                # 添加噪声值
                p.grad.add_(noise)

    def add_noise_to_gradients_sen(self, model, budget):
        # model_para = model.state_dict()
        # for name in model_para.keys():
        bias_grad_dict = {}
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_grad_dict[name] = p.grad  # 存储原始信息
        for name, p in model.named_parameters():
            # p = model_para[name]
            if p.grad is not None:
                sentivity = torch.max(p.grad)-torch.min(p.grad)
                noise_scale = (sentivity / budget).cpu().numpy() + 1e-6
                noise = np.random.laplace(0, noise_scale, p.grad.size())
                noise = torch.from_numpy(noise).float().to(p.grad.device)
                # 添加噪声值
                p.grad.add_(noise)

    def add_noise_to_gradients_budget(self, model, budget):
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
                        print('bias_grad_dict[bias_key]:{}'.format(bias_grad_dict[bias_key]))
                        # print(new_budget, num_bias)
                    # noise = torch.randn_like(p.grad) * noise_scale
                    noise_scale = (sentivity / self.budget).cpu().numpy() + 1e-6
                    noise = np.random.laplace(0, noise_scale, p.grad.size())
                    noise = torch.from_numpy(noise).float().to(p.grad.device)
                else:
                    sentivity = torch.max(p.grad) - torch.min(p.grad)
                    noise_scale = (sentivity / self.budget).cpu().numpy() + 1e-6
                    noise = np.random.laplace(0, noise_scale, p.grad.size())
                    noise = torch.from_numpy(noise).float().to(p.grad.device)
                # 添加噪声值
                p.grad.add_(noise)

    def add_noise_to_gradients_dp(self, model):
        # model_para = model.state_dict()
        # for name in model_para.keys():
        # bias_grad_dict = {}
        # for name, p in model.named_parameters():
        #     if 'bias' in name:
        #         bias_grad_dict[name] = p.grad  # 存储原始信息

        for name, p in model.named_parameters():
            # p = model_para[name]
            if p.grad is not None:

                sentivity = torch.max(p.grad) - torch.min(p.grad)
                noise_scale = (sentivity / self.budget).cpu().numpy() + 1e-6
                noise = np.random.laplace(0, noise_scale, p.grad.size())
                noise = torch.from_numpy(noise).float().to(p.grad.device)
                # 添加噪声值
                p.grad.add_(noise)

    def train(self, net, budget):
        net.train()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        # torch.nn.functional.cross_entropy()
        optimizer = torch.optim.Adam(lr=self.args.lr, params=net.parameters())

        epoch_loss = []
        net_back = copy.deepcopy(net)
        self.tmp_error_sum = 1e-6
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.ldr_train):
                x, labels = x.to(self.args.device).float(), labels.to(self.args.device).float()
                optimizer.zero_grad()
                log_probs = net(x)
                loss = self.loss_func(log_probs, labels.long())
                loss.backward()
                if self.ablation_sen and not self.ablation_budget:
                    self.add_noise_to_gradients_sen(net, budget=budget)
                elif not self.ablation_sen and self.ablation_budget:
                    self.add_noise_to_gradients_budget(net, budget=budget)
                elif self.ablation_sen and self.ablation_budget:
                    self.add_noise_to_gradients(net, budget=budget)
                else:
                    self.add_noise_to_gradients_dp(net)
                # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(x), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                # for name, parms in net.named_parameters():
                #     print('-->name:', name)
                #     print('-->para:', parms)
                #     print('-->grad_requirs:', parms.requires_grad)
                #     print('-->grad_value:', parms.grad)
                #     print("===")
                # 计算本轮更新的error
                # tmp_error = 0
                tmp_error = get_avg_error(global_true=net_back.state_dict(), global_fed=net.state_dict())
                if tmp_error <1e-6:
                    print('tmp_error: {}'.format(tmp_error))
                    for name, parms in net.named_parameters():
                        print('-->name:', name)
                        print('-->para:', parms)
                        print('-->grad_requirs:', parms.requires_grad)
                        print('-->grad_value:', parms.grad)
                        print("===")


                self.tmp_error_sum += tmp_error
                if tmp_error > self.max_error:
                    self.max_error = tmp_error
                # if self.sentivity == None:
                #     self.sentivity = torch.max(gradients_value.float()) - torch.min(gradients_value.float())

                net_back = copy.deepcopy(net)
                # print('tmp_error: {}'.format(tmp_error))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def calc_budget(self):

        num_batchs = len(self.ldr_train)
        assert self.tmp_error_sum > 0, print('error {}'.format(self.tmp_error_sum))
        budget = self.max_error*num_batchs*self.args.local_ep/self.tmp_error_sum * self.budget
        assert budget > 0, print('budget:{}, self.max_error:{}, num_batchs:{}, self.args.local_ep:{}, self.tmp_error_sum:{}, '
                                 'self.budget:{}'.format(budget, self.max_error, num_batchs, self.args.local_ep, self.tmp_error_sum, self.budget))

        return budget

def train_privacy(net, gt_data, gt_label, loss_print=False):
    loss_func = nn.CrossEntropyLoss()
    net.train()
    # train and update
    # optimizer = torch.optim.Adam(lr=self.args.lr, params=net.parameters())
    # compute original gradient
    # gt_data, gt_label = self.ldr_train[0]
    out = net(gt_data)
    y = loss_func(out, gt_label)  # 好像不用变成one hot 编码也可以正常计算的
    if loss_print:
        print('loss: {}'.format(y))
    dy_dx = torch.autograd.grad(y, net.parameters())
    # original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    original_dy_dx = {}
    cnt = 0
    for i in dy_dx:
        cnt += 1
        original_dy_dx['{}'.format(cnt)] = i
    return original_dy_dx

def train_privacy_lossfocal(net, loss_func, gt_data, gt_label, loss_print=False):
    # loss_func = nn.CrossEntropyLoss()
    net.train()
    # train and update
    # optimizer = torch.optim.Adam(lr=self.args.lr, params=net.parameters())
    # compute original gradient
    # gt_data, gt_label = self.ldr_train[0]
    out = net(gt_data)
    y = loss_func(out, gt_label)  # 好像不用变成one hot 编码也可以正常计算的
    if loss_print:
        print('loss: {}'.format(y))
    dy_dx = torch.autograd.grad(y, net.parameters())
    # original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    original_dy_dx = {}
    cnt = 0
    for i in dy_dx:
        cnt += 1
        original_dy_dx['{}'.format(cnt)] = i
    return original_dy_dx

def dlg_attack(net, gt_data, gt_label, method, original_grident, device, Iteration = 30000):
    criterion = nn.CrossEntropyLoss()
    dummy_data = torch.rand(gt_data.size()).to(device).requires_grad_(True)
    # dummy_label = torch.randn(gt_label.size()).to(device).requires_grad_(True)
    dummy_label = gt_label
    if type(original_grident) == dict:
        original_dy_dx = []
        for k in original_grident.keys():
            original_dy_dx.append(original_grident[k])
    else:
        original_dy_dx = original_grident

    if method == 'DLG':
        # optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
        lr=0.1
        # optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
        optimizer = torch.optim.Adam([dummy_data], lr=5e-2)
    elif method == 'iDLG':
        # optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
        optimizer = torch.optim.Adam([dummy_data, ], lr=5e-2)  # 不好用的哦
        # predict the ground-truth label,而不是像DLG那样进行梯度下降更新的
        label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
            (1,)).requires_grad_(False)

    history_iters = []
    losses = []
    mses = []
    train_iters = []
    for iters in range(Iteration):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            if method == 'DLG':
                # dummy_loss = - torch.mean(
                #     torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                dummy_loss = criterion(pred, dummy_label)  # 为什么要这样改变呢？
            elif method == 'iDLG':
                dummy_loss = criterion(pred, label_pred)
            else:
                dummy_loss = 0
                raise NotImplementedError
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

        # 下面是绘图的程序
        if iters % int(Iteration / 30) == 0:  # 每隔Iteration / 30输出一份调试信息，保证最后的图片一共有30张
            # if iters == 0:  # 每隔Iteration / 30输出一份调试信息，保证最后的图片一共有30张
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
            history_iters.append(iters)
            if current_loss < 0.000001:  # converge
                break
    # tmp_mse_list[method].append(mses[-1])
    return mses[-1]

def dlg_attack_focal(net, criterion, gt_data, gt_label, method, original_grident, device, Iteration = 10000):
    # criterion = nn.CrossEntropyLoss()
    # loss_func = nn.CrossEntropyLoss()

    dummy_data = torch.rand(gt_data.size()).to(device).requires_grad_(True)
    # dummy_label = torch.randn(gt_label.size()).to(device).requires_grad_(True)
    dummy_label = gt_label
    if type(original_grident) == dict:
        original_dy_dx = []
        for k in original_grident.keys():
            original_dy_dx.append(original_grident[k])
    else:
        original_dy_dx = original_grident

    if method == 'DLG':
        # optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
        lr=0.1
        # optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
        optimizer = torch.optim.Adam([dummy_data], lr=5e-2)
    elif method == 'iDLG':
        # optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
        optimizer = torch.optim.Adam([dummy_data, ], lr=5e-2)  # 不好用的哦
        # predict the ground-truth label,而不是像DLG那样进行梯度下降更新的
        label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
            (1,)).requires_grad_(False)

    history_iters = []
    losses = []
    mses = []
    train_iters = []
    for iters in range(Iteration):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            if method == 'DLG':
                # dummy_loss = - torch.mean(
                #     torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                dummy_loss = criterion(pred, dummy_label)  # 为什么要这样改变呢？
            elif method == 'iDLG':
                dummy_loss = criterion(pred, label_pred)
            else:
                dummy_loss = 0
                raise NotImplementedError
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

        # 下面是绘图的程序
        if iters % int(Iteration / 30) == 0:  # 每隔Iteration / 30输出一份调试信息，保证最后的图片一共有30张
            # if iters == 0:  # 每隔Iteration / 30输出一份调试信息，保证最后的图片一共有30张
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
            history_iters.append(iters)
            if current_loss < 0.000001:  # converge
                break
    # tmp_mse_list[method].append(mses[-1])
    return mses[-1]

def dlg_attack_for_vesta(net, gt_data, gt_label, method, original_grident, device, Iteration = 30000):
    criterion = nn.CrossEntropyLoss()
    dummy_data = torch.zeros(gt_data.size()).to(device).requires_grad_(True)
    # dummy_label = torch.randn(gt_label.size()).to(device).requires_grad_(True)
    dummy_label = gt_label
    if type(original_grident) == dict:
        original_dy_dx = []
        for k in original_grident.keys():
            original_dy_dx.append(original_grident[k])
    else:
        original_dy_dx = original_grident

    if method == 'DLG':
        # optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
        lr=1
        # optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
        optimizer = torch.optim.Adam([dummy_data], lr=5e-2)
    elif method == 'iDLG':
        # optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
        optimizer = torch.optim.Adam([dummy_data, ], lr=5e-2)  # 不好用的哦
        # predict the ground-truth label,而不是像DLG那样进行梯度下降更新的
        label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
            (1,)).requires_grad_(False)

    history_iters = []
    losses = []
    mses = []
    train_iters = []
    for iters in range(Iteration):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            if method == 'DLG':
                # dummy_loss = - torch.mean(
                #     torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                dummy_loss = criterion(pred, dummy_label)  # 为什么要这样改变呢？
            elif method == 'iDLG':
                dummy_loss = criterion(pred, label_pred)
            else:
                dummy_loss = 0
                raise NotImplementedError
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

        # 下面是绘图的程序
        if iters % int(Iteration / 30) == 0:  # 每隔Iteration / 30输出一份调试信息，保证最后的图片一共有30张
            # if iters == 0:  # 每隔Iteration / 30输出一份调试信息，保证最后的图片一共有30张
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
            history_iters.append(iters)
            if current_loss < 0.000001:  # converge
                break
    # tmp_mse_list[method].append(mses[-1])
    return mses[-1]


def dlg_attack_for_alpha(net, gt_data, gt_label, method, original_grident, device, Iteration = 30000):
    # 要比dlg_attack增加一个噪声的未知量
    criterion = nn.CrossEntropyLoss()
    dummy_data = torch.rand(gt_data.size()).to(device).requires_grad_(True)
    # dummy_label = torch.randn(gt_label.size()).to(device).requires_grad_(True)
    dummy_label = gt_label
    if type(original_grident) == dict:
        original_dy_dx = []
        for k in original_grident.keys():
            original_dy_dx.append(original_grident[k])
    else:
        original_dy_dx = original_grident
    dummy_noise = torch.rand(1)
    if method == 'DLG':
        # optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
        # optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
        optimizer = torch.optim.Adam([dummy_data, dummy_noise], lr=5e-2)
    elif method == 'iDLG':
        # optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
        optimizer = torch.optim.Adam([dummy_data, ], lr=5e-2)  # 不好用的哦
        # predict the ground-truth label,而不是像DLG那样进行梯度下降更新的
        label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
            (1,)).requires_grad_(False)

    history_iters = []
    losses = []
    mses = []
    train_iters = []
    for iters in range(Iteration):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            if method == 'DLG':
                # dummy_loss = - torch.mean(
                #     torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                dummy_loss = criterion(pred, dummy_label)  # 为什么要这样改变呢？
            elif method == 'iDLG':
                dummy_loss = criterion(pred, label_pred)
            else:
                dummy_loss = 0
                raise NotImplementedError
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0

            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                gx += dummy_noise[0]
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

        # 下面是绘图的程序
        if iters % int(Iteration / 30) == 0:  # 每隔Iteration / 30输出一份调试信息，保证最后的图片一共有30张
            # if iters == 0:  # 每隔Iteration / 30输出一份调试信息，保证最后的图片一共有30张
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
            history_iters.append(iters)
            if current_loss < 0.000001:  # converge
                break
    # tmp_mse_list[method].append(mses[-1])
    return mses[-1]

def dlg_attack_for_vesta_alpha(net, gt_data, gt_label, method, original_grident, device, Iteration = 30000):
    # 要比dlg_attack增加一个噪声的未知量
    criterion = nn.CrossEntropyLoss()
    dummy_data = torch.zeros(gt_data.size()).to(device).requires_grad_(True)
    # dummy_label = torch.randn(gt_label.size()).to(device).requires_grad_(True)
    dummy_label = gt_label
    if type(original_grident) == dict:
        original_dy_dx = []
        for k in original_grident.keys():
            original_dy_dx.append(original_grident[k])
    else:
        original_dy_dx = original_grident
    dummy_noise = torch.rand(1)
    if method == 'DLG':
        # optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)
        # optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
        optimizer = torch.optim.Adam([dummy_data, dummy_noise], lr=5e-2)
    elif method == 'iDLG':
        # optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
        optimizer = torch.optim.Adam([dummy_data, ], lr=5e-2)  # 不好用的哦
        # predict the ground-truth label,而不是像DLG那样进行梯度下降更新的
        label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
            (1,)).requires_grad_(False)

    history_iters = []
    losses = []
    mses = []
    train_iters = []
    for iters in range(Iteration):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            if method == 'DLG':
                # dummy_loss = - torch.mean(
                #     torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                dummy_loss = criterion(pred, dummy_label)  # 为什么要这样改变呢？
            elif method == 'iDLG':
                dummy_loss = criterion(pred, label_pred)
            else:
                dummy_loss = 0
                raise NotImplementedError
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0

            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                gx += dummy_noise[0]
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

        # 下面是绘图的程序
        if iters % int(Iteration / 30) == 0:  # 每隔Iteration / 30输出一份调试信息，保证最后的图片一共有30张
            # if iters == 0:  # 每隔Iteration / 30输出一份调试信息，保证最后的图片一共有30张
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            print(current_time, iters, 'loss = %.8f, mse = %.8f' % (current_loss, mses[-1]))
            history_iters.append(iters)
            if current_loss < 0.000001:  # converge
                break
    # tmp_mse_list[method].append(mses[-1])
    return mses[-1]