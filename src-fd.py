#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Traceless'

"""

import argparse
import copy
import os
import time
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.Fed import FedAvg, get_avg_error
from utils.Fed import noise_add_laplace
from utils.data_handle import DataPreprocessing
from utils.update import LocalUpdate_sentry, LocalUpdate



matplotlib.use('AGG')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
SEED = 914
name = 'elliptic'
batch_size = 1024 * 2
learning_rate = 1e-2
epochs = 100
data_handle = DataPreprocessing(res_path='./results')


def numpy_to_tensor(data, device):
    if type(data).__module__ == np.__name__:
        data = th.from_numpy(data)
    return data.to(device).float()

class HiddenLayers(nn.Module):
    def __init__(self, dims, isBatchNorm=True, dropout_rate=0.2):
        """Class to add hidden layers to networks

        Args:
            options (dict): Configuration dictionary.
        """
        super(HiddenLayers, self).__init__()
        self.layers = nn.ModuleList()
        # dims =

        for i in range(1, len(dims)):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            if isBatchNorm:
                self.layers.append(nn.BatchNorm1d(dims[i]))
            self.layers.append(nn.LeakyReLU(inplace=False))
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
        # self.layers.append(nn.Softmax(dim=1))
        self.layers.append(nn.Softmax(dim=1))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class tabular_data(Dataset):
    def __init__(self, x, label):
        self.data, self.label = x, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        cluster = int(self.label[idx])
        return sample, cluster

def main(args):
    # names = ['elliptic']
    # focal_mode = args.focal_loss
    res_path = './results' + '/fed_mlp_exp_performance_sadp_src-fd_scale_elliptic_iid_{}'.format(args.iid)

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    nm = args.data_set
    runs = args.runs  # runs 次实验结果的平均值
    ap = np.zeros(runs)
    precision_list = np.zeros(runs)
    recall_list = np.zeros(runs)
    f1_value_list = np.zeros(runs)
    auc_value_list = np.zeros(runs)
    # res_list = {'rauc': [], 'ap': [], 'fpr': [], 'tpr': [], 'f1': []}
    res_list = {'ap': [], 'precision_list': [], 'recall_list': [], 'f1': []}
    filename = nm.strip()
    ori_data, labels = data_handle.data_handle_for_diff_dataset(data_name=filename,
                                                                file_path=args.input_path + filename + '.csv')
    outlier_indices = np.where(labels == 1)[0]
    outliers = ori_data[outlier_indices]
    device = th.device('cuda:' + args.dn if th.cuda.is_available() else 'cpu')
    args.device = device
    test_items = np.round(np.linspace(start=0.02, stop=0.5, num=25), 2)
    # if args.test_name == 'FedAvg':
    #     test_items = [0]

    for test_item_index, test_item in enumerate(test_items):
        res_df_list = []
        print('current test item is {}'.format(test_item))
        for run_num in np.arange(runs):
            random_seed = run_num
            x_train, x_test, y_train, y_test = train_test_split(ori_data, labels, test_size=0.2,
                                                                random_state=random_seed,
                                                                stratify=labels)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            print(filename + ': round' + str(run_num))
            outlier_indices = np.where(y_train == 1)[0]
            nor_indices = np.where(y_train == 0)[0]
            n_outliers = len(outlier_indices)
            n_norliers = len(nor_indices)
            imbance_rate = n_outliers / n_norliers
            print("Original training size: {}, No. outliers: {}, imbanced rate: {}".format(x_train.shape[0],
                                                                                           n_outliers,
                                                                                           imbance_rate))

            label_rate = args.label_rate
            x_train_labeled, x_train_no_label, y_train_labeled, y_train_no_label = train_test_split(x_train,
                                                                                                    y_train,
                                                                                                    train_size=label_rate,
                                                                                                    stratify=y_train,
                                                                                                    random_state=random_seed)
            # x_train_labeled = x_train
            # y_train_labeled = y_train
            print("Training data size: %d" % (x_train_labeled.shape[0]))
            print('counter: {}'.format(Counter(y_train_labeled)))
            print("Testing data size: %d" % (x_test.shape[0]))
            # load dataset and split users
            # sample users
            if args.iid:
                dict_users = data_handle.iid_sample(y_train_labeled, args.num_users)
                print('iid*********************************')
            else:
                print('no-iid*********************************')
                dict_users = data_handle.noiid_sample_cluster(y_train_labeled, args.num_users, x_train_labeled)

            in_dim = x_train.shape[1]

            hidde_1 = in_dim
            hidde_2 = in_dim
            hidde_3 = 20
            class_num = 2

            model = HiddenLayers(dims=[in_dim, hidde_1, hidde_2, hidde_3, class_num], isBatchNorm=False, dropout_rate=0)
            model = model.to(device)
            print('Model summary ' + '*' * 50)
            print(model)
            # copy weights
            #
            model.train()
            w_glob = model.state_dict()
            # training
            loss_train = []
            cv_loss, cv_acc = [], []
            val_loss_pre, counter = 0, 0
            net_best = None
            best_loss = None
            val_acc_list, net_list = [], []

            print("Aggregation over all clients")
            w_locals = [w_glob for i in range(args.num_users)]

            tmp_res_list = {'ap': [], 'precision_list': [], 'recall_list': [], 'f1': [], 'auc': [], 'error': [],
                            'budget': []}
            args.privacy_budget = test_item
            args.num_items_train = len(y_train_labeled)
            args.delta = 0.01  # 默认是0.01
            # noise_scale = Privacy_account(args=args, threshold_epochs=args.epochs)
            ori_loss = None
            error_max_backup = [0] * args.num_users
            w_locals_backup = [w_glob for i in range(args.num_users)]
            new_budget_list = [args.privacy_budget] * args.num_users
            max_error = [0] * args.num_users
            for iter in range(args.epochs):
                loss_locals = []
                if args.test_name == 'src-fd':
                    idxs_users = range(args.num_users)
                    for idx in idxs_users:
                        local = LocalUpdate_sentry(args=args, x=x_train_labeled, label=y_train_labeled,
                                                   idxs=dict_users[idx],
                                                   budget=args.privacy_budget, focal_mode=True,
                                                   max_error=max_error[idx])

                        w, loss = local.train(net=copy.deepcopy(model).to(device), budget=new_budget_list[idx])
                        max_error[idx] = local.max_error
                        w_locals[idx] = copy.deepcopy(w)
                        loss_locals.append(copy.deepcopy(loss))
                        # with th.no_grad():
                        #     # error = get_avg_error(global_true=w_glob, global_fed=w_local)
                        #     tmp_error = get_avg_error(global_true=w_locals_backup[idx], global_fed=w)
                        new_budget_list[idx] = local.calc_budget()

                    w_glob, _ = FedAvg(w_locals)

                    new_budget = np.mean(new_budget_list)
                    test_name = 'src-fd'
                    tmp_res_list['error'].append(np.mean(loss_locals))
                    tmp_res_list['budget'].append(new_budget)

                elif args.test_name == 'FedAvg':
                    new_budget = args.privacy_budget
                    test_name = 'FedAvg'
                    loss_locals = []

                    idxs_users = range(args.num_users)
                    max_error = [0] * args.num_users
                    for idx in idxs_users:
                        local = LocalUpdate(args=args, x=x_train_labeled, label=y_train_labeled,
                                                   idxs=dict_users[idx],
                                                   focal_mode=True,
                                                   )

                        w, loss = local.train(net=copy.deepcopy(model).to(device))
                        w_locals[idx] = copy.deepcopy(w)
                        loss_locals.append(copy.deepcopy(loss))
                        with th.no_grad():
                            # error = get_avg_error(global_true=w_glob, global_fed=w_local)
                            tmp_error = get_avg_error(global_true=w_locals_backup[idx], global_fed=w)

                    w_glob, _  = FedAvg(w_locals)
                else:
                    raise NotImplementedError

                # copy weight to net_glob
                model.load_state_dict(w_glob)
                # print loss
                loss_avg = sum(loss_locals) / len(loss_locals)
                loss_train.append(loss_avg)

                w_locals_backup = [w_glob for i in range(args.num_users)]
                ## eval 阶段
                with th.no_grad():
                    x_test = numpy_to_tensor(data=x_test, device=device)
                    outputs = model.forward(x=x_test)
                    pre_label = outputs[:, 1]
                    pre_label: th.Tensor
                    pre_label = pre_label.cpu().detach().numpy()
                    ap[run_num], precision_list[run_num], recall_list[run_num], f1_value_list[
                        run_num], auc_value_list[run_num] = data_handle.ap_auc_Performance(mse=pre_label,
                                                                                           labels=y_test)
                    tmp_res_list['ap'].append(ap[run_num])
                    tmp_res_list['precision_list'].append(precision_list[run_num])
                    tmp_res_list['recall_list'].append(recall_list[run_num])
                    tmp_res_list['f1'].append(f1_value_list[run_num])
                    tmp_res_list['auc'].append(auc_value_list[run_num])

                print('Round {:3d}, Training average loss: {:.3f}, new_budget: {},  ap: {:.3f}, '
                      'f1: {:.3f}'.format(iter, loss_avg, new_budget, ap[run_num], f1_value_list[run_num]))

            tmp_res_df = pd.DataFrame(dict([k, pd.Series(v)] for k, v in tmp_res_list.items()))
            res_df_list.append(copy.deepcopy(tmp_res_df))
            print(res_df_list)
            file_path = os.path.join(res_path,
                                     'Performance_fed_avg_src-fd_laplace' + '_{}'.format(nm) + '{}_{}.xlsx'.format(
                                         test_item,
                                         test_name))
            writer = pd.ExcelWriter(path=file_path)
            average_df = pd.DataFrame()

            for idx in range(len(res_df_list)):
                tmp_res_df = res_df_list[idx]
                tmp_res_df.to_excel(excel_writer=writer, sheet_name='res_{}'.format(idx), index=False)
                if idx == 0:
                    average_df = tmp_res_df.copy(deep=True)
                else:
                    average_df = (average_df + tmp_res_df) / 2

            average_df.to_excel(excel_writer=writer, sheet_name='average_res', index=False)
            writer.save()
            writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5,
                        help="how many times we repeat the experiments to obtain the average performance")
    parser.add_argument("--data_set", type=str, default=name, help="data set name")
    parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
    parser.add_argument("--cont_rate", type=float, default=0.03,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--label_rate", type=float, default=0.8,
                        help="the labeled data rate in the training data")
    parser.add_argument('--dn', '--device_num', type=str, default='0', help='gpu num')
    parser.add_argument('--focal_loss', action='store_false', help='所使用的loss函数')
    # federated arguments
    parser.add_argument('--epochs', type=int, default=epochs, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=3, help="number of users: K")  # 横向联邦学习场景下的用户数
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")  # 本地用户更新的次数
    parser.add_argument('--local_bs', type=int, default=batch_size, help="local batch size: B")  # 本地用户更新的时的batch size
    parser.add_argument('--bs', type=int, default=batch_size, help="test batch size")
    parser.add_argument('--lr', type=float, default=learning_rate, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--test_name', type=str, default='src-fd', help="aggregation pattern")

    # other arguments
    # parser.add_argument('--iid', type=bool, default=False, help='whether i.i.d or not')
    # parser.add_argument('--iid', type=bool, default=True, help='whether i.i.d or not')
    parser.add_argument('--iid', action='store_true', default=False, help='whether i.i.d or not')
    args = parser.parse_args()
    main(args)

"""
python src-fd.py --test_name src-fd 
python src-fd.py --iid --test_name src-fd 
python src-fd.py --test_name FedAvg

"""
