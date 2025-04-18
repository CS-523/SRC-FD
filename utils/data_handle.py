#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__author__ = 'Traceless'
"""

import gc
import multiprocessing as mlp
import os
from collections import Counter

import matplotlib.pyplot as plt
# import csv
import numpy as np
import pandas as pd
import seaborn as sns
# import pyreadr
from pandas import set_option
from sklearn import manifold
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve, average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans

cpu_count = int(mlp.cpu_count() / 3)  # 使用1/3个核心进行数据的处理



def get_color_list():
    # Define colors to be used for each class/cluster
    color_list = ['#66BAFF', '#FFB56B', '#8BDD89', '#faa5f3', '#fa7f7f',
                  '#008cff', '#ff8000', '#04b000', '#de4bd2', '#fc3838',
                  '#004c8b', "#964b00", "#026b00", "#ad17a1", '#a80707',
                  "#00325c", "#e41a1c", "#008DF9", "#570950", '#732929']

    color_list2 = ['#66BAFF', '#008cff', '#004c8b', '#00325c',
                   '#FFB56B', '#ff8000', '#964b00', '#e41a1c',
                   '#8BDD89', "#04b000", "#026b00", "#008DF9",
                   "#faa5f3", "#de4bd2", "#ad17a1", "#570950",
                   '#fa7f7f', '#fc3838', '#a80707', '#732929']

    return color_list, color_list2


def tsne(latent):
    """Reduces dimensionality of embeddings to 2, and returns it"""
    mds = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # mds = manifold.
    return mds.fit_transform(latent)


class DataPreprocessing(object):

    def __init__(self, res_path='../../res/'):
        self.res_path = res_path
        set_option('precision', 2)
        # todo
        # print('current res path is {}'.format(self.res_path))

    def get_dir_file(self, file_path='../../dataset/ECC'):
        """
        查看文件夹中的所有文件的路径
        :param file_path: 文件夹路径
        :return:
        """
        for dirname, _, filenames in os.walk(file_path):
            for filename in filenames:
                print("当前文件夹下有以下文件：")
                print(os.path.join(dirname, filename))
        # return

    def remove_same_column(self, data_set: pd.DataFrame):
        """
        1. 检测是否有相同的列名；
        2. 查看相同列名下的数值是否相等；
        3. 只保留一个列名所在的列；
        :param data_set:
        :return: 处理后的数据集
        """

        columns_list = data_set.columns.values.tolist()

    def data_fill_nall(self, df: pd.DataFrame, class_name='isFraud'):
        """
        缺失值填充方法：https://wenku.baidu.com/view/e2616e29874769eae009581b6bd97f192279bf89.html
        :param df:
        :param class_name: 类别所在那一列的名字
        :return:
        """
        class_list = np.unique(df[class_name].values)
        features = df.columns.values
        ori_data_num = df.shape[0]
        # res_df = pd.DataFrame()
        res_list = []
        for tmp_class in class_list:
            tmp_df = df[df[class_name] == tmp_class]  # (386097, 429)  (13903, 429)
            for features_tmp in features:
                # tmp_ori_num = tmp_df
                # tmp_df[features_tmp].fillna(tmp_df[features_tmp].mode(), inplace=True)  # 使用当前类别的众数进行填充
                tmp_df[features_tmp].fillna(tmp_df[features_tmp].mean(), inplace=True)  # 使用当前类别的众数进行填充
                # (386097, 429)-》(13903, 429)
                assert tmp_df[features_tmp].isnull().values.max() == 0, print('fill null error!!')
                # tmp_df[features_tmp] = tmp_df[features_tmp].interpolate()  # 使用插值法进行填充
            res_list.append(tmp_df.copy(deep=True))
        res_df = pd.concat(res_list, ignore_index=True)
        # 以后尽量不要使用append方法---会出现没有拼接成功的情况
        # for ind, res_tmp in enumerate(res_list):
        #     res_tmp: pd.DataFrame
        #
        #     if ind == 0:
        #         res_df = res_tmp.copy(deep=True)  # (386097, 429)
        #     else:  # (13903, 429)
        #
        #         res_df.append(res_tmp.copy(deep=True), ignore_index=False)
        change_data_num = res_df.shape[0]
        assert ori_data_num == change_data_num, print(ori_data_num, change_data_num)
        return res_df

    def get_sta_tar_file(self, file_path, out_sta=False):
        """
        返回目标文件的统计数据：
        1. 数据集的前10个数据；
        2. 数据集的shape；
        3. 数据集的info；
        4. 数据集中是否包含null；

        :param file_path: 目标文件路径
        :return:
        """
        postfix = file_path.split('.')[-1]
        if postfix == 'csv':
            df = pd.read_csv(file_path)
        elif postfix == 'pkl':
            df = pd.read_pickle(file_path)
        else:
            df = None
            print('该文件格式没有实现!!')
            raise NotImplementedError

        if out_sta:
            print('head: \n', df.head())
            print('shape: \n', df.shape)
            print('info: \n', df.info())
            print('isnull: \n', df.isnull().sum())
            set_option('precision', 2)
            print('describe: \n', df.describe())
            print('describe-T: \n', df.describe().T)

        return df

    def change_columns(self, df, char_flag='-', replace_char='_'):
        """
        更改dataframe中的columns，将所有char_flag变成replace_char
        :param df:
        :param char_flag:
        :param replace_char:
        :return:
        """
        df: pd.DataFrame
        columns = list(df.columns)
        new_columns = {}
        for column_name in columns:
            if char_flag in column_name:
                new_column_name = str(column_name).replace(char_flag, replace_char)
            else:
                new_column_name = column_name
            new_columns[column_name] = new_column_name
        print('new_column_name is \n')
        print(new_columns)
        df.rename(columns=new_columns, inplace=True)
        return df

    def get_label_cnt(self, df: pd.DataFrame, class_names: dict, class_column_name='Class'):
        """
        更改dataset中的标签:class_names = {0: 'normal', 1: 'fraud'}
        :param df: 数据集
        :param class_names: 类名所对应的索引值
        :return:
        """
        print('fraud cnt:')
        print(df[class_column_name].value_counts().rename(index=class_names))
        print(df.head(n=5))

        """
        fraud cnt:
                normal    284315
                fraud        492
        Name: Class, dtype: int64
        0    284315
        1       492
        """

    def split_dataset(self, df: pd.DataFrame, test_size=1 / 3, random_state=125, class_column_name='Class'):
        """

        :param df: 目标文件
        :param test_size: test文件的size
        :param random_state: 随机种子
        :param class_column_name: label所在的列名
        :return:
        """
        y = df[class_column_name]
        X = df.loc[:, df.columns != 'Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test

    # 对IEEE-CIS数据集进行处理：https://www.kaggle.com/maxduan/ieee-cis-12nd-solution-part-1
    # 对DeviceInfo，id_30，id_31进行处理，并生成一个是否有identity的特征
    # Process DeviceInfo,id_30,id_31 and generate a feature 'has identity'
    def id_split(self, dataframe, id_30_name="id_30", id_31_name="id_31"):
        """
        讲解：https://www.cnblogs.com/liuzeng/p/13841919.html
        :param dataframe:
        :return:
        """
        # series=data['列名'].str.split(',',expand=True)
        # 参数expand，这个参数取True时，会把切割出来的内容当做一列，产生多列。
        dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]  # 只要第1列
        dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]  # 只要第2列
        dataframe['OS_id_30'] = dataframe[id_30_name].str.split(' ', expand=True)[0]
        dataframe['browser_id_31'] = dataframe[id_31_name].str.split(' ', expand=True)[0]
        # 是否包含表达式，contains()
        # series=data['列名'].str.contains('we')
        # na=nan,     # 默认对空值不处理，即输出结果还是 NaN  na = True 就表示把有NAN的转换为布尔值True
        # na = False 就表示把有NAN的转换为布尔值True
        # 将含有特定字符的数据转换为标准格式
        dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
        dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
        dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
        dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
        dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
        dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
        dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
        dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
        dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
        dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
        dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
        dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
        dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
        dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
        dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
        dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
        dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'
        # 类别太稀疏的置为其他类
        # Classes that are too sparse are placed in other classes
        # 比如 Asus、Sony只有30、40，则把它们统一变成“others”
        dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[
                                                     dataframe.device_name.value_counts() < 100].index), 'device_name'] = "Others"
        dataframe['had_id'] = 1
        gc.collect()  # 清理内存

        return dataframe

    def fearute_handle_for_CIS(self):
        # train_len = len(train)
        pass

    # 对ECC数据集的处理：https://www.kaggle.com/kayademirs/fraud-detection-svm-random-forest-and-cnn/notebook
    def get_ecc_data(self, fraud_file_path):
        # fraud_file_path = os.path.join(file_path, file_name)
        dataset = self.get_sta_tar_file(file_path=fraud_file_path)
        self.get_label_cnt(df=dataset, class_names={0: 'normal', 1: 'fraud'}, class_column_name='Class')
        # 数据标准化
        std_scaler = StandardScaler()
        rob_scaler = RobustScaler()
        dataset['amount_scale'] = rob_scaler.fit_transform(dataset['Amount'].values.reshape(-1, 1))
        dataset['time_scale'] = rob_scaler.fit_transform(dataset['Time'].values.reshape(-1, 1))
        dataset.drop(['Time', 'Amount'], axis=1, inplace=True)
        amount_scale = dataset['amount_scale']
        time_scale = dataset['time_scale']
        dataset.drop(['amount_scale', 'time_scale'], axis=1, inplace=True)
        dataset.insert(0, 'amount_scale', amount_scale)
        dataset.insert(1, 'time_scale', time_scale)

        # 数据下采样
        fraud = dataset[dataset['Class'] == 1]
        normal = dataset[dataset['Class'] == 0][:492]
        normal_distributed_data = pd.concat([fraud, normal])
        sample_data = normal_distributed_data.sample(frac=1, random_state=42)
        X = sample_data.drop('Class', axis=1)
        y = sample_data['Class']

        x_orjinal_train, x_orjinal_test, y_orjinal_train, y_orjinal_test = train_test_split(X, y, test_size=0.33,
                                                                                            random_state=21)

        # print("x_orginal_train shape is {}".format(x_orginal_train.shape))
        # exit()
        y_train = y_orjinal_train
        y_test = y_orjinal_test

        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x_orjinal_train)
        x_train = pd.DataFrame(x_scaled)
        x_scaled1 = min_max_scaler.fit_transform(x_orjinal_test)
        x_test = pd.DataFrame(x_scaled1)

        x_train = x_train.values
        x_test = x_test.values

        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)

        x_train_mean = np.mean(x_train)
        x_train_std = np.std(x_train)

        x_test_mean = np.mean(x_test)
        x_test_std = np.std(x_test)

        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_test_mean) / x_test_std
        # 使用卷积网络的时候需要变成tensor变量
        # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.33, random_state=21)
        return x_train, x_validate, x_test, y_train, y_validate, y_test

    def normalize_data(self, df: pd.DataFrame):
        """

        :param df: 不带标签值的所有数据
        :return: 标准化后的np.array
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        if hasattr(df, 'values'):
            x_scaled = min_max_scaler.fit_transform(df.values)
        else:
            x_scaled = min_max_scaler.fit_transform(df)
        return x_scaled

    def vesta_data_handle(self, df: pd.DataFrame):
        """
        对Vesta数据集进行处理--归一化与数据填充
        :param df:
        :return:
        """

        # 空值填充

        null_sta = df.isnull().sum()
        print('null_sta is \n{}'.format(null_sta))

        # 归一化
        x_scaled = self.normalize_data(df)
        return x_scaled

    def del_nan_value(self):
        pass

    def data_handle_for_diff_dataset(self, data_name, file_path) -> object:
        if data_name != 'elliptic':
            df = self.get_sta_tar_file(file_path=file_path, out_sta=False)
        if data_name == 'handled_train_file':
            # df.dropna
            df.dropna(axis=0, how='any', inplace=True)  # 去掉所有存在空值的行

            class_col_name = 'isFraud'
            labels = df[class_col_name]
            rm_cols = [
                'TransactionID', 'TransactionDT', class_col_name,
            ]
            features_columns: list = df.columns.values.tolist()

            for col in rm_cols:
                if col in features_columns:
                    features_columns.remove(col)

            x_df = df[features_columns]
            x_scaled = self.vesta_data_handle(x_df)

        elif data_name == 'vesta_handle_data_fillna':  # 需要对原始数据进行填充
            class_col_name = 'isFraud'
            labels = df[class_col_name]
            rm_cols = [
                'TransactionID', 'TransactionDT', class_col_name,
            ]
            null_num = df.isnull().sum().max()

            if null_num > 0:
                print("null_num.max is {}".format(null_num))
                df = self.data_fill_nall(df, class_col_name)
            features_columns: list = df.columns.values.tolist()
            for col in rm_cols:
                if col in features_columns:
                    features_columns.remove(col)

            x_df = df[features_columns]
            x_scaled = self.vesta_data_handle(x_df)

        elif data_name == 'vesta_handle_data':
            class_col_name = 'isFraud'
            labels = df[class_col_name]
            rm_cols = [
                'TransactionID', 'TransactionDT', class_col_name,
            ]
            null_num = df.isnull().sum().max()

            if null_num > 0:
                print("null_num.max is {}".format(null_num))
                # df = self.data_fill_nall(df, class_col_name)
            features_columns: list = df.columns.values.tolist()
            for col in rm_cols:
                if col in features_columns:
                    features_columns.remove(col)

            x_df = df[features_columns]
            x_scaled = x_df.values
        elif data_name == 'creditcardfraud_normalised':
            class_col_name = 'class'
            labels = df[class_col_name]
            x_df = df.drop([class_col_name], axis=1)
            x_scaled = x_df.values
        elif data_name == 'ccFraud':

            class_col_name = 'fraudRisk'
            labels = df[class_col_name]
            x_df = df.drop([class_col_name], axis=1)
            x_scaled = x_df.values
        elif data_name == 'bitcoin':
            # save_file = os.path.join(save_path, 'bitcoin.csv')
            x_df = df.drop("label", axis=1)
            x_scaled = x_df.values
            labels = (df["label"] > 0).astype('int')
        elif data_name == 'elliptic':
            # prefix = '/'.join(file_path.split('.')[:-1])
            cat_features = []
            datadir_path = os.path.splitext(file_path)[0]
            # datadir_path = os.path.join(prefix, 'elliptic_bitcoin_dataset')
            classes_csv = os.path.join(datadir_path, 'elliptic_txs_classes.csv')
            edgelist_csv = os.path.join(datadir_path, 'elliptic_txs_edgelist.csv')
            features_csv = os.path.join(datadir_path, 'elliptic_txs_features.csv')

            classes = pd.read_csv(classes_csv)  # labels are 'unknown', '1'(illicit), '2'(licit)
            classes['class'] = classes['class'].map({'unknown': 2, '1': 1, '2': 0})
            edgelist = pd.read_csv(edgelist_csv)

            features = pd.read_csv(features_csv, header=None)  # features of the transactions: (203769, 167)
            # data = pd.concat([classes, features], axis=1)
            transaction_id_map = dict(zip(features[0].values, features.index.values))
            # transaction_id = np.unique(features[0].values)  # 203769
            num_features = features.shape[1]
            timesteps = np.unique(features[1])
            feature_idx = [i + 2 for i in range(93 + 72)]
            x_scaled_all = features.drop(columns=[0, 1])
            #     feat_data, labels, train_idx, test_idx, g, cat_features
            labels_all = classes['class']
            x_scaled = x_scaled_all[labels_all!=2].values
            labels = labels_all[labels_all != 2].values
            # index = list(range(len(labels)))
            # train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
            #                                                         random_state=2,
            #                                                         shuffle=True)
            # edgelist['txId1'] = edgelist['txId1'].map(transaction_id_map)
            # edgelist['txId2'] = edgelist['txId2'].map(transaction_id_map)
            # src = edgelist['txId1'].values
            # tgt = edgelist['txId2'].values
            # g = dgl.graph((src, tgt))
            # # g = dgl.DGLGraph(multigraph=True)
            #
            # g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
            # g.ndata['feat'] = torch.from_numpy(
            #     feat_data.to_numpy()).to(torch.float32)
            # # degrees = g.in_degrees()
            # # out_degrees = g.out_degrees()
            # g = dgl.add_self_loop(g)
            # graph_path = prefix + "graph-{}.bin".format(dataset)
            # dgl.data.utils.save_graphs(graph_path, [g])
        else:
            class_col_name = 'label'
            labels = df[class_col_name]
            x_df = df.drop([class_col_name], axis=1)
            x_scaled = x_df.values


            # x_scaled = df.values
            # raise NotImplementedError
        print("Counter(labels) is {}".format(Counter(labels)))
        imbalanced_rate = labels[labels == 1].shape[0] / labels[labels == 0].shape[0]
        print('balanced rate is \n{}'.format(imbalanced_rate))
        return x_scaled, labels

    def Find_Optimal_Cutoff(self, TPR, FPR, threshold):  # 利用约登指数 求最佳阈值
        y = TPR - FPR
        Youden_index = np.argmax(y)
        optimal_threshold = threshold[Youden_index]
        point = [TPR[Youden_index], FPR[Youden_index], threshold[Youden_index]]

        return point

    def aucPerformance(self, mse, labels, pos_label=1, verbose=False):
        """
        roc_curve与ap：需要设置正类符号
        pos_label = 1 默认正类是1，负类是0
        :param mse:
        :param labels:
        :param pos_label: 符号为1时，视为正类
        :param verbose: 默认不输出各个性能指标
        :return:
        """
        assert np.isnan(labels).any() == False, print(labels)
        assert np.isnan(mse).any() == False, print(mse)
        fpr, tpr, thresholds = roc_curve(labels, mse, pos_label=pos_label)

        point = self.Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
        tpr_value, fpr_value, threshold = point[0], point[1], point[2]
        # roc_auc = roc_auc_score(labels, mse)

        roc_auc = auc(fpr, tpr)

        ap = average_precision_score(labels, mse, pos_label=pos_label)
        mse = np.array(mse)
        mse[mse >= threshold] = 1
        mse[mse < threshold] = 0
        cm = confusion_matrix(y_true=labels, y_pred=mse)
        f1_value = f1_score(y_true=labels, y_pred=mse)
        if verbose:
            print('Test set confusion matrix: \n{}'.format(cm))
            print('Test set f1_value: \n{}'.format(f1_value))
            print("AUC-ROC: %.4f, AUC-PR: %.4f, trp_value： %.4f, fpr_value: %.4f" % (roc_auc, ap, tpr_value, fpr_value))
        return roc_auc, ap, tpr_value, fpr_value, f1_value

    def apPerformance(self, mse, labels, pos_label=1, verbose=False):
        """
        roc_curve与ap：需要设置正类符号
        pos_label = 1 默认正类是1，负类是0
        :param mse:
        :param labels:
        :param pos_label: 符号为1时，视为正类
        :param verbose: 默认不输出各个性能指标
        :return:
        """
        assert np.isnan(labels).any() == False, print(labels)
        assert np.isnan(mse).any() == False, print(mse)
        precision_list, recall_list, thresholds = precision_recall_curve(labels, mse, pos_label=pos_label)
        # fpr, tpr, thresholds = roc_curve(labels, mse, pos_label=pos_label)
        F1_list = 2* precision_list*recall_list/(precision_list+recall_list+1e-6)
        idx = np.argmax(F1_list)
        precision, recall, F1 = precision_list[idx], recall_list[idx], F1_list[idx]
        threshold = thresholds[idx]
        ap = average_precision_score(labels, mse, pos_label=pos_label)
        mse = np.array(mse)
        mse[mse >= threshold] = 1
        mse[mse < threshold] = 0
        cm = confusion_matrix(y_true=labels, y_pred=mse)
        f1_value = f1_score(y_true=labels, y_pred=mse)
        if verbose:
            print('Test set confusion matrix: \n{}'.format(cm))
            print('Test set f1_value: \n{}'.format(f1_value))
            print('Test set F1: \n{}'.format(F1))
        return ap, precision, recall, f1_value

    def ap_auc_Performance(self, mse, labels, pos_label=1, verbose=False):
        """
        roc_curve与ap：需要设置正类符号
        pos_label = 1 默认正类是1，负类是0
        :param mse:
        :param labels:
        :param pos_label: 符号为1时，视为正类
        :param verbose: 默认不输出各个性能指标
        :return:
        """
        assert np.isnan(labels).any() == False, print(labels)
        assert np.isnan(mse).any() == False, print(mse)
        fpr, tpr, _ = roc_curve(labels, mse, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        precision_list, recall_list, thresholds = precision_recall_curve(labels, mse, pos_label=pos_label)
        F1_list = 2* precision_list*recall_list/(precision_list+recall_list+1e-6)
        idx = np.argmax(F1_list)
        precision, recall, F1 = precision_list[idx], recall_list[idx], F1_list[idx]
        threshold = thresholds[idx]
        ap = average_precision_score(labels, mse, pos_label=pos_label)
        mse = np.array(mse)
        mse[mse >= threshold] = 1
        mse[mse < threshold] = 0
        cm = confusion_matrix(y_true=labels, y_pred=mse)
        f1_value = f1_score(y_true=labels, y_pred=mse)
        if verbose:
            print('Test set confusion matrix: \n{}'.format(cm))
            print('Test set f1_value: \n{}'.format(f1_value))
            print('Test set F1: \n{}'.format(F1))
        return ap, precision, recall, f1_value, roc_auc

    def ap_ac_Performance(self, mse, labels, pos_label=1, verbose=False):
        """
        roc_curve与ap：需要设置正类符号
        pos_label = 1 默认正类是1，负类是0
        :param mse:
        :param labels:
        :param pos_label: 符号为1时，视为正类
        :param verbose: 默认不输出各个性能指标
        :return:
        """
        assert np.isnan(labels).any() == False, print(labels)
        assert np.isnan(mse).any() == False, print(mse)
        precision_list, recall_list, thresholds = precision_recall_curve(labels, mse, pos_label=pos_label)
        # fpr, tpr, thresholds = roc_curve(labels, mse, pos_label=pos_label)
        F1_list = 2* precision_list*recall_list/(precision_list+recall_list+1e-6)
        idx = np.argmax(F1_list)
        precision, recall, F1 = precision_list[idx], recall_list[idx], F1_list[idx]
        threshold = thresholds[idx]
        ap = average_precision_score(labels, mse, pos_label=pos_label)
        mse = np.array(mse)
        mse[mse >= threshold] = 1
        mse[mse < threshold] = 0
        cm = confusion_matrix(y_true=labels, y_pred=mse)
        f1_value = f1_score(y_true=labels, y_pred=mse)
        acc = accuracy_score(y_true=labels, y_pred=mse)
        if verbose:
            print('Test set confusion matrix: \n{}'.format(cm))
            print('Test set f1_value: \n{}'.format(f1_value))
            print('Test set F1: \n{}'.format(F1))
            print('Test set acc: \n{}'.format(acc))
        return ap, precision, recall, f1_value, acc

    def get_stat_result(self, rauc_list, ap_list, tpr_list, fpr_list):
        mean_auc = np.mean(rauc_list)
        std_auc = np.std(rauc_list)
        mean_aucpr = np.mean(ap_list)
        std_aucpr = np.std(ap_list)
        mean_tpr = np.mean(tpr_list)
        std_tpr = np.std(tpr_list)
        mean_fpr = np.mean(fpr_list)
        std_fpr = np.std(fpr_list)
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f, average tpr: %.4f, average fpr: %.4f" % (
        mean_auc, mean_aucpr, mean_tpr, mean_fpr))
        print("std AUC-ROC: %.4f, std AUC-PR: %.4f, std tpr: %.4f, std fpr: %.4f" % (
        std_auc, std_aucpr, std_tpr, std_fpr))

    def get_stat_result_ap(self, ap_list):

        mean_aucpr = np.mean(ap_list)
        std_aucpr = np.std(ap_list)
        print("average AUC-PR: %.4f" % (mean_aucpr))
        print("std AUC-PR: %.4f" % (std_aucpr))

    def sve_inter_res(self, value_list, column_list=None):
        if column_list == None:
            column_list = range(len(value_list))
        for i, column in zip(value_list, column_list):
            pass

    def normal_data(self, value):
        value = np.array(value)
        value = (value - np.min(value)) / (np.max(value) - np.min(value))
        return value

    def visualise_clusters(self, embeddings, labels, plt_name="test", alpha=1.0, legend_title=None,
                           legend_labels=None,
                           ncol=1):
        """Function to plot clusters using embeddings from t-SNE and PCA

        Args:
            embeddings (ndarray): Embeddings
            labels (list): Class labels
            plt_name (str): Name to be used for the plot when saving.
            alpha (float): Defines transparency of data poinnts in the scatter plot
            legend_title (str): Legend title
            legend_labels ([str]): Defines labels to use for legends
            ncol (int): Defines number of columns to use for legends of the plot

        """
        # Define colors to be used for each class/cluster
        color_list, _ = get_color_list()
        # Used to adjust space for legends based on number of columns in the legend. ncol: subplot_adjust
        legend_space_adjustment = {"1": 0.9, "2": 0.9, "3": 0.75, "4": 0.65, "5": 0.65}
        # Initialize an empty dictionary to hold the mapping for color palette
        palette = {}
        # Map colors to the indexes.
        for i in range(len(color_list)):
            # palette[str(i)] = color_list[i]
            palette[i] = color_list[i]
        # Make sure that the labels are 1D arrays
        y = labels.reshape(-1, )
        # sizes = labels.reshape(-1)
        # sizes = list(map(str, sizes.tolist()))
        # Turn labels to a list
        # y = list(map(str, y.tolist()))
        y = list(map(int, y.tolist()))
        # Define number of sub-plots to draw. In this case, 2, one for PCA, and one for t-SNE
        SNE_flag = False  # 比较耗时间的哦
        if SNE_flag:

            img_n = 2
        else:
            img_n = 1
        # Initialize subplots
        fig, axs = plt.subplots(1, img_n, figsize=(9, 3.5), facecolor='w', edgecolor='k')
        # Adjust the whitespace around sub-plots
        fig.subplots_adjust(hspace=.1, wspace=.1)
        # adjust the ticks of axis.
        plt.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False)
        # Flatten axes if we have more than 1 plot. Or, return a list of 2 axs to make it compatible with multi-plot case.
        axs = axs.ravel() if img_n > 1 else [axs, axs]
        # Get 2D embeddings, using PCA
        print('Embeddings from PCA')
        pca = PCA(n_components=2)
        # Fit training data and transform
        embeddings_pca = pca.fit_transform(embeddings)  # if embeddings.shape[1]>2 else embeddings
        # Set the title of the sub-plot
        axs[0].title.set_text('Embeddings from PCA')
        # Plot samples, using each class label to define the color of the class.  就是通过PCA取出主成分：2维，横轴、纵轴均为其中的一维
        # styles = y + 1

        # sns_plt = sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], ax=axs[0], palette=palette, hue=y,
        #                           alpha=alpha, size=y, sizes=(20, 200))
        for i in list(set(y)):
            labels_y = labels.reshape(-1, )

            class_emd = embeddings_pca[labels_y == i]
            class_y = labels_y[labels_y == i]
            class_y = list(map(int, class_y.tolist()))
            class_alpha = (i + 0.9) / len(set(y))
            sns_plt = sns.scatterplot(x=class_emd[:, 0], y=class_emd[:, 1], ax=axs[0], palette=palette, hue=class_y,
                                      alpha=class_alpha)
        # Overwrite legend labels
        self.overwrite_legends(sns_plt, fig, ncol=ncol, labels=legend_labels, title=legend_title)
        # Get 2D embeddings, using t-SNE
        if SNE_flag:
            print('Embeddings from t-SNE')
            embeddings_tsne = tsne(embeddings)  # if embeddings.shape[1]>2 else embeddings
            # Set the title of the sub-plot
            axs[1].title.set_text('Embeddings from t-SNE')
            # Plot samples, using each class label to define the color of the class.
            # sns_plt = sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], ax=axs[1], palette=palette, hue=y, s=20,
            #                           alpha=alpha)
            for i in list(set(y)):
                labels_y = labels.reshape(-1, )
                class_emd = embeddings_tsne[labels_y == i]
                class_y = labels_y[labels_y == i]
                class_y = list(map(int, class_y.tolist()))
                class_alpha = (i + 0.9) / len(set(y))
                sns_plt = sns.scatterplot(x=class_emd[:, 0], y=class_emd[:, 1], ax=axs[1], palette=palette, hue=class_y,
                                          alpha=class_alpha)
            # Overwrite legend labels
            self.overwrite_legends(sns_plt, fig, ncol=ncol, labels=legend_labels, title=legend_title)
            # Remove legends in sub-plots
            axs[0].get_legend().remove()
            axs[1].get_legend().remove()
        # Adjust the scaling factor to fit your legend text completely outside the plot
        # (smaller value results in more space being made for the legend)
        plt.subplots_adjust(right=legend_space_adjustment[str(ncol)])
        # Get the path to the project root: 当前脚本路径

        # root_path = os.path.dirname(os.path.dirname(__file__))
        root_path = self.res_path
        # Define the path to save the plot to.
        fig_path = os.path.join(root_path, plt_name + ".png")
        # Define tick params
        plt.tick_params(axis=u'both', which=u'both', length=0)
        # Save the plot
        plt.savefig(fig_path, bbox_inches="tight")
        # plt.show()
        # Clear figure just in case if there is a follow-up plot.
        plt.clf()
        return embeddings_pca, labels

    def save_embeddings_pca(self, embeddings_pca, labels, file_path):
        file_path = file_path + '.csv'
        embeddings_pca_df = pd.DataFrame(embeddings_pca)
        embeddings_pca_df['label'] = labels
        embeddings_pca_df.to_csv(file_path, index=False)

    def overwrite_legends(self, sns_plt, fig, ncol, labels, title=None):
        """Overwrites the legend of the plot

        Args:
            sns_plt (object): Seaborn plot object to manage legends
            c2l (dict): Dictionary mapping classes to labels
            fig (object): Figure to be edited
            ncol (int): Number of columns to use for legends
            title (str): Title of legend
            labels (list): Class labels

        """
        # Get legend handles and labels
        handles, legend_txts = sns_plt.get_legend_handles_labels()
        # Turn str to int before sorting ( to avoid wrong sort order such as having '10' in front of '4' )
        legend_txts = [int(d) for d in legend_txts]
        # legend_txts = [int(d) for d in legend_txts]
        # Sort both handle and texts so that they show up in a alphabetical order on the plot
        legend_txts, handles = (list(t) for t in zip(*sorted(zip(legend_txts, handles))))
        # Define the figure title
        title = title or "Cluster"
        # Overwrite the legend labels and add a title to the legend
        fig.legend(handles, labels, loc="center right", borderaxespad=0.1, title=title, ncol=ncol)
        sns_plt.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
        sns_plt.tick_params(top=False, bottom=False, left=False, right=False)

    def get_visible_flage(self, data, labels, rate=0.2, plt_name="test", alpha=1.0):
        """

        :param data: 数据
        :param labels: 标签
        :param rate: 要显示的比例
        :param plt_name: figure的名称
        :param alpha: 透明度：暂时没有实现
        :param legend_title: legend的名称
        :param legend_labels:
        :param ncol:
        :return:
        """
        legend_title = None
        legend_labels = None
        ncol = 1
        if rate < 1:

            X_plot, _, y_plot, _ = train_test_split(data, labels, train_size=rate)
        else:
            X_plot, y_plot = data, labels
        # Number of columns for legends, where each class corresponds to a cluster，每一个class被当做一类
        ncol = len(list(set(labels)))
        # clegends = ["A", "B", "C", "D", ...]..choose first ncol characters, one per cluster
        clegends = list("0123456789")[0:ncol]
        # Show clusters only
        embeddings_pca, labels = self.visualise_clusters(embeddings=X_plot, labels=y_plot, plt_name=plt_name,
                                                         legend_title="Classes",
                                                         legend_labels=clegends, alpha=alpha, )
        return embeddings_pca, labels

    def plot_embbeding_compare(self, embeddings, labels, plt_name="test", legend_title=None,
                           legend_labels=[''],
                           ncol=1, title_list=['a']):
        img_n = len(embeddings)

        # Number of columns for legends, where each class corresponds to a cluster，每一个class被当做一类
        if img_n > 0:

            ncol = len(list(set(labels[0])))
        else:
            ncol = len(list(set(labels)))
        # clegends = ["A", "B", "C", "D", ...]..choose first ncol characters, one per cluster
        clegends = list("0123456789")[0:ncol]
        legend_labels = clegends
        # Show clusters only
        # Define colors to be used for each class/cluster
        color_list, _ = get_color_list()
        # Used to adjust space for legends based on number of columns in the legend. ncol: subplot_adjust
        legend_space_adjustment = {"1": 0.9, "2": 0.9, "3": 0.75, "4": 0.65, "5": 0.65}
        # Initialize an empty dictionary to hold the mapping for color palette
        palette = {}
        # Map colors to the indexes.
        for i in range(len(color_list)):
            # palette[str(i)] = color_list[i]
            palette[i] = color_list[i]

        title_list = list(range(img_n))
        # Initialize subplots
        fig, axs_tmp = plt.subplots(1, img_n, figsize=(9, 3.5), facecolor='w', edgecolor='k')
        # Adjust the whitespace around sub-plots
        fig.subplots_adjust(hspace=.1, wspace=.1)
        # adjust the ticks of axis.
        plt.tick_params(axis='both', which='both', left=False, right=False, bottom=False, top=False, labelbottom=False)
        # Flatten axes if we have more than 1 plot. Or, return a list of 2 axs to make it compatible with multi-plot case.
        axs = axs_tmp
        # if img_n > 1:
        #     for i in range(img_n):
        #         axs.append(axs_tmp)
        # else:
        #     axs = axs.ravel

        # axs = axs.ravel() if img_n > 1 else [axs, axs, axs]
        # Get 2D embeddings, using PCA
        print('Embeddings from PCA')
        # Set the title of the sub-plot
        # axs[0].title.set_text('Embeddings from PCA')
        for fig_num in range(img_n):
            y = labels[fig_num]
            embeddings_pca = embeddings[fig_num]
            for i in list(set(y)):
                labels_y = y.reshape(-1, )
                class_emd = embeddings_pca[labels_y == i]
                class_y = labels_y[labels_y == i]
                class_y = list(map(int, class_y.tolist()))
                class_alpha = (i + 0.9) / len(set(y))
                axs[fig_num].title.set_text(title_list[fig_num])
                sns_plt = sns.scatterplot(x=class_emd[:, 0], y=class_emd[:, 1], ax=axs[fig_num], palette=palette, hue=class_y,
                                          alpha=class_alpha)
            self.overwrite_legends(sns_plt, fig, ncol=ncol, labels=legend_labels, title=legend_title)

        # Adjust the scaling factor to fit your legend text completely outside the plot
        # (smaller value results in more space being made for the legend)
        # plt.subplots_adjust(right=legend_space_adjustment[str(ncol)])
        # Get the path to the project root: 当前脚本路径

        # root_path = os.path.dirname(os.path.dirname(__file__))
        root_path = self.res_path
        # Define the path to save the plot to.
        fig_path = os.path.join(root_path, plt_name + ".svg")
        # Define tick params
        plt.tick_params(axis=u'both', which=u'both', length=0)
        # Save the plot
        plt.savefig(fig_path, bbox_inches="tight",format='svg',dpi=50)
        plt.show()
        # Clear figure just in case if there is a follow-up plot.
        plt.clf()

    def iid_sample(self, labels, num_users):
        num_items = int(len(labels) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(labels))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    def noiid_sample_cluster(self, y_train, num_users, x_train):
        # 由于自身文章主要关注与隐私安全，并不是性能的提高，因此对于非独立同分布的情况，暂时没有涉及
        # sort labels
        dict_users, all_idxs = {}, np.arange(x_train.shape[0])

        normal_train_data = x_train[y_train==0]
        normal_all_idxs = all_idxs[y_train==0]
        abnormal_train_data = x_train[y_train==1]
        abnormal_all_idxs = all_idxs[y_train == 1]
        kmeans = KMeans(num_users)
        kmeans.fit(normal_train_data)
        labels = kmeans.predict(normal_train_data)
        # data['label'] = labels
        # plt.scatter(x_train[:, 0], x_train[:, 1], c=labels)
        # plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.show()
        # num_items = int(len(labels) / num_users)
        # dict_users, all_idxs = {}, [i for i in range(len(labels))]

        for i in range(num_users):
            dict_users[i] = normal_all_idxs[labels==i].tolist()
            # all_idxs = list(set(all_idxs) - dict_users[i])

        kmeans = KMeans(num_users)
        kmeans.fit(abnormal_train_data)
        labels = kmeans.predict(abnormal_train_data)
        num_data = []
        for i in range(num_users):
            dict_users[i].extend(abnormal_all_idxs[labels==i].tolist())
            # all_idxs = list(set(all_idxs) - dict_users[i])
            num_data.append(len(dict_users[i]))
        print('iid: nums: {}'.format(num_data))
        return dict_users

    def noiid_sample_cluster_same(self, labels, num_users, x_train):
        # 由于自身文章主要关注与隐私安全，并不是性能的提高，因此对于非独立同分布的情况，暂时没有涉及
        # sort labels

        kmeans = KMeans(num_users)
        kmeans.fit(x_train)
        labels = kmeans.predict(x_train)
        # data['label'] = labels
        # plt.scatter(x_train[:, 0], x_train[:, 1], c=labels)
        # plt.xlabel('Feature 1')
        # plt.ylabel('Feature 2')
        # plt.show()
        # num_items = int(len(labels) / num_users)
        # dict_users, all_idxs = {}, [i for i in range(len(labels))]
        dict_users, all_idxs = {}, np.arange(x_train.shape[0])
        for i in range(num_users):
            dict_users[i] = all_idxs[labels == i].tolist()
            # all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    def noiid_sample(self, labels, num_users):
        # 由于自身文章主要关注与隐私安全，并不是性能的提高，因此对于非独立同分布的情况，暂时没有涉及
        # sort labels
        # kmeans = KMeans(num_users)
        # kmeans.fit(x_train)

        raise NotImplementedError

    def extract_rate(self, df, threshold):
        rate_df = np.array(df.values < threshold)
        sum_success = np.sum(rate_df, axis=0)
        success_rate = sum_success / df.shape[0]
        return success_rate

def main():
    pass


if __name__ == '__main__':
    main()
