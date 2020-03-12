"""
A pytorch implementation of deepfm

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.

"""

import torch
from torch import nn
import torch.nn.functional as F


class DeepFM(nn.Module):
    def __init__(self, args):
        super(DeepFM, self).__init__()
        self.field_size = args.field_size  # field的数量 (category类型的变量只算一个field)
        self.feature_sizes = args.feature_sizes  # feature的总数(包括numeric和category变量one-hot后的总数)
        self.embedding_size = args.embedding_size  # 隐变量的数量
        self.is_shallow_dropout = args.is_shallow_dropout  # 1阶组合是否使用dropout
        self.dropout_shallow = args.dropout_shallow  # 1阶组合的dropout概率

        self.deep_layers = args.deep_layers
        self.is_deep_dropout = args.is_deep_dropout
        self.is_batch_norm = args.is_batch_norm
        self.dropout_deep = args.dropout_deep
        self.deep_layers_activation = args.deep_layers_activation

        self.use_lr = args.use_lr
        self.use_fm = args.use_fm
        # self.use_ffm = args.use_ffm
        self.use_deep = args.use_deep

        # torch.manual_seed(args.random_seed)

        """
            check model type
        """
        if self.use_lr:
            print("The model has LR part")
        if self.use_fm:
            print("The model has fm network")
        if self.use_deep:
            print("The model has deep network")
        if not self.use_lr and not self.use_fm and not self.use_deep:
            print("You have to choose more than one of (lr, fm, deep) models to use")
            exit(-1)

        """
            lr part
        """
        if self.use_lr:
            print('Init lr part')
            self.bias = torch.nn.Parameter(torch.randn(1))  # w_0
            self.fm_first_order_embeddings = nn.Embedding(self.feature_sizes, 1)  # 一阶组合的系数 w
            print("Init lr part succeed")

        """
            fm part
        """
        if self.use_fm:
            print("Init fm part")
            self.bias = torch.nn.Parameter(torch.randn(1))  # w_0
            # self.fm_first_order_embeddings = nn.ModuleList(
            #     [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
            self.fm_first_order_embeddings = nn.Embedding(self.feature_sizes, 1)  # 一阶组合的系数 w
            if self.dropout_shallow:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            # self.fm_second_order_embeddings = nn.ModuleList(
            #     [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
            self.fm_second_order_embeddings = nn.Embedding(self.feature_sizes, self.embedding_size)  # 二阶组合的系数矩阵 V
            if self.dropout_shallow:
                self.fm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])
            print("Init fm part succeed")

        """
            deep part
        """
        if self.use_deep:
            print("Init deep part")
            # if not self.use_fm and not self.use_ffm:
            #     self.fm_second_order_embeddings = nn.ModuleList(
            #         [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])

            if self.is_deep_dropout:
                self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])

            self.linear_1 = nn.Linear(self.field_size * self.embedding_size, self.deep_layers[0])
            if self.is_batch_norm:
                self.batch_norm_1 = nn.BatchNorm1d(self.deep_layers[0])
            if self.is_deep_dropout:
                self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])
            for i, h in enumerate(self.deep_layers[1:], 1):
                setattr(self, 'linear_' + str(i + 1), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
                if self.is_batch_norm:
                    setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(self.deep_layers[i]))
                if self.is_deep_dropout:
                    setattr(self, 'linear_' + str(i + 1) + '_dropout', nn.Dropout(self.dropout_deep[i + 1]))

            print("Init deep part succeed")

        print("Init succeed")

    def forward(self, Xi, Xv):
        """
        :param Xi: index input tensor, batch_size * k
        :param Xv: value input tensor, batch_size * k
        :return: the last output
        """
        """
            lr part
        """
        if self.use_lr:
            fm_first_order_emb_arr = self.fm_first_order_embeddings(Xi) * Xv.view(-1, self.field_size, 1)

            # fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
            fm_first_order = fm_first_order_emb_arr.view(-1, self.field_size)

        """
            fm part
        """
        if self.use_fm:
            # fm_first_order_emb_arr = [(emb(Xi[:, i]).t() * Xv[:, i]).t() for i, emb in
            #                           enumerate(self.fm_first_order_embeddings)]
            fm_first_order_emb_arr = self.fm_first_order_embeddings(Xi) * Xv.view(-1, self.field_size, 1)

            # fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
            fm_first_order = fm_first_order_emb_arr.view(-1, self.field_size)

            if self.is_shallow_dropout:
                fm_first_order = self.fm_first_order_dropout(fm_first_order)

            # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
            # fm_second_order_emb_arr = [(emb(Xi[:, i]).t() * Xv[:, i]).t() for i, emb in
            #                            enumerate(self.fm_second_order_embeddings)]
            # fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
            fm_second_order_emb_arr = self.fm_second_order_embeddings(Xi) * Xv.view(-1, self.field_size, 1)
            fm_sum_second_order_emb = torch.sum(fm_second_order_emb_arr, dim=1)

            fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb  # (x+y)^2
            # fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
            # fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)  # x^2+y^2
            fm_second_order_emb_square_sum = torch.sum(torch.pow(fm_second_order_emb_arr, 2), dim=1)

            fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5

            if self.is_shallow_dropout:
                fm_second_order = self.fm_second_order_dropout(fm_second_order)

        """
            deep part
        """
        if self.use_deep:
            # if self.use_fm:
            #     deep_emb = torch.cat(fm_second_order_emb_arr, 1)
            # elif self.use_ffm:
            #     deep_emb = torch.cat([sum(ffm_second_order_embs) for ffm_second_order_embs in ffm_second_order_emb_arr],
            #                          1)
            # else:
            #     deep_emb = torch.cat([(emb(Xi[:, i]).t() * Xv[:, i]).t() for i, emb in
            #                           enumerate(self.fm_second_order_embeddings)], 1)

            deep_emb = fm_second_order_emb_arr.view(-1, self.field_size * self.embedding_size)

            if self.is_deep_dropout:
                deep_emb = self.linear_0_dropout(deep_emb)
            x_deep = self.linear_1(deep_emb)
            if self.is_batch_norm:
                x_deep = self.batch_norm_1(x_deep)
            x_deep = F.relu(x_deep)
            if self.is_deep_dropout:
                x_deep = self.linear_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
                x_deep = F.relu(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)
            # print(x_deep)
        """
            sum
        """
        if self.use_fm and self.use_deep:
            total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + torch.sum(x_deep, 1) + self.bias
            return total_sum
        elif self.use_fm:
            total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + self.bias
            return total_sum
        elif self.use_lr:
            total_sum = torch.sum(fm_first_order, 1) + self.bias
            return total_sum
        else:
            print("forward function return nothing")
            exit(-1)
