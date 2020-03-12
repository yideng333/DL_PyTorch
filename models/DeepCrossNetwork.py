"""
A pytorch implementation of Deep & Cross Network

Reference:
[1] Deep & Cross Network for Ad Click Predictions
Ruoxi Wang,Stanford University,Stanford, CA,ruoxi@stanford.edu
Bin Fu,Google Inc.,New York, NY,binfu@google.com
Gang Fu,Google Inc.,New York, NY,thomasfu@google.com
Mingliang Wang,Google Inc.,New York, NY,mlwang@google.com

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCN(torch.nn.Module):

    def __init__(self, args):
        super(DCN, self).__init__()
        self.cat_field_size = args.cat_field_size  # 类别因子的数量
        self.cat_feature_sizes = args.cat_feature_sizes  # 类别因子one-hot后的数量
        self.num_feature_sizes = args.num_feature_sizes  # 数值型因子的数量
        self.embedding_size = args.embedding_size
        # self.h_depth = h_depth
        self.deep_layers = args.deep_layers
        self.is_deep_dropout = args.is_deep_dropout
        self.dropout_deep = args.dropout_deep
        self.h_cross_depth = args.h_cross_depth
        # self.h_inner_product_depth = args.h_inner_product_depth
        self.inner_product_layers = args.inner_product_layers
        self.is_inner_product_dropout = args.is_inner_product_dropout
        self.dropout_inner_product_deep = args.dropout_inner_product_deep
        self.deep_layers_activation = args.deep_layers_activation

        self.is_batch_norm = args.is_batch_norm
        self.random_seed = args.random_seed
        self.use_cross = args.use_cross
        self.use_inner_product = args.use_inner_product
        self.use_deep = args.use_deep
        self.use_wide = args.use_wide
        self.wide_output_size = args.wide_output_size

        # torch.manual_seed(self.random_seed)

        """
            check model type
        """
        if self.use_wide:
            print("The model has wide network")
        if self.use_cross:
            print("The model has cross network")
        if self.use_deep:
            print("The model has deep network")
        if self.use_inner_product:
            print("The model has inner product network")

        if not self.use_deep and not self.use_cross and not self.use_inner_product and not self.use_wide:
            print("You have to choose more than one of (cross network, deep network, inner product network) to use")
            exit(1)

        cat_size = 0

        """
            embeddings
        """
        # self.embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        # cat_feature需要进行embedding
        self.embeddings = nn.Embedding(self.cat_feature_sizes, self.embedding_size)

        """
            wide part
        """
        if self.use_wide:
            print("Init wide network")
            self.wide_embeddings = nn.Embedding(self.cat_feature_sizes, 1)
            self.wide_linear = nn.Linear(self.cat_field_size + self.num_feature_sizes, self.wide_output_size)
            self.wide_dropout = nn.Dropout(0.5)
            cat_size += self.wide_output_size
        """
            cross part
        """
        if self.use_cross:
            print("Init cross network")
            cross_feature_size = self.num_feature_sizes + self.cat_field_size*self.embedding_size
            for i in range(self.h_cross_depth):
                setattr(self, 'cross_weight_' + str(i+1),
                        torch.nn.Parameter(torch.randn(cross_feature_size)))
                setattr(self, 'cross_bias_' + str(i + 1),
                        torch.nn.Parameter(torch.randn(cross_feature_size)))
            print("Cross network finished")
            cat_size += cross_feature_size

        """
            inner prodcut part
        """
        if self.use_inner_product:
            print("Init inner product network")
            # self.inner_product_mask = torch.triu(torch.ones(self.cat_field_size, self.cat_field_size), diagonal=1).byte()
            self.inner_product_size = int(self.cat_field_size * (self.cat_field_size-1)/2)
            # print(self.inner_product_size)
            if self.is_inner_product_dropout:
                self.inner_product_0_dropout = nn.Dropout(self.dropout_inner_product_deep[0])
            self.inner_product_linear_1 = nn.Linear(self.inner_product_size, self.inner_product_layers[0])
            if self.is_inner_product_dropout:
                self.inner_product_1_dropout = nn.Dropout(self.dropout_inner_product_deep[1])
            if self.is_batch_norm:
                self.inner_product_batch_norm_1 = nn.BatchNorm1d(self.inner_product_layers[0])

            for i, h in enumerate(self.inner_product_layers[1:], 1):
                setattr(self, 'inner_product_linear_' + str(i + 1), nn.Linear(self.inner_product_layers[i - 1], self.inner_product_layers[i]))
                if self.is_batch_norm:
                    setattr(self, 'inner_product_batch_norm_' + str(i + 1), nn.BatchNorm1d(self.inner_product_layers[i]))
                if self.is_deep_dropout:
                    setattr(self, 'inner_product_' + str(i + 1) + '_dropout', nn.Dropout(self.dropout_inner_product_deep[i + 1]))
            cat_size += self.inner_product_layers[-1]
            print("Inner product network finished")

        """
            deep part
        """
        if self.use_deep:
            print("Init deep part")

            if self.is_deep_dropout:
                self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])
            self.linear_1 = nn.Linear(self.num_feature_sizes + self.cat_field_size*self.embedding_size, self.deep_layers[0])
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
            cat_size += self.deep_layers[-1]
            print("Init deep part succeed")

        self.last_layer = nn.Linear(cat_size, 1)
        print("Init succeed")

    def forward(self, Xi_num, Xv_num, Xi_cat, Xv_cat):
        """
        :param Xi_num: index input tensor, batch_size * k
        :param Xv_num: value input tensor, batch_size * k
        :return: the last output
        """

        if self.deep_layers_activation == 'sigmoid':
            activation = F.sigmoid
        elif self.deep_layers_activation == 'tanh':
            activation = F.tanh
        else:
            activation = F.relu

        """
            embeddings
        """
        # emb_arr = [(torch.sum(emb(Xi[:,i,:]),1).t()*Xv[:,i]).t() for i, emb in enumerate(self.embeddings)]
        emb_arr = self.embeddings(Xi_cat) * Xv_cat.view(-1, self.cat_field_size, 1)

        outputs = []

        """
            wide part
        """
        if self.use_wide:
            wide_emb_arr = self.wide_embeddings(Xi_cat) * Xv_cat.view(-1, self.cat_field_size, 1)
            wide_output = self.wide_linear(torch.cat([wide_emb_arr.view(-1, self.cat_field_size), Xv_num], 1))
            wide_output = activation(wide_output)
            wide_output = self.wide_dropout(wide_output)
            outputs.append(wide_output)

        """
            cross part
        """
        if self.use_cross:
            # x_0 = torch.cat(emb_arr,1)
            x_0 = torch.cat([emb_arr.view(-1, self.cat_field_size * self.embedding_size), Xv_num], 1)
            x_l = x_0
            for i in range(self.h_cross_depth):
                x_l = torch.sum(x_0 * x_l, 1).view([-1,1]) * getattr(self,'cross_weight_'+str(i+1)).view([1,-1]) + getattr(self,'cross_bias_'+str(i+1)) + x_l
            outputs.append(x_l)

        """
            inner product part
        """
        if self.use_inner_product:
            fm_wij_arr = []
            for i in range(self.cat_field_size):
                for j in range(i + 1, self.cat_field_size):
                    fm_wij_arr.append(torch.sum(emb_arr[:,i,:] * emb_arr[:,j,:],1).view([-1,1]))
            inner_output = torch.cat(fm_wij_arr, 1)

            # Memory cost is high
            # deep_emb = torch.matmul(emb_arr, emb_arr.permute(0, 2, 1))
            # inner_output = torch.masked_select(deep_emb, self.inner_product_mask).view(-1, self.inner_product_size)

            if self.is_inner_product_dropout:
                inner_output = self.inner_product_0_dropout(inner_output)
            x_deep = self.inner_product_linear_1(inner_output)
            if self.is_batch_norm:
                x_deep = self.inner_product_batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_inner_product_dropout:
                x_deep = self.inner_product_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'inner_product_linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'inner_product_batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'inner_product_' + str(i + 1) + '_dropout')(x_deep)
            outputs.append(x_deep)

        """
            deep part
        """
        if self.use_deep:
            # deep_emb = torch.cat(emb_arr,1)
            deep_emb = torch.cat([emb_arr.view(-1, self.cat_field_size * self.embedding_size), Xv_num], 1)
            if self.is_deep_dropout:
                deep_emb = self.linear_0_dropout(deep_emb)
            x_deep = self.linear_1(deep_emb)
            if self.is_batch_norm:
                x_deep = self.batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = self.linear_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)
            outputs.append(x_deep)

        """
            total
        """
        output = self.last_layer(torch.cat(outputs,1))
        return torch.sum(output,1)
