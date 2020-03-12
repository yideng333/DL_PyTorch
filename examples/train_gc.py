import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from dotmap import DotMap
# from models.DeepFM import DeepFM
from models.DeepCrossNetwork import DCN
from data.gc_l0 import load_gc_data, load_gc_data_leaf, load_gc_data_combined, load_gc_data_concated
from utils.ctr_train import train_model
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score

data_dir = '/data-0/yideng/gc_l0_fq3_20190513/'


def train():
    # 读入数据集
    batch_size = 1024
    start = time.time()
    train_loader, test_loader = load_gc_data_concated(data_dir, batch_size=batch_size)
    print('finish loading data, time={}'.format(time.time()-start))

    # features_sizes = 2000 + 73785
    # features_sizes = 2000
    cat_field_size = 4919
    cat_feature_sizes = 73785
    num_feature_sizes = 2000

    args = DotMap({'cat_field_size': cat_field_size,
                   'cat_feature_sizes': cat_feature_sizes,
                   'num_feature_sizes': num_feature_sizes,
                   'embedding_size': 3,
                   'is_shallow_dropout': False,
                   'dropout_shallow': [0.5, 0.5],
                   'use_cross': True,
                   'h_cross_depth': 2,
                   # 'use_fm': True,
                   # 'use_ffm': False,
                   'use_deep': True,
                   'deep_layers': [128, 32],
                   'is_deep_dropout': True,
                   'dropout_deep': [0.5, 0.5, 0.5, 0.5],
                   'is_batch_norm': True,

                   'use_inner_product': False,
                   'is_inner_product_dropout': True,
                   'dropout_inner_product_deep': [0.5, 0.5, 0.5],
                   'inner_product_layers': [128, 32],

                   'use_wide': True,
                   'wide_output_size': 10,

                   # 'random_seed': 666,
                   'batch_size': batch_size,
                   'wd': 0.001,
                   'device': 'cuda',  # cpu / cuda
                   'epochs': 30,
                   'lr': 0.001,  # learning_rate
                   'log_interval': 10,  # log intervel
                   'save_model': False,
                   'eval_metric': roc_auc_score
                   })

    # net = DeepFM(args)
    net = DCN(args)
    # print(net)

    # for blk in net.children():
    #     X = blk(X)
    #     print('output shape: ', X.shape)

    # 自定义初始化参数
    for params in net.parameters():
        init.normal_(params, mean=0, std=0.001)
    # for name, param in net.named_parameters():
    #     print(name, param)

    print("training on ", args.device)
    model = net.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # loss = torch.nn.CrossEntropyLoss()
    loss = F.binary_cross_entropy_with_logits

    train_model(args, model, loss, train_loader, test_loader, optimizer)

