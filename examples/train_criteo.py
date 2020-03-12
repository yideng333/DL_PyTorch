import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from dotmap import DotMap
from models.DeepFM import DeepFM
from data.criteo import load_criteo_data
from utils.ctr_train import train_model
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score
import os


def train(data_dir):
    # 读入数据集
    feature_sizes = np.load(os.path.join(data_dir, 'sample_feature_sizes.npy'))

    batch_size = 1024
    start = time.time()
    train_loader, test_loader = load_criteo_data(data_dir, batch_size)
    print('finish loading data, time={}'.format(time.time()-start))

    args = DotMap({'field_size': 39,
                   'feature_sizes': sum(feature_sizes),
                   'embedding_size': 64,

                   'use_lr': False,
                   'use_fm': True,
                   'is_shallow_dropout': False,
                   'dropout_shallow': [0.5, 0.5],

                   'use_deep': True,
                   'deep_layers': [512, 128, 32],
                   'is_deep_dropout': True,
                   'dropout_deep': [0, 0.5, 0.5, 0.5],
                   'is_batch_norm': True,

                   # 'random_seed': 666,
                   'batch_size': batch_size,
                   'wd': 0,
                   'device': 'cuda',  # cpu / cuda
                   'epochs': 20,
                   'lr': 0.0001,  # learning_rate
                   'log_interval': 100,  # log intervel
                   'save_model': False,
                   'eval_metric': roc_auc_score
                   })
    print(args)
    net = DeepFM(args)
    # print(net)

    # for blk in net.children():
    #     X = blk(X)
    #     print('output shape: ', X.shape)

    # 以均值为0，方差0.01初始化参数
    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)
    # for name, param in net.named_parameters():
    #     print(name, param)

    print("training on ", args.device)
    model = net.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # loss = torch.nn.CrossEntropyLoss()
    loss = F.binary_cross_entropy_with_logits

    train_model(args, model, loss, train_loader, test_loader, optimizer)
