import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from dotmap import DotMap
from models.RNN import RNNModel
from data.jaychou_lyrics import load_data_jay_lyrics, data_iter_consecutive, to_onehot
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score
import os
import math


def train(data_dir):
    # 读入数据集
    corpus_indices, idx_to_char, char_to_idx, vocab_size = load_data_jay_lyrics(data_dir)

    args = DotMap({'use_basic_rnn': False,
                   'use_lstm': True,
                   'use_gru': False,
                   'num_hiddens': 256,
                   'vocab_size': vocab_size,
                   'num_layers': 2,
                   'bidirectional': False,
                   'num_steps': 30,
                   # 'random_seed': 666,
                   'batch_size': 64,
                   'wd': 0,
                   'device': 'cuda',  # cpu / cuda
                   'epochs': 200,
                   'lr': 0.01,  # learning_rate
                   'pred_period': 10,  # log intervel
                   'save_model': False,
                   'eval_metric': roc_auc_score
                   })

    prefixes = ['我', '不', '爱']

    net = RNNModel(args)

    loss = nn.CrossEntropyLoss()
    model = net.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    state = None
    for epoch in range(args.epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, args.batch_size, args.num_steps)  # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            # 转换成one-hot向量
            X = torch.stack(to_onehot(X, vocab_size))
            X, Y = X.to(args.device), Y.to(args.device)
            output, state = model(X, state)  # output: 形状为(num_steps * batch_size, vocab_size)

            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            # grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')

        if (epoch + 1) % args.pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(prefix, 50, net, vocab_size, idx_to_char, char_to_idx, args))

    # print(predict_rnn_pytorch('分开', 20, net, vocab_size, idx_to_char, char_to_idx))


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, idx_to_char, char_to_idx, args):
    state = None
    output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]]).view(1, 1)
        X = torch.stack(to_onehot(X, vocab_size))
        X = X.to(args.device)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0], state[1])
            else:
                state = state

        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])
