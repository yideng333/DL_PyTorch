import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from dotmap import DotMap
from torch.utils.data import DataLoader
from models.BiLSTM_CRF import BiLSTM_CRF
from models.BERT_CRF import BERT_CRF
import time
from utils.ner_train import train_model, test_model
import random
import pickle
from data.india_sms import prepare_sms_data, bert_prepare_sms_data
import nltk
from nltk.tokenize import WhitespaceTokenizer, TreebankWordTokenizer
from transformers import AlbertTokenizer
import os
# nltk.download('punkt')


def train():
    word_to_ix, tag_to_ix, train_data, train_label, test_data, test_label = prepare_sms_data()

    # train_data, train_label = shuffle_data(train_data, train_label)
    args = DotMap({'vocab_size': len(word_to_ix),
                   'tagset_size': len(tag_to_ix),
                   'embedding_dim': 100,
                   'hidden_dim': 30,
                   'use_basic_rnn': False,
                   'use_lstm': True,
                   'use_gru': False,
                   'num_layers': 1,
                   'bidirectional': False,
                   # 'random_seed': 666,
                   'batch_size': 32,
                   'wd': 0,
                   'device': 'cuda',  # cpu / cuda
                   'epochs': 100,
                   'lr': 0.001,  # learning_rate
                   'pred_period': 10,  # log intervel
                   'save_model': True,
                   })

    net = BiLSTM_CRF(args)
    model = net.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # for name, param in net.named_parameters():
    #     print(name, param.size())
    train_model(args, model, optimizer, train_data, train_label, test_data, test_label)


def test():
    with open('./datasets/word2idx.pk', 'rb') as f:
        word_to_ix = pickle.load(f)

    with open('./datasets/tag2idx.pk', 'rb') as f:
        tag_to_ix = pickle.load(f)
    print(tag_to_ix)

    args = DotMap({'vocab_size': len(word_to_ix),
                   'tagset_size': len(tag_to_ix),
                   'embedding_dim': 100,
                   'hidden_dim': 30,
                   'use_basic_rnn': False,
                   'use_lstm': True,
                   'use_gru': False,
                   'num_layers': 1,
                   'bidirectional': False,
                   # 'random_seed': 666,
                   'batch_size': 32,
                   'wd': 0,
                   'device': 'cpu',  # cpu / cuda
                   'epochs': 40,
                   'lr': 0.001,  # learning_rate
                   'pred_period': 10,  # log intervel
                   'save_model': True,
                   })

    net = BiLSTM_CRF(args)
    model = net.to(args.device)
    # test_data = ['An amount of Rs.2,000000 has been debited from your account number XXXX80005 on 19/12/32 08:00:00 for an online payment txn done using HDFC Bank NetBanking.']
    # test_data = ['Fund Transfer to Sarvendessr - XXXXXXXX04712 is successful. Rs.10400.00 is debited from your account XXXXXXXX16927']
    test_data = ['An Amount of 1521 INR has been debited to A/c no XXXXXXX108806 for APY Contribution on 31-SEP-18 13:42:56. Now Clear balance is Credit INR 48.52']
    print(WhitespaceTokenizer().tokenize(test_data[0]))
    sentences = [torch.LongTensor([word_to_ix.get(s.lower(), word_to_ix['UNK']) for s in WhitespaceTokenizer().tokenize(data)]) for data in test_data]
    print(sentences)

    predict_rlt = test_model(args, model, sentences)
    print(predict_rlt)

    ix_to_word = {}
    for i, v in word_to_ix.items():
        ix_to_word[v] = i

    ix_to_tag = {}
    for i, v in tag_to_ix.items():
        ix_to_tag[v] = i
    print(ix_to_tag)

    final = []
    for i, predict in enumerate(predict_rlt):
        trans = []
        for k, v in enumerate(predict):
            if v != tag_to_ix['other']:
                trans.append((ix_to_word[sentences[i].tolist()[k]], ix_to_tag[v]))
        final.append(trans)

    print(final)


def train_bert():
    bert_model_name = 'albert-base-v2'
    train_data, train_label, test_data, test_label = bert_prepare_sms_data(bert_model_name)

    # train_data, train_label = shuffle_data(train_data, train_label)
    args = DotMap({'bert_model_name': bert_model_name,
                   'tagset_size': 9,
                   'embedding_dim': 768,
                   'hidden_dim': 64,
                   'use_basic_rnn': False,
                   'use_lstm': True,
                   'use_gru': False,
                   'num_layers': 1,
                   'bidirectional': False,
                   # 'random_seed': 666,
                   'batch_size': 32,
                   'wd': 0,
                   'device': 'cuda',  # cpu / cuda
                   'epochs': 100,
                   'lr': 0.00001,  # learning_rate
                   'pred_period': 10,  # log intervel
                   'save_model': True,
                   'load_check_point': True,  # 装载上一次训练的模型，继续训练
                   'save_model_name': 'ner_model_1.pt'
                   })

    net = BERT_CRF(args)
    model = net.to(args.device)
    if args.load_check_point:
        model.load_state_dict(torch.load("./datasets/ner_model.pt", map_location=torch.device(args.device)))
        print('加载模型成功!')
    else:
        print('无保存模型，将从头开始训练！')

    # for name, params in net.named_parameters():
    #     print(name, params.size())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    train_model(args, model, optimizer, train_data, train_label, test_data, test_label)


def test_bert():
    bert_model_name = 'albert-base-v2'
    tokenizer = AlbertTokenizer.from_pretrained(bert_model_name)
    tag_to_ix = {'other': 0, '账户账号＿自己': 1, '账户账号＿他人': 2, '金额＿转入': 3, '日期＿交易时间': 4,
                 '银行卡号＿自己': 5, '金额＿转出': 6, '机构＿交易平台': 7, '金额＿余额': 8}
    print(tag_to_ix)

    args = DotMap({'bert_model_name': bert_model_name,
                   'tagset_size': 9,
                   'embedding_dim': 768,
                   'hidden_dim': 64,
                   'use_basic_rnn': False,
                   'use_lstm': True,
                   'use_gru': False,
                   'num_layers': 1,
                   'bidirectional': False,
                   # 'random_seed': 666,
                   'batch_size': 32,
                   'wd': 0,
                   'device': 'cuda',  # cpu / cuda
                   'epochs': 150,
                   'lr': 0.00001,  # learning_rate
                   'pred_period': 10,  # log intervel
                   'save_model': True,
                   'save_model_name': 'ner_model.pt'
                   })

    net = BERT_CRF(args)
    model = net.to(args.device)
    # test_data = 'Dear customer, An amount of Rs.2,000000 has been debited from your account number XXXX8005 on 19/12/32 08:00:00  for an online payment txn done using HSFC Bank NetBanking.'
    # test_data = 'Fund Transfer to Sarvendessr - XXXXXXXX04712 is successful. Rs.10400.00 is debited from your account XXXXXXXX16927'
    # test_data = 'An Amount of INR 15221 has been debited to A/c no XXXXXXX10386 for APY Contribution on 31-SEP-19 13:42:56. Now Clear balance is Credit INR 48.52'
    test_data = 'Thank you for paying Rs.1,30.00 from A/c XXXX88505 to PAYTMWALLLOADING via NetBanking.'
    test_data = '[CLS] ' + test_data + ' [SEP]'
    print(test_data.split(' '))
    print(len(test_data))
    sentences = [torch.LongTensor(tokenizer.convert_tokens_to_ids(list(map(lambda x: x.lower() if x not in ['[CLS]', '[SEP]'] else x, test_data.split(' ')))))]
    print(sentences)

    predict_rlt = test_model(args, model, sentences)
    print(predict_rlt)

    # ix_to_word = {}
    # for i, v in word_to_ix.items():
    #     ix_to_word[v] = i
    #
    ix_to_tag = {}
    for i, v in tag_to_ix.items():
        ix_to_tag[v] = i
    print(ix_to_tag)

    final = []
    for i, predict in enumerate(predict_rlt):
        trans = []
        for k, v in enumerate(predict):
            if v != tag_to_ix['other']:
                trans.append((sentences[i][k], test_data.split(' ')[k], ix_to_tag[v]))
        final.append(trans)

    print(final)
