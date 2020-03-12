import torch
import os
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import re
import json
import nltk
from nltk.tokenize import WhitespaceTokenizer, TreebankWordTokenizer
# nltk.download('punkt')
from transformers import BertTokenizer, AlbertTokenizer


def prepare_toy_data():
    # Make up some training data
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag_to_ix = {"B": 0, "I": 1, "O": 2}
    # tag_to_ix = {"PAD": 0, "B": 1, "I": 2, "O": 3}

    # 4 sentences
    training_data = ["a university in georgia georgia tech is".split(),
                     "the wall street journal reported today that apple corporation made money".split(),
                     "that apple corporation made money the wall street journal reported today".split(),
                     "georgia tech is a university in georgia".split()
                     ]
    training_label = ["O O O B B I O".split(),
                      "B I I I O O O B I O O".split(),
                      "O B I O O B I I I O O".split(),
                      "B I O O O O B".split()
                      ]

    print(training_data)
    print(training_label)

    word_to_ix = {'PAD': 0}
    for sentence in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print(word_to_ix)
    train_data = [torch.LongTensor([word_to_ix[w] for w in seq]) for seq in training_data]
    train_label = [torch.LongTensor([tag_to_ix[w] for w in target]) for target in training_label]

    print(train_data)
    print(train_label)

    return train_data, train_label, word_to_ix, tag_to_ix


# 从原始数据中获取短信和标注数据
def get_india_sms_data(data_dir):
    data_all = []
    for idx, fn in enumerate(os.listdir(data_dir)):
        # print(fn)
        despatcher = fn.split('.')[0]
        # print(despatcher)
        with open(os.path.join(data_dir, fn), 'rb') as fin:
            tasks = pickle.load(fin)
            # print(len(tasks))

        group_num = 0
        for key, group in itertools.groupby(tasks, key=lambda x: x.output[0]):
            regex_id = key
            sms_type = None
            fields = []
            field_types = []
            text = None
            spans = None
            for t in list(group):
                # print('----', t.value)
                output = t.output
                val = t.value
                regex_id = output[0]
                group_id = output[1]
                gened_regex = output[2]
                box_markups = output[3]
                # print(output)
                # print(val)
                # print('*' * 50)
                if group_id == 0:
                    sms_type = val
                    # if sms_type:
                    #     print(regex_id, gened_regex)
                    #     print('短信类型: ', sms_type)
                    text = box_markups[0][0].text
                    print('text :', text)
                    spans = box_markups[0][0].spans
                    print('spans:', spans)
                    for s in spans:
                        fields.append(text[s.start: s.stop])
                else:
                    field_types.append(val)
            # print(field_types)
            # if None not in field_types:
            field_dict = {}
            for i, j in zip(fields, field_types):
                field_dict[i] = j
            # new_text = 'Reg_id_' + str(regex_id) + ' ' + text
            #             print(cluster_id)
            # print(text)
            # text = re.sub(r'\(|\)', ' ', text)
            print(text)
            print(fields)
            print(field_types)
            print(field_dict)
            data_all.append([text, spans, json.dumps(fields), json.dumps(field_types), json.dumps(field_dict), sms_type])
            group_num += 1
        print(group_num)
        print('------------------------------------------------------')

    print(len(data_all))
    # print(ALL)

    df_data = pd.DataFrame(data_all, columns=['text', 'spans', 'fields', 'field_types', 'field_dict', 'sms_type'])
    print(df_data)
    # df_data.to_csv('./datasets/raw_sms_df.csv', index=False, encoding='utf-8')


# 进行分词
def tokenize_sms_data():
    df_sms = pd.read_csv('./datasets/raw_sms_df.csv', header=0, encoding='utf-8')
    print(df_sms.shape)

    print(df_sms.groupby('sms_type')['text'].count())

    # 过滤短信类型
    # df_sms = df_sms[~df_sms.sms_type.isin(['False', 'sms_other', '贷前申请＿审核拒绝', '贷前申请＿审核通过',
    #                                       '贷前申请＿申请交互', '贷后提醒＿到期提醒', '贷后提醒＿成功放款', '贷后提醒＿逾期催收',
    #                                        '账号异常＿信用额度不足', '账户账号＿自己', '信用卡＿申请失败'])]
    df_sms = df_sms[df_sms.sms_type.isin(['交易流水＿转账'])]

    print(df_sms.shape)
    print(df_sms.head())

    # 去重
    # df_sms = df_sms.drop_duplicates(['text'], keep='last')
    # print(df_sms.shape)

    text_all = df_sms.text.tolist()
    field_dict = df_sms.field_dict.tolist()
    # print(field_dict)

    text_tokenized = []
    field_tokenized = []

    for i, text in enumerate(text_all):
        print(i, text)
        # print(nltk.word_tokenize(text))
        # 这里使用nltk的工具进行分词
        t = WhitespaceTokenizer().tokenize(text)
        print(t)

        fields = json.loads(field_dict[i])
        print(fields)
        new_filed = {}
        # 标注也要进行相应的分词
        for k, f in fields.items():
            if f in [False, None, '账号＿借款编号', '金额＿信用额度', '金额＿应还金额']:
                    # '金额＿薪资', '天数＿逾期天数', '账号＿借款编号', '金额＿逾期罚金', '交易流水＿转账'
                f = 'other'
            if f == '金额＿薪资':
                f = '金额＿转出'
            if f == '日期＿还款日期':
                f = '日期＿交易时间'
            x = WhitespaceTokenizer().tokenize(k)
            for v in x:
                new_filed[v] = f

        print(new_filed)
        tag_set = list(set([i for i in new_filed.values()]))
        if tag_set == ['other']:
            print('get all other tags, ignore!!')
        else:
            text_tokenized.append(json.dumps(t))
            field_tokenized.append(json.dumps(new_filed))

    df_sms_tokenized = pd.DataFrame()
    df_sms_tokenized['text_tokenized'] = text_tokenized
    df_sms_tokenized['field_tokenized'] = field_tokenized
    # df_sms_tokenized.to_csv('./datasets/tokenized_sms_df.csv', index=False, encoding='utf-8')


def prepare_sms_data(exist_idx=True):
    df_sms = pd.read_csv('./datasets/tokenized_sms_df.csv', header=0, encoding='utf-8')
    print(df_sms.shape)
    # print(df_sms.head())

    text_tokenized = df_sms.text_tokenized.tolist()
    field_tokenized = df_sms.field_tokenized.tolist()

    if not exist_idx:
        # word字典
        words = list(set([k for i in text_tokenized for k in json.loads(i)]))
        print(len(words))
        # 转换成lower case
        words = list(set([i.lower() for i in words]))
        print(len(words))
        word2idx = {w: i + 2 for i, w in enumerate(words)}
        word2idx['UNK'] = 1
        word2idx['PAD'] = 0
        print(len(word2idx))
        # print(word2idx)

        # tag字典
        tags = list(set([v for i in field_tokenized for k, v in json.loads(i).items()]))
        print(tags)
        tag2idx = {t: i for i, t in enumerate(tags)}
        print(len(tag2idx))
        print(tag2idx)

        with open('./datasets/word2idx.pk', 'wb') as f:
            pickle.dump(word2idx, f)
        with open('./datasets/tag2idx.pk', 'wb') as f:
            pickle.dump(tag2idx, f)
    else:
        print('load word2idx and tag2idx from files!')
        with open('./datasets/word2idx.pk', 'rb') as f:
            word2idx = pickle.load(f)
        print(len(word2idx))
        with open('./datasets/tag2idx.pk', 'rb') as f:
            tag2idx = pickle.load(f)
        print(tag2idx)

    # sentence进行映射
    # sentences_X = [torch.LongTensor([word2idx.get(w[0], word2idx['UNK']) for w in s]) for s in sentences]
    sentences_X = [torch.LongTensor([word2idx.get(w.lower(), word2idx['UNK']) for w in json.loads(s)]) for s in text_tokenized]
    # print(sentences_X[-2:])
    print(len(sentences_X))

    # tag进行映射
    targets = []
    for i, text in enumerate(text_tokenized):
        target = []
        tags_dict = json.loads(field_tokenized[i])
        sentence = json.loads(text)
        for s in sentence:
            if tags_dict.get(s):
                target.append(tags_dict.get(s))
            else:
                target.append('other')
        targets.append(target)
    # print(targets[:2])

    targets_X = [torch.LongTensor([tag2idx.get(t) for t in s]) for s in targets]
    print(targets_X[:2])
    print(len(targets_X))

    train_data, test_data = train_test_split(sentences_X, test_size=0.1, random_state=66)
    print(len(train_data))
    print(len(test_data))

    train_label, test_label = train_test_split(targets_X, test_size=0.1, random_state=66)
    print(len(train_label))
    print(len(test_label))

    # print(train_data[:2])

    return word2idx, tag2idx, train_data, train_label, test_data, test_label


# 使用bert_tokenizer进行分词
def bert_tokenize_sms_data():
    # Load pre-trained model tokenizer (vocabulary)

    df_sms = pd.read_csv('./datasets/raw_sms_df.csv', header=0, encoding='utf-8')
    print(df_sms.shape)

    print(df_sms.groupby('sms_type')['text'].count())

    # 过滤短信类型
    # df_sms = df_sms[~df_sms.sms_type.isin(['False', 'sms_other', '贷前申请＿审核拒绝', '贷前申请＿审核通过',
    #                                       '贷前申请＿申请交互', '贷后提醒＿到期提醒', '贷后提醒＿成功放款', '贷后提醒＿逾期催收',
    #                                        '账号异常＿信用额度不足', '账户账号＿自己', '信用卡＿申请失败'])]
    df_sms = df_sms[df_sms.sms_type.isin(['交易流水＿转账'])]

    print(df_sms.shape)
    print(df_sms.head())

    # 去重
    # df_sms = df_sms.drop_duplicates(['text'], keep='last')
    # print(df_sms.shape)

    text_all = df_sms.text.tolist()
    field_dict = df_sms.field_dict.tolist()
    # print(field_dict)

    text_tokenized = []
    field_tokenized = []

    for i, text in enumerate(text_all):
        print(i, text)

        text = '[CLS] ' + text + ' [SEP]'
        print(text)

        temp_text = text.split(' ')
        print(temp_text)

        fields = json.loads(field_dict[i])
        print(fields)
        temp_filed = {}
        # 标注也要进行相应的分词

        for k, f in fields.items():
            if f in [False, None, '账号＿借款编号', '金额＿信用额度', '金额＿应还金额']:
                    # '金额＿薪资', '天数＿逾期天数', '账号＿借款编号', '金额＿逾期罚金', '交易流水＿转账'
                f = 'other'
            if f == '金额＿薪资':
                f = '金额＿转出'
            if f == '日期＿还款日期':
                f = '日期＿交易时间'
            x = k.split(' ')
            for v in x:
                temp_filed[v] = f

        print(temp_filed)

        new_field = []
        for t in temp_text:
            if t in temp_filed:
                new_field.append(temp_filed[t])
            elif t in ['[CLS]', '[SEP]']:
                new_field.append('<PAD>')
            else:
                new_field.append('other')

        print(new_field)
        assert len(new_field) == len(temp_text)

        tag_set = list(set(new_field))
        if tag_set == ['other']:
            print('get all other tags, ignore!!')
        else:
            text_tokenized.append(json.dumps(temp_text))
            field_tokenized.append(json.dumps(new_field))

    df_sms_tokenized = pd.DataFrame()
    df_sms_tokenized['text_tokenized'] = text_tokenized
    df_sms_tokenized['field_tokenized'] = field_tokenized
    df_sms_tokenized.to_csv('./datasets/tokenized_sms_df_new.csv', index=False, encoding='utf-8')


def bert_prepare_sms_data(bert_model_name):
    if bert_model_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        print('load tokenizer of {}'.format(bert_model_name))
    elif bert_model_name.startswith('albert-'):
        tokenizer = AlbertTokenizer.from_pretrained(bert_model_name)
        print('load tokenizer of {}'.format(bert_model_name))
    else:
        print('{} not found!!!'.format(bert_model_name))

    df_sms = pd.read_csv('./datasets/tokenized_sms_df.csv', header=0, encoding='utf-8')
    print(df_sms.shape)
    # print(df_sms.head())

    text_tokenized = df_sms.text_tokenized.tolist()
    field_tokenized = df_sms.field_tokenized.tolist()

    # tag字典
    # tags = list(set([v for i in field_tokenized for v in json.loads(i)]))
    # print(tags)
    # tag2idx = {t: i for i, t in enumerate(tags)}
    # print(len(tag2idx))
    # print(tag2idx)
    tag2idx = {'other': 0, '账户账号＿自己': 1, '账户账号＿他人': 2, '金额＿转入': 3, '日期＿交易时间': 4,
               '银行卡号＿自己': 5, '金额＿转出': 6, '机构＿交易平台': 7, '金额＿余额': 8}

    # sentence进行映射
    # sentences_X = [torch.LongTensor([word2idx.get(w[0], word2idx['UNK']) for w in s]) for s in sentences]
    sentences_X = [torch.LongTensor(tokenizer.convert_tokens_to_ids(list(map(lambda x:  x.lower() if x not in ['[CLS]', '[SEP]'] else x, json.loads(s))))) for s in text_tokenized]
    # print(text_tokenized[:2])
    # print(sentences_X[:2])
    print(len(sentences_X))

    targets_X = [torch.LongTensor([tag2idx.get(t, 0) for t in json.loads(s)]) for s in field_tokenized]
    # print(field_tokenized[:2])
    # print(targets_X[:2])
    print(len(targets_X))

    train_data, test_data = train_test_split(sentences_X, test_size=0.1, random_state=11)
    print(len(train_data))
    print(len(test_data))

    train_label, test_label = train_test_split(targets_X, test_size=0.1, random_state=11)
    print(len(train_label))
    print(len(test_label))

    # print(train_data[:2])

    return train_data, train_label, test_data, test_label