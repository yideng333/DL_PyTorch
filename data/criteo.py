import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import math
import numpy as np
from collections import Counter


cat_features = ['C' + str(i) for i in range(1, 27)]
num_features = ['I' + str(i) for i in range(1, 14)]
columns = ['label'] + num_features + cat_features


def process_raw_data(data_dir):
    # 读取原始数据
    data = pd.read_csv(os.path.join(data_dir, './sample.txt'), names=columns, encoding='utf-8', sep='\t')
    print(data.shape)

    # 填充NaN的值
    data[cat_features] = data[cat_features].fillna('-1', )
    data[num_features] = data[num_features].fillna(0, )

    print(data.shape)
    print(data.head())

    print('start label encoder')
    # 类别型的特征做label encoder
    for feat in cat_features:
        print(feat)
        feat_list = data[feat].tolist()
        feat_counter = Counter(feat_list)
        # 出现次数少于10的值给一个特定值
        data[feat] = data[feat].apply(lambda v: v if feat_counter[v] >= 10 else 'LESS')

        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    print('start num normalization')
    # 数值型的特征做normalization
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])
    for feat in num_features:
        print(feat)
        data[feat] = data[feat].apply(lambda v: math.log(v, 2) if v > 2 else v)
    print(data.head())

    print('get feature sizes')
    # 得到每个特征的个数
    num_features_columns = [1 for feat in num_features]
    cat_features_columns = [data[feat].nunique() for feat in cat_features]
    feature_sizes = num_features_columns + cat_features_columns
    print(feature_sizes)
    print(sum(feature_sizes))

    print('save data files')
    np.save(os.path.join(data_dir, 'sample_feature_sizes.npy'), feature_sizes)
    np.save(os.path.join(data_dir, 'sample_data_values.npy'), data.values)

    # data.to_csv(os.path.join(data_dir, './criteo_example.csv'), index=False, encoding='utf-8')


def prepare_train_test_data(data_dir):
    feature_sizes = np.load(os.path.join(data_dir, 'sample_feature_sizes.npy'))
    data = np.load(os.path.join(data_dir, 'sample_data_values.npy'))
    print(feature_sizes)
    print(sum(feature_sizes))
    print(data.shape)

    train, test = train_test_split(data, test_size=0.2, random_state=66)
    print(train.shape)
    print(test.shape)

    print('start get data dict')
    train_label, train_dict = get_data_dict(train, feature_sizes)
    # print(train_dict)
    test_label, test_dict = get_data_dict(test, feature_sizes)
    # print(test_dict)

    print('train_pos_ratio:', sum(train_label) / len(train_label))
    print('test_pos_ratio:', sum(test_label) / len(test_label))

    np.save(os.path.join(data_dir, 'sample_train_label.npy'), train_label)
    np.save(os.path.join(data_dir, 'sample_train_index.npy'), train_dict['index'])
    np.save(os.path.join(data_dir, 'sample_train_value.npy'), train_dict['value'])

    np.save(os.path.join(data_dir, 'sample_test_label.npy'), test_label)
    np.save(os.path.join(data_dir, 'sample_test_index.npy'), test_dict['index'])
    np.save(os.path.join(data_dir, 'sample_test_value.npy'), test_dict['value'])


# 将特征转换成成全局唯一的index和value
def get_data_dict(data, feature_sizes):
    # data_values = data.values.tolist()
    label = data[:, 0]
    result = {'index': [], 'value': []}

    for i in range(data.shape[0]):
        # print(i)
        if i % 10000 == 0:
            print(i)
        num_indexs = [k for k in range(len(num_features))]
        cat_indexs = [int(sum(feature_sizes[:k]) + data[i][k+1]) for k in range(len(num_features), len(num_features) + len(cat_features))]

        num_values = list(data[i, 1:len(num_features) + 1])
        cat_values = [1.0 for i in range(len(cat_features))]

        result['index'].append(num_indexs + cat_indexs)
        result['value'].append(num_values + cat_values)
    # print(data[0, :])
    # print(result['index'][0])
    # print(len(result['index'][0]))
    # print(result['value'][0])
    # print(len(result['value'][0]))
    return label, result


def load_criteo_data(data_dir, batch_size):
    train_label = np.load(os.path.join(data_dir, 'sample_train_label.npy'))
    train_index = np.load(os.path.join(data_dir, 'sample_train_index.npy'))
    train_value = np.load(os.path.join(data_dir, 'sample_train_value.npy'))

    test_label = np.load(os.path.join(data_dir, 'sample_test_label.npy'))
    test_index = np.load(os.path.join(data_dir, 'sample_test_index.npy'))
    test_value = np.load(os.path.join(data_dir, 'sample_test_value.npy'))

    train_data = TensorDataset(torch.FloatTensor(train_label), torch.LongTensor(train_index),
                               torch.FloatTensor(train_value))
    print(len(train_data))
    test_data = TensorDataset(torch.FloatTensor(test_label), torch.LongTensor(test_index),
                              torch.FloatTensor(test_value))
    print(len(test_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=16)

    return train_loader, test_loader