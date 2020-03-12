import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn import preprocessing
import numpy as np
import os


def read_gc_data(data):
    print(data[0].shape)
    print(len(data[1]))

    result = {'index': [], 'value': []}
    label = data[1]
    result['value'] = data[0].toarray()

    for i in range(len(data[1])):
        indexs = [0 for item in range(2000)]
        result['index'].append(indexs)

    return label, result


def load_gc_data(data_dir, batch_size):
    # train_data = load_svmlight_file(os.path.join(data_dir, 'train_top_2K_normal.libsvm'), dtype=np.float64)
    # test_data = load_svmlight_file(os.path.join(data_dir, 'test_top_2K_normal.libsvm'), dtype=np.float64)

    # train_label, train_dict = read_gc_data_new(os.path.join(data_dir, 'train_top_2K_normal.libsvm'))
    # print(train_dict)
    # np.save(os.path.join(data_dir, 'train_top_2K_normal_index.npy'), train_dict['index'])
    # np.save(os.path.join(data_dir, 'train_top_2K_normal_value.npy'), train_dict['value'])
    train_index = np.load(os.path.join(data_dir, 'train_top_2K_normal_index.npy'))
    train_value = np.load(os.path.join(data_dir, 'train_top_2K_normal_value.npy'))
    train_label = np.load(os.path.join(data_dir, 'train_top_2K_gbdt_label.npy'))

    # test_label, test_dict = read_gc_data_new(os.path.join(data_dir, 'test_top_2K_normal.libsvm'))
    # print(test_dict)
    # np.save(os.path.join(data_dir, 'test_top_2K_normal_index.npy'), test_dict['index'])
    # np.save(os.path.join(data_dir, 'test_top_2K_normal_value.npy'), test_dict['value'])
    test_index = np.load(os.path.join(data_dir, 'test_top_2K_normal_index.npy'))
    test_value = np.load(os.path.join(data_dir, 'test_top_2K_normal_value.npy'))
    test_label = np.load(os.path.join(data_dir, 'test_top_2K_gbdt_label.npy'))

    print(sum(train_label) / len(train_label))
    print(sum(test_label) / len(test_label))

    train_data = TensorDataset(torch.FloatTensor(train_label), torch.LongTensor(train_index),
                               torch.FloatTensor(train_value))
    print(len(train_data))
    test_data = TensorDataset(torch.FloatTensor(test_label), torch.LongTensor(test_index),
                              torch.FloatTensor(test_value))
    print(len(test_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)

    return train_loader, test_loader


def read_gc_data_leaf(data, label):
    print(data.shape)
    print(len(label))
    result = {'label': [], 'index': [], 'value': []}
    result['label'] = label
    result['index'] = data

    for i in range(len(label)):
        data = [1 for item in range(4919)]
        result['value'].append(data)

    return result


def read_gc_data_leaf_new(file_path):
    label = []
    result = {'index': [], 'value': []}
    f = open(file_path, 'r')
    for line in f:
        datas = line.strip().split(' ')
        label.append(int(datas[0]))

        indexs = []
        values = []
        for k, item in enumerate(datas[1:]):
            index, value = item.split(':')
            indexs.append(int(index))
            values.append(float(value))

        result['index'].append(indexs)
        result['value'].append(values)
    return label, result


def load_gc_data_leaf(data_dir, batch_size):
    # train_data = np.load(os.path.join(data_dir, 'train_top_2K_gbdt_leaf_pred.npy'))
    # train_label = np.load(os.path.join(data_dir, 'train_top_2K_gbdt_label.npy'))
    #
    # test_data = np.load(os.path.join(data_dir, 'test_top_2K_gbdt_leaf_pred.npy'))
    # test_label = np.load(os.path.join(data_dir, 'test_top_2K_gbdt_label.npy'))

    # train_dict = read_gc_data_leaf(train_data, train_label)
    # # print(train_dict)
    # test_dict = read_gc_data_leaf(test_data, test_label)
    # # print(test_dict)

    # train_label, train_dict = read_gc_data_leaf_new(os.path.join(data_dir, 'train_top_2K_gbdt.libsvm'))
    # np.save(os.path.join(data_dir, 'train_top_2K_gbdt_index.npy'), train_dict['index'])
    # np.save(os.path.join(data_dir, 'train_top_2K_gbdt_value.npy'), train_dict['value'])
    train_index = np.load(os.path.join(data_dir, 'train_top_2K_gbdt_index.npy'))
    train_value = np.load(os.path.join(data_dir, 'train_top_2K_gbdt_value.npy'))
    train_label = np.load(os.path.join(data_dir, 'train_top_2K_gbdt_label.npy'))
    # print(train_dict['index'][:10])
    # test_label, test_dict = read_gc_data_leaf_new(os.path.join(data_dir, 'test_top_2K_gbdt.libsvm'))
    # np.save(os.path.join(data_dir, 'test_top_2K_gbdt_index.npy'), test_dict['index'])
    # np.save(os.path.join(data_dir, 'test_top_2K_gbdt_value.npy'), test_dict['value'])
    test_index = np.load(os.path.join(data_dir, 'test_top_2K_gbdt_index.npy'))
    test_value = np.load(os.path.join(data_dir, 'test_top_2K_gbdt_value.npy'))
    test_label = np.load(os.path.join(data_dir, 'test_top_2K_gbdt_label.npy'))
    # print(test_dict)

    print(sum(train_label) / len(train_label))
    print(sum(test_label) / len(test_label))

    train_data = TensorDataset(torch.FloatTensor(train_label), torch.LongTensor(train_index),
                               torch.FloatTensor(train_value))
    print(len(train_data))
    test_data = TensorDataset(torch.FloatTensor(test_label), torch.LongTensor(test_index),
                              torch.FloatTensor(test_value))
    print(len(test_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)

    return train_loader, test_loader


def read_gc_data_new(file_path):
    label = []
    result = {'index': [], 'value': []}
    f = open(file_path, 'r')
    for line in f:
        datas = line.strip().split(' ')
        label.append(int(datas[0]))

        indexs = []
        values = []
        for k, item in enumerate(datas[1:]):
            index, value = item.split(':')
            indexs.append(int(index))
            values.append(round(float(value), 4))

        result['index'].append(indexs)
        result['value'].append(values)
    return label, result


def load_gc_data_combined(data_dir, batch_size):
    # prepare data
    # train_label_normal, train_dict_normal = read_gc_data_new(os.path.join(data_dir, 'train_top_2K_normal.libsvm'))
    # test_label_normal, test_dict_normal = read_gc_data_new(os.path.join(data_dir, 'test_top_2K_normal.libsvm'))
    #
    # train_label_leaf, train_dict_leaf = read_gc_data_leaf_new(os.path.join(data_dir, 'train_top_2K_gbdt.libsvm'))
    # test_label_leaf, test_dict_leaf = read_gc_data_leaf_new(os.path.join(data_dir, 'test_top_2K_gbdt.libsvm'))
    #
    # train_label = train_label_normal
    # test_label = test_label_normal
    #
    # print('finish loading data')
    # train_index = []
    # train_value = []
    # for i in range(len(train_label)):
    #     train_index.append(train_dict_normal['index'][i] + [i+2000 for i in train_dict_leaf['index'][i]])
    #     train_value.append(train_dict_normal['value'][i] + train_dict_leaf['value'][i])
    #
    # test_index = []
    # test_value = []
    # for i in range(len(test_label)):
    #     test_index.append(test_dict_normal['index'][i] + [i+2000 for i in test_dict_leaf['index'][i]])
    #     test_value.append(test_dict_normal['value'][i] + test_dict_leaf['value'][i])
    #
    # np.save(os.path.join(data_dir, 'train_top_2K_combined_index.npy'), train_index)
    # np.save(os.path.join(data_dir, 'train_top_2K_combined_value.npy'), train_value)
    # np.save(os.path.join(data_dir, 'test_top_2K_combined_index.npy'), test_index)
    # np.save(os.path.join(data_dir, 'test_top_2K_combined_value.npy'), test_value)

    # load data
    train_index = np.load(os.path.join(data_dir, 'train_top_2K_combined_index.npy'))
    train_value = np.load(os.path.join(data_dir, 'train_top_2K_combined_value.npy'))
    train_label = np.load(os.path.join(data_dir, 'train_top_2K_gbdt_label.npy'))

    test_index = np.load(os.path.join(data_dir, 'test_top_2K_combined_index.npy'))
    test_value = np.load(os.path.join(data_dir, 'test_top_2K_combined_value.npy'))
    test_label = np.load(os.path.join(data_dir, 'test_top_2K_gbdt_label.npy'))

    # print(train_index[0])
    # print(train_value[0])

    print(len(train_index[0]), len(train_value[0]))
    print(len(test_index[0]), len(test_value[0]))

    print(sum(train_label) / len(train_label))
    print(sum(test_label) / len(test_label))

    train_data = TensorDataset(torch.FloatTensor(train_label), torch.LongTensor(train_index),
                               torch.FloatTensor(train_value))
    print(len(train_data))
    test_data = TensorDataset(torch.FloatTensor(test_label), torch.LongTensor(test_index),
                              torch.FloatTensor(test_value))
    print(len(test_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)

    return train_loader, test_loader


def load_gc_data_concated(data_dir, batch_size):
    # load data
    train_index_num = np.load(os.path.join(data_dir, 'train_top_2K_normal_index.npy'))
    train_value_num = np.load(os.path.join(data_dir, 'train_top_2K_normal_value.npy'))
    train_label = np.load(os.path.join(data_dir, 'train_top_2K_gbdt_label.npy'))

    test_index_num = np.load(os.path.join(data_dir, 'test_top_2K_normal_index.npy'))
    test_value_num = np.load(os.path.join(data_dir, 'test_top_2K_normal_value.npy'))
    test_label = np.load(os.path.join(data_dir, 'test_top_2K_gbdt_label.npy'))

    train_index_cat = np.load(os.path.join(data_dir, 'train_top_2K_gbdt_index.npy'))
    train_value_cat = np.load(os.path.join(data_dir, 'train_top_2K_gbdt_value.npy'))

    test_index_cat = np.load(os.path.join(data_dir, 'test_top_2K_gbdt_index.npy'))
    test_value_cat = np.load(os.path.join(data_dir, 'test_top_2K_gbdt_value.npy'))

    print(train_index_num[0], test_value_num[0])
    print(train_index_cat[0], train_value_cat[0])

    print(sum(train_label) / len(train_label))
    print(sum(test_label) / len(test_label))

    train_data = TensorDataset(torch.FloatTensor(train_label), torch.LongTensor(train_index_num),
                               torch.FloatTensor(train_value_num), torch.LongTensor(train_index_cat),
                               torch.FloatTensor(train_value_cat))
    print(len(train_data))
    test_data = TensorDataset(torch.FloatTensor(test_label), torch.LongTensor(test_index_num),
                              torch.FloatTensor(test_value_num), torch.LongTensor(test_index_cat),
                              torch.FloatTensor(test_value_cat))
    print(len(test_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)

    return train_loader, test_loader
