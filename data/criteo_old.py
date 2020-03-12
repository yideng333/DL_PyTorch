import torch
from torch.utils.data import TensorDataset, DataLoader


def load_criteo_category_index(file_path):
    f = open(file_path,'r')
    cate_dict = []
    for i in range(39):
        cate_dict.append({})
    for line in f:
        datas = line.strip().split(',')
        cate_dict[int(datas[0])][datas[1]] = int(datas[2])
    return cate_dict


def read_criteo_data(file_path, features_sizes):
    label = []
    result = {'index': [], 'value': []}
    f = open(file_path, 'r')
    for line in f:
        datas = line.strip().split(',')
        label.append(int(datas[0]))

        indexs = []
        for k, item in enumerate(datas[1:]):
            indexs.append(sum(features_sizes[:k]) + int(item))
        # indexs = [int(item) for item in datas[1:]]
        values = [1 for i in range(39)]
        result['index'].append(indexs)
        result['value'].append(values)
    return label, result


def load_fm_data(data_dir, batch_size, features_sizes):

    train_label, train_dict = read_criteo_data('./datasets/tiny_train_input.csv', features_sizes)
    # print(train_dict)
    test_label, test_dict = read_criteo_data('./datasets/tiny_valid_5W.csv', features_sizes)
    # print(test_dict)
    # print(test_dict['index'][:10], test_dict['value'][:10])

    print(sum(train_label) / len(train_label))

    print(sum(test_label) / len(test_label))

    train_data = TensorDataset(torch.FloatTensor(train_label), torch.LongTensor(train_dict['index']),
                               torch.FloatTensor(train_dict['value']))
    print(len(train_data))
    test_data = TensorDataset(torch.FloatTensor(test_label), torch.LongTensor(test_dict['index']),
                              torch.FloatTensor(test_dict['value']))
    print(len(test_data))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8)

    return train_loader, test_loader
