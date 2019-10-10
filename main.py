import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
from dotmap import DotMap
# from models.MLP import MLP
# from models.LeNet import LeNet
# from models.AlexNet import AlexNet
# from models.GoogLeNet import GooLeNet
from models.ResNet import ResNet_18
from utils.basic_train import train, test
from dataloader.Fashion_MNIST import load_data_fashion_mnist
data_dir = '~/datasets/FashionMNIST'


if __name__ == '__main__':
    args = DotMap({
                   # 'flatten': True,
                   # 'num_inpus': 784,
                   # 'num_outputs': 10,
                   # 'num_hiddens': [256, 128, 64],
                   # 'verbose': True,

                   'batch_size': 256,
                   'wd': 0,
                   'device': 'cuda',  # cpu / cuda
                   'epochs': 5,
                   'lr': 0.001,  # learning_rate
                   'log_interval': 10,  # log intervel
                   'save_model': False
                   })

    net = ResNet_18(args)
    print(net)

    # for blk in net.children():
    #     X = blk(X)
    #     print('output shape: ', X.shape)

    # 以均值为0，方差0.01初始化参数
    # for params in net.parameters():
    #     init.normal_(params, mean=0, std=0.01)
    # for name, param in net.named_parameters():
    #     print(name, param.size())

    # 读入数据集
    train_loader, test_loader = load_data_fashion_mnist(data_dir, args.batch_size, resize=96)
    print("training on ", args.device)
    model = net.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train(args, model, loss, train_loader, optimizer, epoch)
        test(args, model, loss, test_loader, epoch)

    # if args.save_model:
    #     torch.save(model.state_dict(), "fashion_mnist_mlp.pt")
