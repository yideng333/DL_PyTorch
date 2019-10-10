import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    # d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def load_data_fashion_mnist(data_dir, batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True,
                                                   transform=transforms.ToTensor())
    print(len(mnist_train), len(mnist_test))
    print(mnist_train[0][0].shape)

    train_loader = torch.utils.data.DataLoader(mnist_train,
                                               batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(mnist_test,
                                              batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, test_loader
