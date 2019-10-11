import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''
    :parameter
    'flatten': 是否把二维图像摊平,
    'num_inpus': 输入的特征数,
    'num_outputs': 输出的类别数,
    'num_hiddens': 每个隐藏层的神经元数量 e.g., [256, 128, 64],
    '''
    def __init__(self, args):
        super(MLP, self).__init__()

        self.args = args
        print('MLP get parameters:{}'.format(args))

        num_inputs = self.args.num_inpus
        num_outputs = self.args.num_outputs
        num_hiddens = self.args.num_hiddens

        self.input_layer = torch.nn.Sequential()
        self.input_layer.add_module('input', nn.Linear(num_inputs, num_hiddens[0]))
        self.input_layer.add_module('input_relu', nn.ReLU())

        if len(num_hiddens) > 1:
            self.hidden_layers = torch.nn.Sequential()
            for i in range(len(num_hiddens) - 1):
                self.hidden_layers.add_module('hidden_{}'.format(i), nn.Linear(num_hiddens[i], num_hiddens[i+1]))
                self.hidden_layers.add_module('relu_{}'.format(i), nn.ReLU())
            # self.hidden_layers = [nn.Linear(num_hiddens[i], num_hiddens[i+1]) for i in range(len(num_hiddens)-1)]
        else:
            self.hidden_layers = None

        self.output_layer = nn.Linear(num_hiddens[-1], num_outputs)

    def forward(self, x):
        # print('input:{}'.format(x.shape))
        if self.args.flatten:
            x = x.view(x.shape[0], -1)
            # print('flatten output:{}'.format(x.shape))

        x = self.input_layer(x)
        if self.hidden_layers:
            x = self.hidden_layers(x)
        y = self.output_layer(x)
        return y
