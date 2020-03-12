import torch
from torch import nn


class RNNModel(nn.Module):
    def __init__(self, args):
        super(RNNModel, self).__init__()

        self.num_hiddens = args.num_hiddens
        self.vocab_size = args.vocab_size
        self.num_layers = args.num_layers
        self.bidirectional = args.bidirectional

        if args.use_basic_rnn:
            self.rnn_layer = nn.RNN(input_size=self.vocab_size, hidden_size=self.num_hiddens,
                                    num_layers=self.num_layers, bidirectional=self.bidirectional)
        if args.use_lstm:
            self.rnn_layer = nn.LSTM(input_size=self.vocab_size, hidden_size=self.num_hiddens,
                                     num_layers=self.num_layers, bidirectional=self.bidirectional)

        self.dense = nn.Linear(self.num_hiddens * (2 if self.bidirectional else 1), self.vocab_size)
        self.state = None

    def forward(self, input, state):
        # input: (seq_len, batch, input_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        Y, self.state = self.rnn_layer(input, state)  # Y: (seq_len, batch, num_directions * hidden_size)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state
