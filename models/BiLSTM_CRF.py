import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
# from torchcrf import CRF
from models.CRF import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# torch.manual_seed(1)


class BiLSTM_CRF(nn.Module):

    def __init__(self, args):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.vocab_size = args.vocab_size
        # self.tag_to_ix = args.tag_to_ix
        # don't count the padding tag for the classifier output
        self.tagset_size = args.tagset_size

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = 0
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            dropout=0.5, num_layers=1, bidirectional=True)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim,
                          dropout=0.5, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.tagset_size)

        # initial crf layer
        self.crf = CRF(self.tagset_size)

    # def init_hidden(self):
    #     return (torch.randn(2, 2, self.hidden_dim),
    #             torch.randn(2, 2, self.hidden_dim))

    def _get_lstm_features(self, sentence, lengths):  # (batch_size, seq_length)
        embeds = self.word_embeds(sentence).transpose(1, 0)  # (seq_length, batch_size, embedding_size)
        embeds_packed = pack_padded_sequence(embeds, lengths)
        lstm_out, hidden = self.lstm(embeds_packed)  # (seq_length, batch_size, hidden_size)
        lstm_out_padded, _ = pad_packed_sequence(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out_padded)  # (seq_length, batch_size, tag_size)
        # print(lstm_feats.shape)
        return lstm_feats

    def neg_log_likelihood(self, sentence, targets, lengths):
        feats = self._get_lstm_features(sentence, lengths)
        # feats: (seq_length, batch_size, tag_size)
        # tags: (batch_size, seq_length)
        mask = (sentence > 0).transpose(1, 0)
        return - self.crf(feats, targets.transpose(0, 1), mask)

    def forward(self, sentence, lengths, concated=False):  # use for prediction
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, lengths)

        # Find the best path, given the features.
        mask = (sentence > 0).transpose(1, 0)
        tag_seq = self.crf.decode(lstm_feats, mask, concated)
        return tag_seq
