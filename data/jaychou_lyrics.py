import torch
import random
import zipfile
import os
from sklearn.preprocessing import LabelEncoder


def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(X, n_class):
    # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def load_data_jay_lyrics(data_dir):
    with open(os.path.join(data_dir, 'jaychou_lyrics.txt'), 'r') as f:
        corpus_chars = f.read()

    print(corpus_chars[:40])
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    print(corpus_chars[:40])

    idx_to_char = list(set(corpus_chars))
    # print(idx_to_char)
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    # print(char_to_idx)
    vocab_size = len(char_to_idx)
    print(vocab_size)

    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    sample = corpus_indices[:20]
    print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    print('indices:', sample)

    return corpus_indices, idx_to_char, char_to_idx, vocab_size


def data_iter_consecutive(corpus_indices, batch_size, num_steps):
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    # print(indices)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
