import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import math
from scipy.linalg import block_diag

from batched_attn_mask.transformer import TransformerEncoderLayer
from batched_term_transformer.term_attn import *

NUM_AA = 21

# resnet based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# and https://arxiv.org/pdf/1603.05027.pdf

num_features = len(['phi', 'psi', 'omega', 'env', 'rmsd', 'term_len', 'num_alignments'])

def conv1xN(channels, N):
    return nn.Conv2d(channels, channels, kernel_size = (1, N), padding = (0, N//2))

def size(tensor):
    return str((tensor.element_size() * tensor.nelement())/(1<<20)) + " MB"

class Conv1DResidual(nn.Module):
    def __init__(self, filter_len = 3, channels = 64):
        super(Conv1DResidual, self).__init__()
        self.filter_len = 3
        self.channels = 64

        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = conv1xN(channels, filter_len)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = conv1xN(channels, filter_len)

    def forward(self, X):
        identity = X

        out = self.bn1(X)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out

class Conv1DResNet(nn.Module):
    def __init__(self, filter_len = 3, channels = 64, num_blocks = 6):
        super(Conv1DResNet, self).__init__()
        self.filter_len = 3
        self.channels = channels
        self.num_blocks = num_blocks

        #self.bn = BatchNorm2d(channels)

        blocks = [self._make_layer(filter_len, channels) for _ in range(num_blocks)]
        self.resnet = nn.Sequential(*blocks)


    def _make_layer(self, filter_len, channels):
        return Conv1DResidual(filter_len = filter_len, channels = channels)

    def forward(self, X):
        # X: num batches x num channels x TERM length x num alignments
        # out retains the shape of X
        # X = self.bn(X)
        out = self.resnet(X)

        # average alone axis of alignments
        # out: num batches x hidden dim x TERM length
        out = out.mean(dim = -1)

        # put samples back in rows
        # out: num batches x TERM length x hidden dim
        out = out.transpose(1,2)

        return out

class ResidueFeatures(nn.Module):
    def __init__(self, hidden_dim = 64, num_features = num_features):
        super(ResidueFeatures, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.embedding = nn.Embedding(NUM_AA, hidden_dim - num_features)

    def forward(self, X, features):
        # X: num batches x num alignments x sum TERM length
        # features: num_batches x num alignments x sum TERM length x num features
        # samples in X are in rows
        embedded = self.embedding(X)

        # hidden dim = embedding hidden dim + num features
        # out: num batches x num alignments x TERM length x hidden dim
        out = torch.cat((embedded, features), dim=3)

        # transpose so that features = number of channels for convolution
        # out: num batches x num channels x TERM length x num alignments
        out = out.transpose(1,3)

        return out

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class FocusEncoding(nn.Module):
    def __init__(self, hidden_dim = 64, dropout = 0.1, max_len = 1000):
        super(FocusEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X, focuses, mask = None):
        fe = self.pe[focuses, :]
        if mask is not None:
            fe = fe * mask.unsqueeze(-1)

        return self.dropout(X + fe)

class CondenseMSA(nn.Module):
    def __init__(self, hidden_dim = 64, num_features = num_features, filter_len = 3, num_blocks = 4, nheads = 8, device = 'cuda:0'):
        super(CondenseMSA, self).__init__()
        channels = hidden_dim
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.embedding = ResidueFeatures(hidden_dim = hidden_dim, num_features = num_features)
        self.fe = FocusEncoding(hidden_dim = self.hidden_dim, dropout = 0.1, max_len = 1000)
        self.resnet = Conv1DResNet(filter_len = filter_len, channels = channels, num_blocks = num_blocks)
        self.transformer = TERMTransformerLayer(num_hidden = hidden_dim, num_heads = nheads)
        self.encoder = TERMTransformer(self.transformer, num_layers=4)
        self.batchify = BatchifyTERM()
        if torch.cuda.is_available():
            self.dev = device
        else:
            print('No CUDA device detected. Defaulting to cpu')
            self.dev = 'cpu'

    """
    S p e e e e d
    Fully batched
    """
    def forward(self, X, features, seq_lens, focuses, term_lens, src_mask, src_key_mask):
        n_batches = X.shape[0]
        max_seq_len = max(seq_lens)
        import time
        last_timepoint = time.time()

        # zero out all positions used as padding so they don't contribute to aggregation
        negate_padding_mask = (~src_key_mask).unsqueeze(-1).expand(-1,-1, self.hidden_dim)

        # embed MSAs and concat other features on
        embeddings = self.embedding(X, features)

        # use Convolutional ResNet and averaging for further embedding and to reduce dimensionality
        convolution = self.resnet(embeddings)
        # zero out biases introduced into padding
        convolution *= negate_padding_mask

        # add absolute positional encodings before transformer
        batched_flat_terms = self.fe(convolution, focuses, mask = ~src_key_mask)
        # reshape batched flat terms into batches of terms
        batchify_terms = self.batchify(batched_flat_terms, term_lens)
        # also reshape the mask
        batchify_src_key_mask = self.batchify(~src_key_mask, term_lens)
        # big transform
        node_embeddings = self.transformer(batchify_terms, mask_attend = batchify_src_key_mask)

        """
        # zero out biases introduced into padding
        node_embeddings *= negate_padding_mask
        """

        # we also need to batch focuses to we can aggregate data
        batched_focuses = self.batchify(focuses, term_lens)

        # create a space to aggregate term data
        aggregate = torch.zeros((n_batches, max_seq_len, self.hidden_dim)).to(self.dev)
        count = torch.zeros((n_batches, max_seq_len, 1)).to(self.dev).long()

        # this make sure each batch stays in the same layer during aggregation
        layer = torch.arange(n_batches).unsqueeze(-1).unsqueeze(-1).expand(batched_focuses.shape).long().to(self.dev)

        # aggregate node embeddings and associated counts
        aggregate = aggregate.index_put((layer, batched_focuses), node_embeddings, accumulate=True)
        count_idx = torch.ones_like(batched_focuses).unsqueeze(-1).to(self.dev)
        count = count.index_put((layer, batched_focuses), count_idx, accumulate=True)

        # average the aggregate
        aggregate /= count

        return aggregate


class DummyLSTM(nn.Module):
    def __init__(self):
        super(DummyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size = 32, hidden_size=32, num_layers = 2, batch_first=True, bidirectional=True)
        self.out_shape = nn.LSTM(input_size = 64, hidden_size = 11, num_layers = 1, batch_first = True, bidirectional = True)


    def forward(self, X):
        self.lstm.flatten_parameters()
        self.out_shape.flatten_parameters()
        states = self.lstm(X)[0]
        return self.out_shape(states)[0]
