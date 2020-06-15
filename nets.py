import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import math
from scipy.linalg import block_diag

from batched_attn_mask.transformer import TransformerEncoderLayer, TransformerEncoder

NUM_AA = 22

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

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X, focuses):
        fe = self.pe[focuses, :]
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
        self.transformer = TransformerEncoderLayer(d_model = hidden_dim, nhead = nheads, dim_feedforward = hidden_dim)
        self.encoder = TransformerEncoder(self.transformer, num_layers=4)
        if torch.cuda.is_available():
            self.dev = device
        else:
            self.dev = 'cpu'

    """
    S p e e e e d
    Fully batched
    """
    def forward(self, X, features, seq_lens, focuses, src_mask, src_key_mask):
        n_batches = X.shape[0]
        max_seq_len = max(seq_lens)
        import time
        last_timepoint = time.time()

        # use source mask so that terms can attend to themselves but not each
        # for more efficient computation, generate the mask over all heads first
        # first, stack the current source mask nhead times, and transpose to get batches of the same mask
        print(size(src_mask))
        src_mask = src_mask.unsqueeze(0).expand(self.nheads,-1,-1,-1).transpose(0,1)
        print(size(src_mask))
        # next, flatten the 0th dim to generate a 3d tensor
        dim = focuses.shape[1]
        src_mask = src_mask.contiguous()
        src_mask = src_mask.view(-1, dim, dim)
        #src_mask = torch.flatten(src_mask, 0, 1)
        print(size(src_mask))

        # zero out all positions used as padding so they don't contribute to aggregation
        negate_padding_mask = (~src_key_mask).unsqueeze(-1).expand(-1,-1, self.hidden_dim)

        current_timepoint = time.time()
        print('mask', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # embed MSAs and concat other features on
        embeddings = self.embedding(X, features)
        print(size(embeddings))

        current_timepoint = time.time()
        print('embed', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # use Convolutional ResNet and averaging for further embedding and to reduce dimensionality
        convolution = self.resnet(embeddings)
        # zero out biases introduced into padding
        convolution *= negate_padding_mask.float()
        print(size(convolution))

        current_timepoint = time.time()
        print('convolve', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint


        # add absolute positional encodings before transformer
        batch_terms = self.fe(convolution, focuses)
        # transpose because pytorch transformer uses weird shape
        batch_terms = batch_terms.transpose(0,1)
        # create node embeddings
        node_embeddings = self.encoder(batch_terms, mask = src_mask.float(), src_key_padding_mask = src_key_mask.byte())
        # transpose back
        node_embeddings = node_embeddings.transpose(0,1)
        print(size(node_embeddings))

        # zero out biases introduced into padding
        node_embeddings *= negate_padding_mask.float()

        current_timepoint = time.time()
        print('transform', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # create a space to aggregate term data
        aggregate = torch.zeros((n_batches, max_seq_len, self.hidden_dim)).to(self.dev)
        count = torch.zeros((n_batches, max_seq_len, 1)).to(self.dev)

        # this make sure each batch stays in the same layer during aggregation
        layer = torch.arange(n_batches).unsqueeze(-1).to(self.dev)
        #import pdb; pdb.set_trace()
        #breakpoint()

        focuses = focuses.to(self.dev)
        node_embeddings = node_embeddings.to(self.dev)

        # aggregate node embeddings and associated counts
        aggregate = aggregate.index_put((layer, focuses), node_embeddings, accumulate=True)
        count_idx = torch.ones_like(focuses).unsqueeze(-1).float().to(self.dev)
        count = count.index_put((layer, focuses), count_idx, accumulate=True)

        current_timepoint = time.time()
        print('aggregate', current_timepoint-last_timepoint)
        last_timepoint = current_timepoint

        # average the aggregate
        aggregate /= count

        return aggregate

class DummyLSTM(nn.Module):
    def __init__(self):
        self.lstm = nn.LSTM(input_size = 64, hidden_size=64, num_layers = 1, batch_first=True, bidirectional=True)

    def forward(self, X):
        return self.lstm(X)
