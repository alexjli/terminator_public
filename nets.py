import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

import numpy as np
import math
from scipy.linalg import block_diag

from batched_attn_mask.transformer import TransformerEncoderLayer
from batched_term_transformer.term_attn import *

NUM_AA = 21

# resnet based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# and https://arxiv.org/pdf/1603.05027.pdf

NUM_FEATURES = len(['sin_phi', 'sin_psi', 'sin_omega', 'cos_phi', 'cos_psi', 'cos_omega', 'env', 'rmsd', 'term_len'])

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

        self.bn1.register_forward_hook(inf_nan_hook_fn)
        self.conv1.register_forward_hook(inf_nan_hook_fn)
        self.bn2.register_forward_hook(inf_nan_hook_fn)
        self.conv2.register_forward_hook(inf_nan_hook_fn)


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

        # average along axis of alignments
        # out: num batches x hidden dim x TERM length
        out = out.mean(dim = -1)

        # put samples back in rows
        # out: num batches x TERM length x hidden dim
        out = out.transpose(1,2)

        return out

class ResidueFeatures(nn.Module):
    def __init__(self, hidden_dim = 64, num_features = NUM_FEATURES):
        super(ResidueFeatures, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        
        self.embedding = nn.Embedding(NUM_AA, hidden_dim - num_features)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        """
        assert hidden_dim%2 == 0, "please choose a hidden dim that's a multiple of 2"
        self.embedding = nn.Embedding(NUM_AA, hidden_dim//2)
        self.linear = nn.Linear(num_features, hidden_dim//2)
        """
        self.relu = nn.ReLU(inplace = True)
        self.tanh = nn.Tanh()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm2d(hidden_dim)

    def forward(self, X, features):
        # X: num batches x num alignments x sum TERM length
        # features: num_batches x num alignments x sum TERM length x num features
        # samples in X are in rows
        embedded = self.embedding(X)

        """
        features = self.bn(features.transpose(1,3))
        embed_features = self.linear(features.transpose(1,3))
        """

        # hidden dim = embedding hidden dim + num features
        # out: num batches x num alignments x TERM length x hidden dim
        out = torch.cat((embedded, features), dim=3)

        # transpose so that features = number of channels for convolution
        # out: num batches x num channels x TERM length x num alignments
        out = out.transpose(1,3)

        # normalize over channels (TERM length x num alignments)
        out = self.bn(out)

        # embed features using ffn
        out = out.transpose(1,3)
        out = self.lin1(out)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.tanh(out)

        # retranspose so features are channels
        out = out.transpose(1,3)

        return out

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class FocusEncoding(nn.Module):
    def __init__(self, hidden_dim = 64, dropout = 0.1, max_len = 1000):
        super(FocusEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.double).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).double() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X, focuses, mask = None):
        fe = self.pe[focuses, :]
        if mask is not None:
            fe = fe * mask.unsqueeze(-1).float()

        return self.dropout(X + fe)

# TODO: differential positional encodings

class CondenseMSA(nn.Module):
    def __init__(self, hidden_dim = 64, num_features = NUM_FEATURES, filter_len = 3, num_blocks = 4, num_transformers = 4, nheads = 8, device = 'cuda:0', track_nans = True):
        super(CondenseMSA, self).__init__()
        channels = hidden_dim
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.embedding = ResidueFeatures(hidden_dim = hidden_dim, num_features = num_features)
        self.fe = FocusEncoding(hidden_dim = self.hidden_dim, dropout = 0.1, max_len = 1000)
        self.resnet = Conv1DResNet(filter_len = filter_len, channels = channels, num_blocks = num_blocks)
        self.transformer = TERMTransformerLayer(num_hidden = hidden_dim, num_heads = nheads)
        self.encoder = TERMTransformer(self.transformer, num_layers=num_transformers)
        self.batchify = BatchifyTERM()
        self.track_nan = track_nans

        if torch.cuda.is_available():
            self.dev = device
        else:
            print('No CUDA device detected. Defaulting to cpu')
            self.dev = 'cpu'

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def get_aa_embedding_layer(self):
        return self.embedding.embedding

    """
    S p e e e e d
    Fully batched
    """
    def forward(self, X, features, seq_lens, focuses, term_lens, src_key_mask):
        n_batches = X.shape[0]
        max_seq_len = max(seq_lens)

        # zero out all positions used as padding so they don't contribute to aggregation
        negate_padding_mask = (~src_key_mask).unsqueeze(-1).expand(-1,-1, self.hidden_dim)

        # embed MSAs and concat other features on
        embeddings = self.embedding(X, features)

        if self.track_nan:
            process_nan(embeddings, (X, features, self.embedding.embedding.weight), 'embed')

        # use Convolutional ResNet and averaging for further embedding and to reduce dimensionality
        convolution = self.resnet(embeddings)
        # zero out biases introduced into padding
        convolution *= negate_padding_mask.float()
        #convolution = embeddings.mean(dim=-1).transpose(1,2)

        if self.track_nan: process_nan(convolution, embeddings, 'convolve')

        #print("embed", torch.isnan(convolution).any())
        # add absolute positional encodings before transformer
        batched_flat_terms = self.fe(convolution, focuses, mask = ~src_key_mask)
        # reshape batched flat terms into batches of terms
        batchify_terms = self.batchify(batched_flat_terms, term_lens)
        # also reshape the mask
        batchify_src_key_mask = self.batchify(~src_key_mask, term_lens)
        # big transform
        node_embeddings = self.encoder(batchify_terms, src_mask = batchify_src_key_mask.float(), mask_attend = batchify_src_key_mask)

        if self.track_nan: process_nan(node_embeddings, convolution, 'transform')

        # we also need to batch focuses to we can aggregate data
        batched_focuses = self.batchify(focuses, term_lens).to(self.dev)

        # create a space to aggregate term data
        aggregate = torch.zeros((n_batches, max_seq_len, self.hidden_dim)).to(self.dev)
        count = torch.zeros((n_batches, max_seq_len, 1)).to(self.dev).long()

        # this make sure each batch stays in the same layer during aggregation
        layer = torch.arange(n_batches).unsqueeze(-1).unsqueeze(-1).expand(batched_focuses.shape).long().to(self.dev)

        # aggregate node embeddings and associated counts
        aggregate = aggregate.index_put((layer, batched_focuses), node_embeddings, accumulate=True)
        count_idx = torch.ones_like(batched_focuses).unsqueeze(-1).to(self.dev)
        count = count.index_put((layer, batched_focuses), count_idx, accumulate=True)

        # set all the padding zeros in count to 1 so we don't get nan's from divide by zero
        for batch, sel in enumerate(seq_lens):
            count[batch, sel:] = 1

        # average the aggregate
        aggregate /= count.float()

        if self.track_nan: process_nan(aggregate, node_embeddings, 'aggregate')

        return aggregate

class Wrapper(nn.Module):
    def __init__(self, hidden_dim = 64, num_features = NUM_FEATURES, filter_len = 3, num_blocks = 4, nheads = 8, device = 'cuda:0'):
        super(Wrapper, self).__init__()
        self.condense = CondenseMSA(hidden_dim = hidden_dim, num_features = num_features, filter_len = filter_len, num_blocks = num_blocks, nheads = nheads, device = device)
        self.shape1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.shape2 = nn.Linear(hidden_dim, 22)

    def forward(self, msas, features, seq_lens, focuses, term_lens, src_mask, src_key_mask):
        output = self.condense(msas, features, seq_lens, focuses, term_lens, src_mask, src_key_mask)
        output = self.shape1(output)
        output = self.relu(output)
        reshape = self.shape2(output)
        return reshape

def process_nan(t, prev_t, msg):
    if torch.isnan(t).any():
        with open('/home/ifsdata/scratch/grigoryanlab/alexjli/run.error', 'w') as fp:
            fp.write(repr(prev_t) + '\n')
            fp.write(repr(t) + '\n')
            fp.write(str(msg))
        raise KeyboardInterrupt

def stat_cuda(msg):
    dev1, dev2, dev3 = 0, 1, 2
    print('--', msg)
    print('allocated 1: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated(dev1) / 1024 / 1024,
        torch.cuda.max_memory_allocated(dev1) / 1024 / 1024,
        torch.cuda.memory_cached(dev1) / 1024 / 1024,
        torch.cuda.max_memory_cached(dev1) / 1024 / 1024
    ))
    print('allocated 2: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated(dev2) / 1024 / 1024,
        torch.cuda.max_memory_allocated(dev2) / 1024 / 1024,
        torch.cuda.memory_cached(dev2) / 1024 / 1024,
        torch.cuda.max_memory_cached(dev2) / 1024 / 1024
    ))
    print('allocated 3: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated(dev3) / 1024 / 1024,
        torch.cuda.max_memory_allocated(dev3) / 1024 / 1024,
        torch.cuda.memory_cached(dev3) / 1024 / 1024,
        torch.cuda.max_memory_cached(dev3) / 1024 / 1024
    ))

def inf_nan_hook_fn(self, input, output):
    """
    try:
        print('mean', self.running_mean)
        print('var', self.running_var)
    except:
        pass
    """
    if has_large(input[0]):
        print("large mag input")
        with open('/home/ifsdata/scratch/grigoryanlab/alexjli/run.error', 'w') as fp:
            abs_input = torch.abs(input[0])
            fp.write('Input to ' + repr(self) + ' forward\n')
            fp.write('Inputs from b_idx 0 channel 0' + repr(input[0][0][0]) + '\n')
            fp.write("top 5 " + repr(torch.topk(abs_input.view([-1]), 5)) + '\n')
            fp.write('Weights ' + repr(self.weight) + '\n')
            fp.write('Biases ' + repr(self.bias) + '\n')
            try:
                fp.write('Running mean ' + repr(self.running_mean) + '\n')
                fp.write('Running var ' + repr(self.running_var) + '\n')
            except:
                fp.write('Not a batchnorm layer')
        exit()

    elif (output == float('inf')).any() or (output == float('-inf')).any() or torch.isnan(output).any():
        print('we got an inf/nan rip')
        with open('/home/ifsdata/scratch/grigoryanlab/alexjli/run.error', 'w') as fp:
            fp.write('Inf/nan is from ' + repr(self) + ' forward\n')
            fp.write('Inputs from b_idx 0 channel 0' + repr(input[0][0][0]) + '\n')
            fp.write('Outputs from b_idx 0 channel 0' + repr(output[0][0]) + '\n')
            fp.write('Weights ' + repr(self.weight) + '\n')
            fp.write('Biases ' + repr(self.bias) + '\n')
            try:
                fp.write('Running mean ' + repr(self.running_mean) + '\n')
                fp.write('Running var ' + repr(self.running_var) + '\n')
            except:
                fp.write('Not a batchnorm layer')
        exit()
    else:
        try:
            if is_nan_inf(self.running_mean) or is_nan_inf(self.running_var):
                with open('/home/ifsdata/scratch/grigoryanlab/alexjli/run.error', 'w') as fp:
                    fp.write('Inf/nan is from ' + repr(self) + ' forward\n')
                    fp.write('Inputs from b_idx 0 channel 0' + repr(input[0][0][0]) + '\n')
                    fp.write('Outputs from b_idx 0 channel 0' + repr(output[0][0]) + '\n')
                    fp.write('Weights ' + repr(self.weight) + '\n')
                    fp.write('Biases ' + repr(self.bias) + '\n')
                    fp.write('Running mean ' + repr(self.running_mean) + '\n')
                    fp.write('Running var ' + repr(self.running_var) + '\n')
                exit()
        except:
            pass


def is_nan_inf(output):
    return (output == float('inf')).any() or (output == float('-inf')).any() or torch.isnan(output).any()

def has_large(input):
    return torch.max(torch.abs(input)) > 1000
