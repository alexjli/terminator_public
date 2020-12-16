import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

import numpy as np
import math
from scipy.linalg import block_diag

from batched_term_transformer.term_attn import *

NUM_AA = 21

# resnet based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# and https://arxiv.org/pdf/1603.05027.pdf

NUM_FEATURES = len(['sin_phi', 'sin_psi', 'sin_omega', 'cos_phi', 'cos_psi', 'cos_omega', 'env', 'rmsd', 'term_len'])

ERROR_FILE = '/nobackup/users/vsundar/TERMinator/run.error'

def conv1xN(channels, N):
    return nn.Conv2d(channels, channels, kernel_size = (1, N), padding = (0, N//2))

def size(tensor):
    return str((tensor.element_size() * tensor.nelement())/(1<<20)) + " MB"

class Conv1DResidual(nn.Module):
    def __init__(self, hparams):
        super(Conv1DResidual, self).__init__()

        self.bn1 = nn.BatchNorm2d(hparams['hidden_dim'])
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = conv1xN(hparams['hidden_dim'], hparams['conv_filter'])
        self.bn2 = nn.BatchNorm2d(hparams['hidden_dim'])
        self.conv2 = conv1xN(hparams['hidden_dim'], hparams['conv_filter'])

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
    def __init__(self, hparams):
        super(Conv1DResNet, self).__init__()
        self.hparams = hparams

        blocks = [self._make_layer(hparams) for _ in range(hparams['resnet_blocks'])]
        self.resnet = nn.Sequential(*blocks)


    def _make_layer(self, hparams):
        return Conv1DResidual(hparams)

    def forward(self, X):
        # X: num batches x num channels x TERM length x num alignments
        # out retains the shape of X
        # X = self.bn(X)
        if self.hparams['resnet_linear']:
            out = X
        else:
            out = self.resnet(X)

        # average along axis of alignments
        # out: num batches x hidden dim x TERM length
        out = out.mean(dim = -1)

        # put samples back in rows
        # out: num batches x TERM length x hidden dim
        out = out.transpose(1,2)

        return out

class ResidueFeatures(nn.Module):
    def __init__(self, hparams):
        super(ResidueFeatures, self).__init__()
        self.hparams = hparams
        
        self.embedding = nn.Embedding(NUM_AA, hparams['hidden_dim'] - hparams['num_features'])
        self.linear = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'])
        
        self.relu = nn.ReLU(inplace = True)
        self.tanh = nn.Tanh()
        self.lin1 = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'])
        self.lin2 = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'])
        self.bn = nn.BatchNorm2d(hparams['hidden_dim'])

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

        # normalize over channels (TERM length x num alignments)
        out = self.bn(out)

        # embed features using ffn
        out = out.transpose(1,3)
        out = self.lin1(out)
        if not self.hparams['resnet_linear']:
            out = self.relu(out)
            out = self.lin2(out)
            out = self.tanh(out)

        # retranspose so features are channels
        out = out.transpose(1,3)

        return out

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class FocusEncoding(nn.Module):
    def __init__(self, hparams):
        super(FocusEncoding, self).__init__()
        self.hparams = hparams
        self.dropout = nn.Dropout(p=hparams['fe_dropout'])
        self.hidden_dim = hparams['hidden_dim']

        pe = torch.zeros(hparams['fe_max_len'], hparams['hidden_dim'])
        position = torch.arange(0, hparams['fe_max_len'], dtype=torch.double).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hparams['hidden_dim'], 2).double() * (-math.log(10000.0) / hparams['hidden_dim']))
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
    def __init__(self, hparams, device = 'cuda:0', track_nans = True):
        super(CondenseMSA, self).__init__()
        self.hparams = hparams
        self.embedding = ResidueFeatures(hparams = self.hparams)
        self.fe = FocusEncoding(hparams = self.hparams)
        self.resnet = Conv1DResNet(hparams = self.hparams)
        self.transformer = TERMTransformerLayer(hparams = self.hparams)
        self.encoder = TERMTransformer(hparams = self.hparams, transformer = self.transformer)
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
    def forward(self, X, features, seq_lens, focuses, term_lens, src_key_mask, max_seq_len):
        n_batches = X.shape[0]
        seq_lens = seq_lens.tolist()
        term_lens = term_lens.tolist()
        for i in range(len(term_lens)):
            for j in range(len(term_lens[i])):
                if term_lens[i][j] == -1:
                    term_lens[i] = term_lens[i][:j]
                    break
        local_dev = X.device

        # zero out all positions used as padding so they don't contribute to aggregation
        negate_padding_mask = (~src_key_mask).unsqueeze(-1).expand(-1,-1, self.hparams['hidden_dim'])

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
        if self.hparams['resnet_linear']:
            batched_flat_terms = convolution
        else:
            batched_flat_terms = self.fe(convolution, focuses, mask = ~src_key_mask)
        # reshape batched flat terms into batches of terms
        batchify_terms = self.batchify(batched_flat_terms, term_lens)
        # also reshape the mask
        batchify_src_key_mask = self.batchify(~src_key_mask, term_lens)
        # big transform
        if self.hparams['transformer_linear']:
            node_embeddings = batchify_terms
        else:
            node_embeddings = self.encoder(batchify_terms, src_mask = batchify_src_key_mask.float(), mask_attend = batchify_src_key_mask)

        if self.track_nan: process_nan(node_embeddings, convolution, 'transform')

        # we also need to batch focuses to we can aggregate data
        batched_focuses = self.batchify(focuses, term_lens).to(local_dev)

        # create a space to aggregate term data
        aggregate = torch.zeros((n_batches, max_seq_len, self.hparams['hidden_dim'])).to(local_dev)
        count = torch.zeros((n_batches, max_seq_len, 1)).to(local_dev).long()

        # this make sure each batch stays in the same layer during aggregation
        layer = torch.arange(n_batches).unsqueeze(-1).unsqueeze(-1).expand(batched_focuses.shape).long().to(local_dev)

        # aggregate node embeddings and associated counts
        aggregate = aggregate.index_put((layer, batched_focuses), node_embeddings, accumulate=True)
        count_idx = torch.ones_like(batched_focuses).unsqueeze(-1).to(local_dev)
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
        with open(ERROR_FILE, 'w') as fp:
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
        with open(ERROR_FILE, 'w') as fp:
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
        with open(ERROR_FILE, 'w') as fp:
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
                with open(ERROR_FILE, 'w') as fp:
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
