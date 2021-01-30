import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

import numpy as np
import math
from scipy.linalg import block_diag

from layers.utils import *
from layers.term.matches.cnn import Conv1DResNet
from layers.term.matches.attn import TERMMatchTransformerEncoder
from layers.term.struct.self_attn import TERMTransformerLayer, TERMTransformer
from layers.term.struct.s2s import S2STERMTransformerEncoder, TERMGraphTransformerEncoder
from layers.term.struct.gvp import TERMGraphGVPEncoder
from layers.graph_features import *
from layers.gvp import *

NUM_AA = 21
NUM_FEATURES = len(['sin_phi', 'sin_psi', 'sin_omega', 'cos_phi', 'cos_psi', 'cos_omega', 'env', 'rmsd', 'term_len'])
NUM_TARGET_FEATURES = len(['sin_phi', 'sin_psi', 'sin_omega', 'cos_phi', 'cos_psi', 'cos_omega', 'env'])


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
        if hparams['matches'] == 'resnet':
            self.matches = Conv1DResNet(hparams = self.hparams)
        elif hparams['matches'] == 'transformer':
            self.matches = TERMMatchTransformerEncoder(hparams = hparams)
        else:
            raise InvalidArgumentError("arg for matches condenser doesn't look right")
        self.transformer = TERMTransformerLayer(hparams = self.hparams)
        self.encoder = TERMTransformer(hparams = self.hparams, transformer = self.transformer)
        self.batchify = BatchifyTERM()

        self.W_ppoe = nn.Linear(NUM_TARGET_FEATURES, hparams['hidden_dim'])

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
    def forward(self, X, features, seq_lens, focuses, term_lens, src_key_mask, max_seq_len, ppoe):
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


        # use Convolutional ResNet or Transformer 
        # for further embedding and to reduce dimensionality
        if self.hparams['matches'] == 'transformer':
            # project ppoe
            ppoe = self.W_ppoe(ppoe)
            # gather to generate target ppoe per term residue
            focuses_gather = focuses.unsqueeze(-1).expand(-1, -1, self.hparams['hidden_dim'])
            target = torch.gather(ppoe, 1, focuses_gather)

            # output dimensionality is a little different for transformer
            embeddings = embeddings.transpose(1,3).transpose(1,2)
            condensed_matches = self.matches(embeddings, target, mask = ~src_key_mask)
        else:
            condensed_matches = self.matches(embeddings)
        # zero out biases introduced into padding
        condensed_matches *= negate_padding_mask

        if self.track_nan: process_nan(condensed_matches, embeddings, 'convolve')


        # add absolute positional encodings before TERM transformer
        if self.hparams['resnet_linear']:
            batched_flat_terms = condensed_matches
        else:
            batched_flat_terms = self.fe(condensed_matches, focuses, mask = ~src_key_mask)
        # reshape batched flat terms into batches of terms
        batchify_terms = self.batchify(batched_flat_terms, term_lens)
        # also reshape the mask
        batchify_src_key_mask = self.batchify(~src_key_mask, term_lens)
        # big transform
        if self.hparams['transformer_linear']:
            node_embeddings = batchify_terms
        else:
            node_embeddings = self.encoder(batchify_terms, src_mask = batchify_src_key_mask, mask_attend = batchify_src_key_mask)

        if self.track_nan: process_nan(node_embeddings, condensed_matches, 'transform')

        # zero out padding for aggregation
        node_embeddings *= batchify_src_key_mask.unsqueeze(-1)

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


class MultiChainCondenseMSA(nn.Module):
    def __init__(self, hparams, device = 'cuda:0', track_nans = True):
        super(MultiChainCondenseMSA, self).__init__()
        self.hparams = hparams
        self.embedding = ResidueFeatures(hparams = self.hparams)
        self.fe = FocusEncoding(hparams = self.hparams)
        if hparams['matches'] == 'resnet':
            self.matches = Conv1DResNet(hparams = self.hparams)
        elif hparams['matches'] == 'transformer':
            self.matches = TERMMatchTransformerEncoder(hparams = hparams)
        else:
            raise InvalidArgumentError("arg for matches condenser doesn't look right")
        self.encoder = S2STERMTransformerEncoder(hparams = self.hparams)
        self.batchify = BatchifyTERM()
        self.term_features = TERMProteinFeatures(edge_features = hparams['hidden_dim'], node_features =  hparams['hidden_dim'])

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
    def forward(self, X, features, seq_lens, focuses, term_lens, src_key_mask, max_seq_len, chain_idx, coords):
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


        # use Convolutional ResNet or Transformer 
        # for further embedding and to reduce dimensionality
        if self.hparams['matches'] == 'transformer':
            # output dimensionality is a little different for transformer
            embeddings = embeddings.transpose(1,3).transpose(1,2)
            condensed_matches = self.matches(embeddings, ~src_key_mask)
        else:
            condensed_matches = self.matches(embeddings)
        # zero out biases introduced into padding
        condensed_matches *= negate_padding_mask.float()

        if self.track_nan: process_nan(condensed_matches, embeddings, 'convolve')


        # add absolute positional encodings before TERM transformer
        if self.hparams['resnet_linear']:
            batched_flat_terms = condensed_matches
        else:
            batched_flat_terms = self.fe(condensed_matches, focuses, mask = ~src_key_mask)
        # reshape batched flat terms into batches of terms
        batchify_terms = self.batchify(batched_flat_terms, term_lens)
        # also reshape the mask
        batchify_src_key_mask = self.batchify(~src_key_mask, term_lens)
        # we also need to batch focuses to we can aggregate data
        batched_focuses = self.batchify(focuses, term_lens).to(local_dev)

        # transform?
        if self.hparams['transformer_linear']:
            node_embeddings = batchify_terms
        else: 
            # and reshape the coordinates
            term_coords = torch.gather(coords, 1, focuses.view(list(focuses.shape) + [1,1]).expand(-1, -1, 4, 3).to(local_dev))
            batchify_coords = self.batchify(term_coords, term_lens)

            # generate node, edge features from batchified coords
            batch_V, batch_E, batch_E_idx, _ = self.term_features(batchify_coords, batched_focuses, chain_idx, batchify_src_key_mask.float())

            # big transform
            node_embeddings = self.encoder(batchify_terms, batch_E, batch_E_idx, mask = batchify_src_key_mask.float())

        if self.track_nan: process_nan(node_embeddings, condensed_matches, 'transform')


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

class MultiChainCondenseMSA_g(nn.Module):
    def __init__(self, hparams, device = 'cuda:0', track_nans = True):
        super(MultiChainCondenseMSA_g, self).__init__()
        self.hparams = hparams
        self.embedding = ResidueFeatures(hparams = self.hparams)
        self.fe = FocusEncoding(hparams = self.hparams)
        if hparams['matches'] == 'resnet':
            self.matches = Conv1DResNet(hparams = self.hparams)
        elif hparams['matches'] == 'transformer':
            self.matches = TERMMatchTransformerEncoder(hparams = hparams)
        else:
            raise InvalidArgumentError("arg for matches condenser doesn't look right")
        self.encoder = TERMGraphTransformerEncoder(hparams = self.hparams)
        self.batchify = BatchifyTERM()
        self.term_features = TERMProteinFeatures(edge_features = hparams['hidden_dim'], node_features =  hparams['hidden_dim'])

        self.W_ppoe = nn.Linear(NUM_TARGET_FEATURES, hparams['hidden_dim'])

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
    def forward(self, X, features, seq_lens, focuses, term_lens, src_key_mask, max_seq_len, chain_idx, coords, ppoe):
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


        # use Convolutional ResNet or Transformer 
        # for further embedding and to reduce dimensionality
        if self.hparams['matches'] == 'transformer':
            # project ppoe
            ppoe = self.W_ppoe(ppoe)
            # gather to generate target ppoe per term residue
            focuses_gather = focuses.unsqueeze(-1).expand(-1, -1, self.hparams['hidden_dim'])
            target = torch.gather(ppoe, 1, focuses_gather)

            # output dimensionality is a little different for transformer
            embeddings = embeddings.transpose(1,3).transpose(1,2)
            condensed_matches = self.matches(embeddings, target, ~src_key_mask)
        else:
            condensed_matches = self.matches(embeddings)
        # zero out biases introduced into padding
        condensed_matches *= negate_padding_mask.float()

        if self.track_nan: process_nan(condensed_matches, embeddings, 'convolve')

        # reshape batched flat terms into batches of terms
        batchify_terms = self.batchify(condensed_matches, term_lens)
        # also reshape the mask
        batchify_src_key_mask = self.batchify(~src_key_mask, term_lens)
        # we also need to batch focuses to we can aggregate data
        batched_focuses = self.batchify(focuses, term_lens).to(local_dev)
        
        # and reshape the coordinates
        term_coords = torch.gather(coords, 1, focuses.view(list(focuses.shape) + [1,1]).expand(-1, -1, 4, 3).to(local_dev))
        batchify_coords = self.batchify(term_coords, term_lens)

        # generate node, edge features from batchified coords
        batch_V, batch_E, batch_rel_E_idx, batch_abs_E_idx = self.term_features(batchify_coords, batched_focuses, chain_idx, batchify_src_key_mask.float())

        # big transform
        node_embeddings, edge_embeddings = self.encoder(batchify_terms, batch_E, batch_rel_E_idx, mask = batchify_src_key_mask.float())

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

        agg_edges = aggregate_edges(edge_embeddings, batch_abs_E_idx, max_seq_len)

        return aggregate, agg_edges


class GVPCondenseMSA(nn.Module):
    def __init__(self, hparams, device = 'cuda:0', track_nans = True):
        super(GVPCondenseMSA, self).__init__()
        self.hparams = hparams
        self.embedding = ResidueFeatures(hparams = self.hparams)
        self.fe = FocusEncoding(hparams = self.hparams)
        if hparams['matches'] == 'resnet':
            self.matches = Conv1DResNet(hparams = self.hparams)
        elif hparams['matches'] == 'transformer':
            self.matches = TERMMatchTransformerEncoder(hparams = hparams)
        else:
            raise InvalidArgumentError("arg for matches condenser doesn't look right")
        self.encoder = TERMGraphGVPEncoder(hparams = self.hparams)
        self.batchify = BatchifyTERM()
        self.term_features = GVPTProteinFeatures(edge_features = hparams['hidden_dim']//2, node_features =  hparams['hidden_dim']//2)

        self.track_nan = track_nans
        self.W_ppoe = nn.Linear(NUM_TARGET_FEATURES, hparams['hidden_dim'])

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
    def forward(self, X, features, seq_lens, focuses, term_lens, src_key_mask, max_seq_len, chain_idx, coords, ppoe):
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


        # use Convolutional ResNet or Transformer 
        # for further embedding and to reduce dimensionality
        if self.hparams['matches'] == 'transformer':
            # project ppoe
            ppoe = self.W_ppoe(ppoe)
            # gather to generate target ppoe per term residue
            focuses_gather = focuses.unsqueeze(-1).expand(-1, -1, self.hparams['hidden_dim'])
            target = torch.gather(ppoe, 1, focuses_gather)

            # output dimensionality is a little different for transformer
            embeddings = embeddings.transpose(1,3).transpose(1,2)
            condensed_matches = self.matches(embeddings, target, ~src_key_mask)
        else:
            condensed_matches = self.matches(embeddings)
        # zero out biases introduced into padding
        condensed_matches *= negate_padding_mask.float()

        if self.track_nan: process_nan(condensed_matches, embeddings, 'convolve')

        # reshape batched flat terms into batches of terms
        batchify_terms = self.batchify(condensed_matches, term_lens)
        # also reshape the mask
        batchify_src_key_mask = self.batchify(~src_key_mask, term_lens)
        # we also need to batch focuses to we can aggregate data
        batched_focuses = self.batchify(focuses, term_lens).to(local_dev)
        
        # and reshape the coordinates
        term_coords = torch.gather(coords, 1, focuses.view(list(focuses.shape) + [1,1]).expand(-1, -1, 4, 3).to(local_dev))
        batchify_coords = self.batchify(term_coords, term_lens)

        # generate node, edge features from batchified coords
        batch_V, batch_E, batch_rel_E_idx, batch_abs_E_idx = self.term_features(batchify_coords, batchify_src_key_mask.float(), chain_idx, batched_focuses = batched_focuses)

        # we can do this because the scale channels are on the bottom
        # and there are no vector channels to add
        batch_V = torch.cat([batch_V, batchify_terms], dim=-1)

        # big transform
        node_embeddings, edge_embeddings = self.encoder(batch_V, batch_E, batch_rel_E_idx, mask = batchify_src_key_mask.float())

        # create a space to aggregate term data
        aggregate = torch.zeros((n_batches, max_seq_len, self.hparams['hidden_dim']*2)).to(local_dev)
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

        agg_edges = aggregate_edges(edge_embeddings, batch_abs_E_idx, max_seq_len)

        return aggregate, agg_edges


