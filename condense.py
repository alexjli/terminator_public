import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
import numpy as np

from batched_term_transformer.term_attn import *
from batched_term_transformer.term_features import * 
from struct2seq.self_attention import cat_neighbors_nodes, gather_nodes
from struct2seq.struct2seq import Struct2Seq
from nets import *

NUM_FEATURES = len(['sin_phi', 'sin_psi', 'sin_omega', 'cos_phi', 'cos_psi', 'cos_omega', 'env', 'rmsd', 'term_len'])

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
    def __init__(self, hidden_dim = 64, num_features = NUM_FEATURES, filter_len = 3, 
        num_blocks = 4, num_transformers = 4, nheads = 4, device = 'cuda:0', track_nans = True):

        super(MultiChainCondenseMSA_g, self).__init__()
        channels = hidden_dim
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.embedding = ResidueFeatures(hidden_dim = hidden_dim, num_features = num_features)
        self.term_features = TERMProteinFeatures(edge_features = hidden_dim, node_features = hidden_dim)
        self.resnet = Conv1DResNet(filter_len = filter_len, channels = channels, num_blocks = num_blocks)
        self.encoder = TERMGraphTransformerEncoder(node_features = hidden_dim, edge_features = hidden_dim, hidden_dim = hidden_dim, num_heads = nheads)

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
    def forward(self, msas, features, seq_lens, focuses, term_lens, src_key_mask, chain_idx, coords):
        n_batches = msas.shape[0]
        max_seq_len = max(seq_lens)

        # zero out all positions used as padding so they don't contribute to aggregation
        negate_padding_mask = (~src_key_mask).unsqueeze(-1).expand(-1,-1, self.hidden_dim)

        # embed MSAs and concat other features on
        embeddings = self.embedding(msas, features)

        # use Convolutional ResNet and averaging for further embedding and to reduce dimensionality
        convolution = self.resnet(embeddings)
        # zero out biases introduced into padding
        convolution *= negate_padding_mask.float()
       
        # reshape batched flat terms into batches of terms
        batchify_terms = self.batchify(convolution, term_lens)
        # also reshape the mask
        batchify_src_key_mask = self.batchify(~src_key_mask, term_lens)
        # and the coordinates
        term_coords = torch.gather(coords, 1, focuses.view(list(focuses.shape) + [1,1]).expand(-1, -1, 4, 3).to(self.dev))
        batchify_coords = self.batchify(term_coords, term_lens)
        # we also need to batch focuses to we can aggregate data
        batched_focuses = self.batchify(focuses, term_lens).to(self.dev)

        # generate node, edge features from batchified coords
        batch_V, batch_E, batch_rel_E_idx, batch_abs_E_idx = self.term_features(batchify_coords, batched_focuses, chain_idx, batchify_src_key_mask.float())

        # big transform
        node_embeddings, edge_embeddings = self.encoder(batchify_terms, batch_E, batch_rel_E_idx, mask = batchify_src_key_mask.float())

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

        agg_edges = aggregate_edges(edge_embeddings, batch_abs_E_idx)

        return aggregate, agg_edges

def aggregate_edges(edge_embeddings, E_idx):
    dev = edge_embeddings.device
    n_batch, n_terms, n_aa, n_neighbors, hidden_dim = edge_embeddings.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_terms, n_aa, n_aa, hidden_dim)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, -1, hidden_dim).to(dev)
    collection.scatter_add_(3, neighbor_idx, edge_embeddings)

    # we also need counts for averaging
    count = torch.zeros((n_batch, n_terms, n_aa, n_aa,1)).to(dev)
    count_idx = torch.ones_like(neighbor_idx).to(dev)
    count.scatter_add_(3, neighbor_idx, count_idx)

    return collection / count.float()
