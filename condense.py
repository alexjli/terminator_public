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
    def __init__(self, hidden_dim = 64, num_features = NUM_FEATURES, filter_len = 3, 
        num_blocks = 4, num_transformers = 4, nheads = 4, device = 'cuda:0', track_nans = True):

        super(MultiChainCondenseMSA, self).__init__()
        channels = hidden_dim
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.embedding = ResidueFeatures(hidden_dim = hidden_dim, num_features = num_features)
        self.term_features = TERMProteinFeatures(edge_features = hidden_dim, node_features = hidden_dim)
        self.resnet = Conv1DResNet(filter_len = filter_len, channels = channels, num_blocks = num_blocks)
        self.encoder = S2STERMTransformerEncoder(node_features = hidden_dim, edge_features = hidden_dim, hidden_dim = hidden_dim, num_heads = nheads)
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

        if self.track_nan:
            process_nan(embeddings, (msas, features, self.embedding.embedding.weight), 'embed')

        # use Convolutional ResNet and averaging for further embedding and to reduce dimensionality
        convolution = self.resnet(embeddings)
        # zero out biases introduced into padding
        convolution *= negate_padding_mask.float()
        #convolution = embeddings.mean(dim=-1).transpose(1,2)

        if self.track_nan: process_nan(convolution, embeddings, 'convolve')

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
        batch_V, batch_E, batch_E_idx = self.term_features(batchify_coords, batched_focuses, chain_idx, batchify_src_key_mask.float())
        """
        print(batch_E_idx)
        print(batchify_src_key_mask)
        """

        # big transform
        node_embeddings = self.encoder(batchify_terms, batch_E, batch_E_idx, mask = batchify_src_key_mask.float())

        if self.track_nan: process_nan(node_embeddings, convolution, 'transform')
        #print("batched_focuses", batched_focuses[:, 4:5])
        #exit()

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


