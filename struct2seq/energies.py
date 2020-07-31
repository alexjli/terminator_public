from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_attention import *
from .protein_features import ProteinFeatures
from .struct2seq import Struct2Seq

class SelfEnergies(Struct2Seq):
    def __init__(self, num_letters, node_features, edge_features, input_dim,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=20, k_neighbors=30, protein_features='full', augment_eps=0.,
        dropout=0.1, forward_attention_decoder=True, use_mpnn=False,
        output_dim = 20):

        super(SelfEnergies, self).__init__(num_letters, node_features, edge_features,
            hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
            vocab=20, k_neighbors=30, protein_features='full', augment_eps=0.,
            dropout=0.1, forward_attention_decoder=True, use_mpnn=False)

        # Embedding layers
        self.W_v = nn.Linear(node_features + input_dim, hidden_dim, bias=True)
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, V_embed, X, x_mask):
        """ Graph-conditioned sequence model """

        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, x_mask)
        V = torch.cat([V, V_embed], dim = -1)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(x_mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = x_mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=x_mask, mask_attend=mask_attend)

        return self.W_out(h_V)

class PairEnergies(nn.Module):
    def __init__(self, num_letters, node_features, edge_features, input_dim,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=20, k_neighbors=30, protein_features='full', augment_eps=0.,
        dropout=0.1, forward_attention_decoder=True, use_mpnn=False,
        output_dim = 20 * 20):
        """ Graph labeling network """
        super(PairEnergies, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Featurization layers
        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors,
            features_type=protein_features, augment_eps=augment_eps,
            dropout=dropout
        )

        # Embedding layers
        self.W_v = nn.Linear(node_features + input_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        layer = EdgeTransformerLayer

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            layer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.W_out = nn.Linear(hidden_dim, output_dim, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V_embed, X, x_mask, sparse = False):
        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, x_mask)
        V = torch.cat([V, V_embed], dim = -1)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(x_mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = x_mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_edge_endpoints(h_E, h_V, E_idx)
            h_E = layer(h_E, h_EV, E_idx, mask_E=x_mask, mask_attend=mask_attend)

        h_E = self.W_out(h_E)

        if not sparse:
            return h_E, E_idx
        else:
            n_batch, n_nodes, k = h_E.shape[:-1]

            h_i_idx = E_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, k).unsqueeze(-1)
            h_j_idx = E_idx.unsqueeze(-1)
            batch_num = torch.arange(n_batch).view(n_batch, 1, 1, 1).expand(-1, n_nodes, k, -1)
            ij = torch.cat([batch_num, h_i_idx, h_j_idx], dim=-1)

            flat_ij = ij.view(-1, 3).transpose(0,1)
            flat_h_E = h_E.view(-1, self.output_dim).float()

            etab = torch.sparse.FloatTensor(flat_ij, flat_h_E)

            return etab, E_idx
