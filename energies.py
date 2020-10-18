from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from struct2seq.self_attention import *
from struct2seq.protein_features import ProteinFeatures
from struct2seq.struct2seq import Struct2Seq
from batched_term_transformer.term_features import IndexDiffEncoding

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

class RawSelfEnergies(Struct2Seq):
    def __init__(self, num_letters, node_features, edge_features, input_dim,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=20, k_neighbors=30, protein_features='full', augment_eps=0.,
        dropout=0.1, forward_attention_decoder=True, use_mpnn=False,
        output_dim = 20):

        super(RawSelfEnergies, self).__init__(num_letters, node_features, edge_features,
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

        return h_V, h_E, E_idx


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

        h_EV = self.W_out(h_E)#/30
        h_EV = merge_duplicate_edges(h_EV, E_idx)

        if not sparse:
            return h_EV, E_idx
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


class MultiChainProteinFeatures(ProteinFeatures):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, features_type='full', augment_eps=0., dropout=0.1):
        """ Extract protein features """
        super(MultiChainProteinFeatures, self).__init__(edge_features, node_features, 
            num_positional_embeddings=16, num_rbf=16, top_k=30, features_type='full', 
            augment_eps=0., dropout=0.1)

        # so uh this is designed to work on the batched TERMS
        # but if we just treat the whole sequence as one big TERM
        # the math is the same so i'm not gonna code a new module lol
        self.embeddings = IndexDiffEncoding(num_positional_embeddings)

    def forward(self, X, chain_idx, mask):
        """ Featurize coordinates as an attributed graph """

        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        # Build k-Nearest Neighbors graph
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx, mask_neighbors = self._dist(X_ca, mask)

        # Pairwise features
        AD_features, O_features = self._orientations_coarse(X_ca, E_idx)
        RBF = self._rbf(D_neighbors)

        # Pairwise embeddings
        # we unsqueeze to generate "1 TERM" per sequence, 
        # then squeeze it back to get rid of it
        E_positional = self.embeddings(E_idx.unsqueeze(1), chain_idx).squeeze(1)

        if self.features_type == 'coarse':
            # Coarse backbone features
            V = AD_features
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'hbonds':
            # Hydrogen bonds and contacts
            neighbor_HB = self._hbonds(X, E_idx, mask_neighbors)
            neighbor_C = self._contacts(D_neighbors, E_idx, mask_neighbors)
            # Dropout
            neighbor_C = self.dropout(neighbor_C)
            neighbor_HB = self.dropout(neighbor_HB)
            # Pack
            V = mask.unsqueeze(-1) * torch.ones_like(AD_features)
            neighbor_C = neighbor_C.expand(-1,-1,-1, int(self.num_positional_embeddings / 2))
            neighbor_HB = neighbor_HB.expand(-1,-1,-1, int(self.num_positional_embeddings / 2))
            E = torch.cat((E_positional, neighbor_C, neighbor_HB), -1)
        elif self.features_type == 'full':
            # Full backbone angles
            V = self._dihedrals(X)
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'dist':
            # Full backbone angles
            V = self._dihedrals(X)
            E = torch.cat((E_positional, RBF), -1)

        # Embed the nodes
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        # DEBUG
        # U = (np.nan * torch.zeros(X.size(0),X.size(1),X.size(1),3)).scatter(2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), E[:,:,:,:3])
        # plt.imshow(U.data.numpy()[0,:,:,0])
        # plt.show()
        # exit(0)
        return V, E, E_idx


class MultiChainPairEnergies(PairEnergies):
    def __init__(self, num_letters, node_features, edge_features, input_dim,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=20, k_neighbors=30, protein_features='full', augment_eps=0.,
        dropout=0.1, forward_attention_decoder=True, use_mpnn=False,
        output_dim = 20 * 20):
        """ Graph labeling network """
        super(MultiChainPairEnergies, self).__init__(num_letters, node_features, edge_features, 
            input_dim, hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
            vocab=20, k_neighbors=30, protein_features='full', augment_eps=0.,
            dropout=0.1, forward_attention_decoder=True, use_mpnn=False,
            output_dim = 20 * 20)
    
        # Featurization layers
        self.features = MultiChainProteinFeatures(
            node_features, edge_features, top_k=k_neighbors,
            features_type=protein_features, augment_eps=augment_eps,
            dropout=dropout
        )


    def forward(self, V_embed, X, x_mask, chain_idx, sparse = False):
        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, chain_idx, x_mask)
        V = torch.cat([V, V_embed], dim = -1)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(x_mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = x_mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_EV = cat_edge_endpoints(h_E, h_V, E_idx)
            h_E = layer(h_E, h_EV, E_idx, mask_E=x_mask, mask_attend=mask_attend)

        h_EV = self.W_out(h_E)
        h_EV = merge_duplicate_edges(h_EV, E_idx)

        if not sparse:
            return h_EV, E_idx
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
   

class PairEnergiesFullGraph(nn.Module):
    def __init__(self, num_letters, node_features, edge_features, input_dim,
        hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
        vocab=20, k_neighbors=30, protein_features='full', augment_eps=0.,
        dropout=0.1, forward_attention_decoder=True, use_mpnn=False,
        output_dim = 20 * 20):
        """ Graph labeling network """
        super(PairEnergiesFullGraph, self).__init__()

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
        edge_layer = EdgeTransformerLayer
        node_layer = TransformerLayer

        # Encoder layers
        self.edge_encoder = nn.ModuleList([
            edge_layer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.node_encoder = nn.ModuleList([
            node_layer(hidden_dim, hidden_dim*2, dropout=dropout)
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
        for edge_layer, node_layer in zip(self.edge_encoder, self.node_encoder):
            h_EV_edges = cat_edge_endpoints(h_E, h_V, E_idx)
            h_E_new = edge_layer(h_E, h_EV_edges, E_idx, mask_E=x_mask, mask_attend=mask_attend)

            h_EV_nodes = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = node_layer(h_V, h_EV_nodes, mask_V = x_mask, mask_attend = mask_attend)
            h_E = h_E_new

        h_E = self.W_out(h_E)
        h_E = merge_duplicate_edges(h_E, E_idx)

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
