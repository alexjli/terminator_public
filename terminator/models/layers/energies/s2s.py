from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from terminator.models.layers.graph_features import *
from terminator.models.layers.utils import *
from terminator.models.layers.gvp import *
from terminator.models.layers.s2s_modules import *
from terminator.models.layers.struct2seq import Struct2Seq
#from terminator.models.struct2seq.protein_features import *


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
    def __init__(self, hparams):
        """ Graph labeling network """
        super(PairEnergies, self).__init__()
        self.hparams = hparams

        hdim = hparams['energies_hidden_dim']


        # Featurization layers
        self.features = S2SProteinFeatures(node_features = hdim, 
                                        edge_features = hdim, 
                                        top_k = hparams['k_neighbors'], 
                                        features_type = hparams['energies_protein_features'], 
                                        augment_eps = hparams['energies_augment_eps'], 
                                        dropout = hparams['energies_dropout'])

        # Embedding layers
        self.W_v = nn.Linear(hdim + hparams['energies_input_dim'], hdim, bias=True)
        self.W_e = nn.Linear(hdim, hdim, bias=True)
        layer = EdgeTransformerLayer if not hparams['energies_use_mpnn'] else EdgeMPNNLayer

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            layer(hdim, hdim*3, dropout=hparams['energies_dropout'])
            for _ in range(hparams['energies_encoder_layers'])
        ])

        self.W_out = nn.Linear(hdim, hparams['energies_output_dim'], bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, X, x_mask, V_embed = [], sparse = False):
        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, x_mask)
        if not isinstance(V_embed, list) or V_embed != []:
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
            flat_h_E = h_E.view(-1, self.hparams['energies_output_dim']).float()

            etab = torch.sparse.FloatTensor(flat_ij, flat_h_E)

            return etab, E_idx


class AblatedPairEnergies(nn.Module):
    def __init__(self, hparams):
        """ Graph labeling network """
        super(AblatedPairEnergies, self).__init__()
        hdim = hparams['energies_hidden_dim']
        self.hparams = hparams


        # Featurization layers
        self.features = S2SProteinFeatures(node_features = hdim, 
                                        edge_features = hdim, 
                                        top_k = hparams['k_neighbors'], 
                                        features_type = hparams['energies_protein_features'], 
                                        augment_eps = hparams['energies_augment_eps'], 
                                        dropout = hparams['energies_dropout'])


        self.k_neighbors = hparams['k_neighbors']
        self.W = nn.Linear(hparams['energies_input_dim'] * 2, hparams['energies_output_dim'])

    def forward(self, X, x_mask, V_embed, sparse = False):
        # Prepare node and edge embeddings
        _, _, E_idx = self.features(X, x_mask)

        h_nodes = V_embed

        h_i_idx = E_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, self.k_neighbors).contiguous()
        h_j_idx = E_idx

        h_i = gather_nodes(h_nodes, h_i_idx)
        h_j = gather_nodes(h_nodes, h_j_idx)

        h_E = torch.cat((h_i, h_j), -1)
        h_EV = self.W(h_E)

        return h_EV, E_idx

class AblatedPairEnergies_g(nn.Module):
    def __init__(self, hparams):
        """ Graph labeling network """
        super().__init__()
        hdim = hparams['energies_hidden_dim']
        self.hparams = hparams

        # Featurization layers
        self.features = MultiChainProteinFeatures(
            node_features = hdim, 
            edge_features = hdim, 
            top_k = hparams['k_neighbors'], 
            features_type = hparams['energies_protein_features'], 
            augment_eps = hparams['energies_augment_eps'], 
            dropout = hparams['energies_dropout']
        )
 
        self.k_neighbors = hparams['k_neighbors']
        self.W = nn.Linear(hparams['energies_input_dim'] * 3, hparams['energies_output_dim'])

    def forward(self, V_embed, E_embed, X, x_mask, chain_idx, sparse = False):
        # Prepare node and edge embeddings
        _, _, E_idx = self.features(X, chain_idx, x_mask)
        E_embed_neighbors = gather_edges(E_embed, E_idx)

        h_E = cat_edge_endpoints(E_embed_neighbors, V_embed, E_idx)
        h_EV = self.W(h_E)

        return h_EV, E_idx


class MultiChainPairEnergies(PairEnergies):
    def __init__(self, hparams):
        """ Graph labeling network """
        super(MultiChainPairEnergies, self).__init__(hparams)
        self.hparams = hparams

        hdim = hparams['energies_hidden_dim']

    
        # Featurization layers
        self.features = MultiChainProteinFeatures(
            node_features = hdim, 
            edge_features = hdim, 
            top_k = hparams['k_neighbors'], 
            features_type = hparams['energies_protein_features'], 
            augment_eps = hparams['energies_augment_eps'], 
            dropout = hparams['energies_dropout']
        )

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


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


class MultiChainPairEnergies_g(PairEnergies):
    def __init__(self, hparams):
        """ Graph labeling network """
        super(MultiChainPairEnergies_g, self).__init__(hparams)
        self.hparams = hparams

        hdim = hparams['energies_hidden_dim']
    
        # Featurization layers
        self.features = MultiChainProteinFeatures(
            node_features = hdim, 
            edge_features = hdim, 
            top_k = hparams['k_neighbors'], 
            features_type = hparams['energies_protein_features'], 
            augment_eps = hparams['energies_augment_eps'], 
            dropout = hparams['energies_dropout']
        )
        
        self.W_e = nn.Linear(hdim + hparams['energies_input_dim'], hdim, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self, V_embed, E_embed, X, x_mask, chain_idx, sparse = False):
        # Prepare node and edge embeddings

        V, E, E_idx = self.features(X, chain_idx, x_mask)
        V = torch.cat([V, V_embed], dim = -1)
        E_embed_neighbors = gather_edges(E_embed, E_idx)
        E = torch.cat([E, E_embed_neighbors], dim = -1)

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
    def __init__(self, hparams):
        """ Graph labeling network """
        super(PairEnergiesFullGraph, self).__init__()
        self.hparams = hparams

        hdim = hparams['energies_hidden_dim']

        # Hyperparameters
        self.node_features = hdim
        self.edge_features = hdim
        self.input_dim = hdim
        hidden_dim = hdim
        output_dim = hparams['energies_output_dim']
        dropout = hparams['energies_dropout']
        num_encoder_layers = hparams['energies_encoder_layers']

        # Featurization layers
        self.features = MultiChainProteinFeatures(
            node_features = hdim, 
            edge_features = hdim, 
            top_k = hparams['k_neighbors'], 
            features_type = hparams['energies_protein_features'], 
            augment_eps = hparams['energies_augment_eps'], 
            dropout = hparams['energies_dropout']
        )
        
        # Embedding layers
        self.W_v = nn.Linear(hdim + hparams['energies_input_dim'], hdim, bias=True)
        self.W_e = nn.Linear(hdim + hparams['energies_input_dim'], hdim, bias=True)
        edge_layer = EdgeTransformerLayer if not hparams['energies_use_mpnn'] else EdgeMPNNLayer
        node_layer = TransformerLayer if not hparams['energies_use_mpnn'] else NodeMPNNLayer

        # Encoder layers
        self.edge_encoder = nn.ModuleList([
            edge_layer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.node_encoder = nn.ModuleList([
            node_layer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        if "node_self_sub" in hparams.keys():
            self.W_proj = nn.Linear(hidden_dim, 20)

        self.W_out = nn.Linear(hidden_dim, output_dim, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V_embed, E_embed, X, x_mask, chain_idx, sparse = False):
        # Prepare node and edge embeddings
        if self.hparams['energies_input_dim'] != 0:
            V, E, E_idx = self.features(X, chain_idx, x_mask)

            if not self.hparams['use_coords']: # this is hacky/inefficient but i am lazy
                V = torch.zeros_like(V)
                E = torch.zeros_like(E)

            if torch.isnan(V).any() or torch.isnan(E).any() or torch.isnan(E_idx).any():
                raise RuntimeError("nan found during struct2seq feature generation")
            h_V = self.W_v(torch.cat([V, V_embed], dim = -1))
            if torch.isnan(h_V).any():
                raise RuntimeError("nan after lin comb of V V_embed")
            E_embed_neighbors = gather_edges(E_embed, E_idx)
            h_E = self.W_e(torch.cat([E, E_embed_neighbors], dim = -1))
            if torch.isnan(h_E).any():
                raise RuntimeError("nan after lin comb of E E_embed")

        else:
            V, E, E_idx = self.features(X, chain_idx, x_mask)
            h_V = self.W_v(V)
            h_E = self.W_e(E)

        #if torch.isnan(h_V).any() or torch.isnan(h_E).any() or torch.isnan(E_idx).any():
        #    raise RuntimeError("nan found at net1/struct2seq intersection")

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(x_mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = x_mask.unsqueeze(-1) * mask_attend
        for edge_layer, node_layer in zip(self.edge_encoder, self.node_encoder):
            h_EV_edges = cat_edge_endpoints(h_E, h_V, E_idx)
            h_E = edge_layer(h_E, h_EV_edges, E_idx, mask_E=x_mask, mask_attend=mask_attend)
            #if torch.isnan(h_E).any():
            #    raise RuntimeError("nan found after edge layer")

            h_EV_nodes = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = node_layer(h_V, h_EV_nodes, mask_V = x_mask, mask_attend = mask_attend)
            #if torch.isnan(h_V).any():
            #    raise RuntimeError("nan found after node layer")


        h_E = self.W_out(h_E)
        n_batch, n_res, k, out_dim = h_E.shape
        h_E = h_E.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
        h_E = merge_duplicate_pairE(h_E, E_idx)

        if "node_self_sub" in self.hparams.keys():
            h_V = self.W_proj(h_V)
            h_E[..., 0, :, :] = torch.diag_embed(h_V, dim1=-2, dim2=-1)

        h_E = h_E.view(n_batch, n_res, k, out_dim)

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
