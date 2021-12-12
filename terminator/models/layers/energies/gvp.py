from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph_features import *
from ..gvp import *
from ..utils import *


class GVPPairEnergies(nn.Module):
    def __init__(self, hparams):
        """ Graph labeling network """
        super(GVPPairEnergies, self).__init__()
        self.hparams = hparams
        #self.scalar_in = True #hparams['gvp_energies_scalar_in']

        node_features = (4, 50)  #(8,100)
        edge_features = (1, 32)
        hidden_dim = (8, 50)  #(16,100)

        # Featurization layers
        self.features = GVPProteinFeatures(node_features=node_features,
                                           edge_features=edge_features,
                                           top_k=hparams['k_neighbors'])

        # Hyperparameters
        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_dim
        self.ev, self.es = edge_features
        input_dim = hparams['energies_input_dim']
        dropout = hparams['energies_dropout']
        output_dim = hparams['energies_output_dim']
        num_encoder_layers = hparams['energies_encoder_layers']
        node_layer = GVPNodeLayer
        edge_layer = GVPEdgeLayer

        # Embedding layers
        #self.W_v = GVP(vi=self.nv*2, vo=self.hv, si=self.hs*2, so=self.hs,
        #                nls=None, nlv=None)
        #self.W_e = GVP(vi=self.ev*2, vo=self.ev, si=self.hs*2, so=self.hs,
        #                nls=None, nlv=None)
        self.W_v = GVP(vi=self.nv,
                       vo=self.hv,
                       si=self.ns + input_dim,
                       so=self.hs,
                       nls=None,
                       nlv=None)
        self.W_e = GVP(vi=self.ev,
                       vo=self.hv,
                       si=self.es + input_dim,
                       so=self.hs,
                       nls=None,
                       nlv=None)

        # Encoder layers
        self.edge_encoder = nn.ModuleList([
            edge_layer(nv=self.hv,
                       ns=self.hs,
                       ev=self.hv,
                       es=self.hs,
                       dropout=dropout) for _ in range(num_encoder_layers)
        ])
        self.node_encoder = nn.ModuleList([
            node_layer(nv=self.hv,
                       ns=self.hs,
                       ev=self.hv,
                       es=self.hs,
                       dropout=dropout) for _ in range(num_encoder_layers)
        ])

        self.W_out = GVP(vi=self.hv,
                         vo=0,
                         si=self.hs,
                         so=output_dim,
                         nls=None,
                         nlv=None)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V_term, E_term, X, x_mask, chain_idx):
        # get graph features
        V, E, E_idx = self.features(X, x_mask, chain_idx)

        E_term = gather_edges(E_term, E_idx)

        h_V = self.W_v(torch.cat([V, V_term], -1))
        h_E = self.W_e(torch.cat([E, E_term], -1))
        #h_V = self.W_v(vs_concat(V, V_term, self.nv, self.nv))
        #h_E = self.W_e(vs_concat(E, E_term, self.ev, self.ev))

        # Encoder is unmasked self-attention
        mask = x_mask  # hacky alias
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for edge_layer, node_layer in zip(self.edge_encoder,
                                          self.node_encoder):
            # update nodes using edges
            h_EV_nodes = cat_gvp_edge_endpoints(h_E, h_V, E_idx, self.nv,
                                                self.ev)
            h_V = node_layer(h_V,
                             h_EV_nodes,
                             mask_V=mask,
                             mask_attend=mask_attend)

            # update edges using nodes
            h_EV_edges = cat_gvp_edge_endpoints(h_E, h_V, E_idx, self.nv,
                                                self.ev)
            h_E = edge_layer(h_E,
                             h_EV_edges,
                             mask_E=mask_attend,
                             mask_attend=mask_attend)

        h_E = self.W_out(h_E)
        # merge directional edges features
        n_batch, n_res, k, out_dim = h_E.shape
        h_E = h_E.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
        h_E = merge_duplicate_pairE(h_E, E_idx)
        h_E = h_E.view(n_batch, n_res, k, out_dim)

        return h_E, E_idx
