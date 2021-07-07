import torch
import torch.nn as nn

from layers.gvp import *
from layers.utils import gather_nodes, gather_term_nodes, cat_gvp_term_edge_endpoints

class TERMGVPEncoder(nn.Module):
    def __init__(self, hparams):
        super(TERMGVPEncoder, self).__init__()
        self.hparams = hparams
        node_features = hparams['term_hidden_dim']
        edge_features = hparams['term_hidden_dim']
        hidden_dim = hparams['term_hidden_dim']
        dropout = hparams['transformer_dropout']
        num_encoder_layers = hparams['term_layers']

        # Hyperparameters
        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_dim
        self.ev, self.es = edge_features

        # Embedding layers
        self.W_v = GVP(vi=self.nv, vo=self.hv, si=self.hs * 2, so=self.hs,
                        nls=None, nlv=None)
        self.W_e = GVP(vi=self.ev, vo=self.hv, si=self.hs, so=self.hs,
                        nls=None, nlv=None)
        layer = GVPNodeLayer

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            layer(vec_in = self.nv + self.ev, num_hidden = hidden_dim, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.W_out = GVP(vi=self.ev, vo=self.ev, si=self.hs, so=self.hs,
                        nls=None, nlv=None)

    def forward(self, V, E, E_idx, mask):
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_gvp_term_edge_endpoints(h_E, h_V, E_idx, self.nv, self.ev)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        return self.W_out(h_V)

class TERMGraphGVPEncoder(nn.Module):
    def __init__(self, hparams):
        """ Graph labeling network """
        super(TERMGraphGVPEncoder, self).__init__()

        self.hparams = hparams
        node_features = (hparams['term_hidden_dim']//2, hparams['term_hidden_dim']//2)
        edge_features = (hparams['term_hidden_dim']//2, hparams['term_hidden_dim']//2)
        hidden_dim = hparams['term_hidden_dim']
        hidden_features = (hparams['term_hidden_dim']//2, hparams['term_hidden_dim']//2)
        dropout = hparams['transformer_dropout']
        num_encoder_layers = hparams['term_layers']

        # Hyperparameters
        self.nv, self.ns = node_features
        self.hv, self.hs = hidden_features
        self.ev, self.es = edge_features
        
        # Embedding layers
        self.W_v = GVP(vi=self.nv, vo=self.hv, si=self.hs + hidden_dim, so=self.hs,
                        nls=None, nlv=None)
        self.W_e = GVP(vi=self.ev, vo=self.hv, si=self.hs, so=self.hs,
                        nls=None, nlv=None)
        edge_layer = GVPEdgeLayer
        node_layer = GVPNodeLayer

        # Encoder layers
        self.edge_encoder = nn.ModuleList([
            edge_layer(nv = self.nv, 
                       ns = self.ns, 
                       ev = 2*self.nv + self.ev, 
                       es = 2*self.ns + self.es,
                       dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.node_encoder = nn.ModuleList([
            node_layer(nv = self.nv, 
                       ns = self.ns, 
                       ev = 2*self.nv + self.ev, 
                       es = 2*self.ns + self.es,
                       dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.W_out = GVP(vi=self.ev, vo=self.ev, si=self.hs, so=self.hs,
                        nls=None, nlv=None)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V, E, E_idx, mask):
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_term_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for edge_layer, node_layer in zip(self.edge_encoder, self.node_encoder):
            # update nodes using edges
            h_EV_nodes = cat_gvp_term_edge_endpoints(h_E, h_V, E_idx, self.nv, self.ev)
            h_V = node_layer(h_V, h_EV_nodes, mask_V = mask, mask_attend = mask_attend)

            # update edges using nodes
            h_EV_edges = cat_gvp_term_edge_endpoints(h_E, h_V, E_idx, self.nv, self.ev)
            h_E = edge_layer(h_E, h_EV_edges, mask_E=mask_attend, mask_attend=mask_attend)

        h_E = self.W_out(h_E)

        return h_V, h_E
