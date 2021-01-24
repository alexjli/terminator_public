import torch
import torch.nn as nn
from gvp import *

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx, nv_nodes, nv_neighbors):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return vs_concat(h_neighbors, h_nodes, nv_neighbors, nv_nodes)

def gather_term_nodes(nodes, neighbor_idx):
    # Features [B,T,N,C] at Neighbor indices [B,T,N,K] => [B,T,N,K,C]
    # Flatten and expand indices per batch [B,T,N,K] => [B,T,NK] => [B,T,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], neighbor_idx.shape[1], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, -1, nodes.size(3))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 2, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:4] + [-1])
    return neighbor_features

def gather_term_edges(edges, neighbor_idx):
    # Features [B,T,N,N,C] at Neighbor indices [B,T,N,K] => Neighbor features [B,T,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 3, neighbors)
    return edge_features

def cat_term_edge_endpoints(h_edges, h_nodes, E_idx, n_node, n_edge):
    # Neighbor indices E_idx [B,T,N,K]
    # Edge features h_edges [B,T,N,N,C]
    # Node features h_nodes [B,T,N,C]
    n_batches, n_terms, n_nodes, k = E_idx.shape

    h_i_idx = E_idx[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, k).contiguous()
    h_j_idx = E_idx

    h_i = gather_term_nodes(h_nodes, h_i_idx)
    h_j = gather_term_nodes(h_nodes, h_j_idx)

    #e_ij = gather_edges(h_edges, E_idx)
    e_ij = h_edges

    # output features [B, T, N, K, 3C]
    h_nn = vs_concat(vs_concat(h_i, h_j, n_node, n_node), e_ij, n_node * 2, n_edge)
    return h_nn


class TERMGVPEncoder(nn.Module):
    def __init__(self, hparams):
        super(TERMGVPEncoder, self).__init__()
        self.hparams = hparams
        node_features = hparams['hidden_dim']
        edge_features = hparams['hidden_dim']
        hidden_dim = hparams['hidden_dim']
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
            h_EV = cat_term_edge_endpoints(h_E, h_V, E_idx, self.nv, self.ev)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        return self.W_out(h_V)


class TERMGraphGVPEncoder(nn.Module):
    def __init__(self, hparams):
        """ Graph labeling network """
        super(TERMGraphGVPEncoder, self).__init__()

        self.hparams = hparams
        node_features = (hparams['hidden_dim']//2, hparams['hidden_dim']//2)
        edge_features = (hparams['hidden_dim']//2, hparams['hidden_dim']//2)
        hidden_dim = hparams['hidden_dim']
        hidden_features = (hparams['hidden_dim']//2, hparams['hidden_dim']//2)
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
            h_EV_nodes = cat_term_edge_endpoints(h_E, h_V, E_idx, self.nv, self.ev)
            h_V = node_layer(h_V, h_EV_nodes, mask_V = mask, mask_attend = mask_attend)

            # update edges using nodes
            h_EV_edges = cat_term_edge_endpoints(h_E, h_V, E_idx, self.nv, self.ev)
            h_E = edge_layer(h_E, h_EV_edges, mask_E=mask_attend, mask_attend=mask_attend)

        h_E = self.W_out(h_E)

        return h_V, h_E
