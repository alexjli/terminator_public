import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# TODO: imports
from terminator.models.layers.s2s_modules import (Normalize, PositionWiseFeedForward)
from terminator.models.layers.utils import *


class TERMNeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4):
        super(TERMNeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        mask_attn_dev = mask_attend.device
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(mask_attn_dev))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend.float() * attend
        return attend

    def forward(self, h_V, h_EV, mask_attend=None, src_key_mask=None):
        """ Self-attention, graph-structured O(Nk)
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_hidden]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V:            Node update
        """

        # Queries, Keys, Values
        n_batch, n_terms, n_nodes, n_neighbors = h_EV.shape[:4]
        n_heads = self.num_heads

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_batch, n_terms, n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_EV).view([n_batch, n_terms, n_nodes, n_neighbors, n_heads, d, 1])
        V = self.W_V(h_EV).view([n_batch, n_terms, n_nodes, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view([n_batch, n_terms, n_nodes, n_neighbors, n_heads]).transpose(-2, -1)
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            mask = mask_attend.unsqueeze(3).expand(-1, -1, -1, n_heads, -1)
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(3, 4))
        h_V_update = h_V_update.view([n_batch, n_terms, n_nodes, self.num_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update


class S2STERMTransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_in=None, num_heads=4, dropout=0.1):
        super(S2STERMTransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.attention = TERMNeighborAttention(num_hidden, num_hidden * 2, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # Self-attention
        dh = self.attention(h_V, h_E, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class S2STERMTransformerEncoder(nn.Module):
    def __init__(self, hparams):
        super(S2STERMTransformerEncoder, self).__init__()
        self.hparams = hparams
        node_features = hparams['term_hidden_dim']
        edge_features = hparams['term_hidden_dim']
        hidden_dim = hparams['term_hidden_dim']
        num_heads = hparams['term_heads']
        dropout = hparams['transformer_dropout']
        num_encoder_layers = hparams['term_layers']

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        layer = S2STERMTransformerLayer

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [layer(hidden_dim, num_heads, dropout=dropout) for _ in range(num_encoder_layers)])

        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, V, E, E_idx, mask):
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        return self.W_out(h_V)


class TERMEdgeEndpointAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4):
        super(TERMEdgeEndpointAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        mask_attn_dev = mask_attend.device
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(mask_attn_dev))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend.float() * attend
        return attend

    def forward(self, h_E, h_EV, E_idx, mask_attend=None):
        """ Self-attention, graph-structured O(Nk)
        Args:
            h_E:            Edge features               [N_batch, N_nodes, K, N_hidden]
            h_EV:           Edge + endpoint features    [N_batch, N_nodes, K, N_hidden * 3]
            mask_attend:    Mask for attention          [N_batch, N_nodes, K]
        Returns:
            h_E_update      Edge update
        """

        # Queries, Keys, Values
        n_batch, n_terms, n_aa, n_neighbors = h_E.shape[:-1]
        n_heads = self.num_heads

        assert self.num_hidden % n_heads == 0

        d = self.num_hidden // n_heads
        Q = self.W_Q(h_E).view([n_batch, n_terms, n_aa, n_neighbors, n_heads, d]).transpose(3, 4)
        K = self.W_K(h_EV).view([n_batch, n_terms, n_aa, n_neighbors, n_heads, d]).transpose(3, 4)
        V = self.W_V(h_EV).view([n_batch, n_terms, n_aa, n_neighbors, n_heads, d]).transpose(3, 4)

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d)

        if mask_attend is not None:
            # we need to reshape the src key mask for edge-edge attention
            # expand to num_heads
            mask = mask_attend.unsqueeze(3).expand(-1, -1, -1, n_heads, -1).unsqueeze(-1).double()
            mask_t = mask.transpose(-2, -1)
            # perform outer product
            mask = mask @ mask_t
            mask = mask.bool()
            # Masked softmax
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_E_update = torch.matmul(attend, V).transpose(3, 4).contiguous()
        h_E_update = h_E_update.view([n_batch, n_terms, n_aa, n_neighbors, self.num_hidden])
        h_E_update = self.W_O(h_E_update)
        # nondirected edges are actually represented as two directed edges in opposite directions
        # to allow information flow, merge these duplicate edges
        h_E_update = merge_duplicate_term_edges(h_E_update, E_idx)
        return h_E_update


class TERMEdgeTransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.1):
        super(TERMEdgeTransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.attention = TERMEdgeEndpointAttention(num_hidden, num_in, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_E, h_EV, E_idx, mask_E=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # Self-attention
        dh = self.attention(h_E, h_EV, E_idx, mask_attend)
        h_E = self.norm[0](h_E + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_E)
        h_E = self.norm[1](h_E + self.dropout(dh))

        if mask_E is not None:
            mask_E = mask_E.unsqueeze(-1)
            h_E = mask_E * h_E
        return h_E


class TERMNodeMPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(TERMNodeMPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(F.relu(self.W2(F.relu(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class TERMEdgeMPNNLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(TERMEdgeMPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.W1 = nn.Linear(num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_E, h_EV, E_idx, mask_E=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        dh = self.W3(F.relu(self.W2(F.relu(self.W1(h_EV)))))
        dh = merge_duplicate_term_edges(dh, E_idx)
        if mask_attend is not None:
            dh = mask_attend.unsqueeze(-1) * dh

        h_E = self.norm[0](h_E + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_E)
        h_E = self.norm[1](h_E + self.dropout(dh))

        if mask_E is not None:
            mask_E = mask_E.unsqueeze(-1)
            h_E = mask_E * h_E
        return h_E


class TERMGraphTransformerEncoder(nn.Module):
    def __init__(self, hparams):
        """ Graph labeling network """
        super(TERMGraphTransformerEncoder, self).__init__()

        self.hparams = hparams
        node_features = hparams['term_hidden_dim']
        edge_features = hparams['term_hidden_dim']
        hidden_dim = hparams['term_hidden_dim']
        num_heads = hparams['term_heads']
        dropout = hparams['transformer_dropout']
        num_encoder_layers = hparams['term_layers']

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.input_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        edge_layer = TERMEdgeTransformerLayer if not hparams['term_use_mpnn'] else TERMEdgeMPNNLayer
        node_layer = S2STERMTransformerLayer if not hparams['term_use_mpnn'] else TERMNodeMPNNLayer

        # Encoder layers
        self.edge_encoder = nn.ModuleList(
            [edge_layer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_encoder_layers)])
        self.node_encoder = nn.ModuleList([
            node_layer(hidden_dim, num_in=hidden_dim * 2, num_heads=num_heads, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V, E, E_idx, mask):
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_term_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for edge_layer, node_layer in zip(self.edge_encoder, self.node_encoder):
            h_EV_edges = cat_term_edge_endpoints(h_E, h_V, E_idx)
            h_E = edge_layer(h_E, h_EV_edges, E_idx, mask_E=mask_attend, mask_attend=mask_attend)

            h_EV_nodes = cat_term_neighbors_nodes(h_V, h_E, E_idx)
            h_V = node_layer(h_V, h_EV_nodes, mask_V=mask, mask_attend=mask_attend)

        h_E = self.W_out(h_E)
        h_E = merge_duplicate_term_edges(h_E, E_idx)

        return h_V, h_E


class TERMGraphTransformerEncoder_cnkt(nn.Module):
    def __init__(self, hparams):
        """ Graph labeling network """
        super(TERMGraphTransformerEncoder_cnkt, self).__init__()

        self.hparams = hparams
        node_features = hparams['term_hidden_dim']
        edge_features = hparams['term_hidden_dim']
        hidden_dim = hparams['term_hidden_dim']
        num_heads = hparams['term_heads']
        dropout = hparams['transformer_dropout']
        num_encoder_layers = hparams['term_layers']

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.input_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        edge_layer = TERMEdgeTransformerLayer if not hparams['term_use_mpnn'] else TERMEdgeMPNNLayer
        node_layer = S2STERMTransformerLayer if not hparams['term_use_mpnn'] else TERMNodeMPNNLayer

        # Encoder layers
        self.edge_encoder = nn.ModuleList(
            [edge_layer(hidden_dim, hidden_dim * 5, dropout=dropout) for _ in range(num_encoder_layers)])
        self.node_encoder = nn.ModuleList([
            node_layer(hidden_dim, num_in=hidden_dim * 4, num_heads=num_heads, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V, E, E_idx, mask, contact_idx):
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_term_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for edge_layer, node_layer in zip(self.edge_encoder, self.node_encoder):
            h_EV_edges = cat_term_edge_endpoints(h_E, h_V, E_idx)
            h_EV_edges = cat_term_edge_endpoints(h_EV_edges, contact_idx, E_idx)
            h_E = edge_layer(h_E, h_EV_edges, E_idx, mask_E=mask_attend, mask_attend=mask_attend)

            h_EI = cat_term_edge_endpoints(h_E, contact_idx, E_idx)
            h_EV_nodes = cat_term_neighbors_nodes(h_V, h_EI, E_idx)
            h_V = node_layer(h_V, h_EV_nodes, mask_V=mask, mask_attend=mask_attend)

        h_E = self.W_out(h_E)
        h_E = merge_duplicate_term_edges(h_E, E_idx)

        return h_V, h_E
