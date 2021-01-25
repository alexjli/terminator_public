import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from struct2seq.self_attention import *


import numpy as np
import copy

# pads both dims 1 and 2 to max length
def pad_sequence_12(sequences, padding_value=0):
    n_batches = len(sequences)
    out_dims = list(sequences[0].size())
    dim1, dim2 = 0,1
    max_dim1 = max([s.size(dim1) for s in sequences])
    max_dim2 = max([s.size(dim2) for s in sequences])
    out_dims[dim1] = max_dim1
    out_dims[dim2] = max_dim2
    out_dims = [n_batches] + out_dims
    #print(out_dims)

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        len1 = tensor.size(0)
        len2 = tensor.size(1)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :len1, :len2, ...] = tensor

    return out_tensor

# The following gather functions
def gather_nodes(nodes, neighbor_idx):
    # Features [B,T,N,C] at Neighbor indices [B,T,N,K] => [B,T,N,K,C]
    # Flatten and expand indices per batch [B,T,N,K] => [B,T,NK] => [B,T,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], neighbor_idx.shape[1], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, -1, nodes.size(3))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 2, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:5] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class BatchifyTERM(nn.Module):
    def __init__(self):
        super(BatchifyTERM, self).__init__()

    def forward(self, batched_flat_terms, term_lens):
        n_batches = batched_flat_terms.shape[0]
        flat_terms = torch.unbind(batched_flat_terms)
        list_terms = [torch.split(flat_terms[i], term_lens[i]) for i in range(n_batches)]
        padded_terms = [pad_sequence(terms) for terms in list_terms]
        padded_terms = [term.transpose(0,1) for term in padded_terms]
        batchify = pad_sequence_12(padded_terms)
        return batchify


class TERMAttention(nn.Module):
    def __init__(self, hparams):
        super(TERMAttention, self).__init__()
        self.hparams = hparams

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'], bias=False)
        self.W_K = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'], bias=False)
        self.W_V = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'], bias=False)
        self.W_O = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'], bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        mask_attn_dev = mask_attend.device
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(mask_attn_dev))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend.float() * attend
        return attend

    def forward(self, src, mask_attend = None, src_key_mask = None):
        query, key, value = src, src, src

        n_batches, n_terms, n_aa = query.shape[:3]
        n_heads = self.hparams['term_heads']
        num_hidden = self.hparams['hidden_dim']

        assert num_hidden % n_heads == 0

        d = num_hidden // n_heads
        Q = self.W_Q(query).view([n_batches, n_terms, n_aa, n_heads, d]).transpose(2,3)
        K = self.W_K(key).view([n_batches, n_terms, n_aa, n_heads, d]).transpose(2,3)
        V = self.W_V(value).view([n_batches, n_terms, n_aa, n_heads, d]).transpose(2,3)

        attend_logits = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(d)

        if mask_attend is not None:
            # we need to reshape the src key mask for residue-residue attention
            # expand to num_heads
            mask = mask_attend.unsqueeze(2).expand(-1, -1, n_heads, -1).unsqueeze(-1).float()
            mask_t = mask.transpose(-2, -1)
            # perform outer product
            mask = mask @ mask_t
            mask = mask.bool()
            # Masked softmax
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        src_update = torch.matmul(attend, V).transpose(2,3).contiguous()
        src_update = src_update.view([n_batches, n_terms, n_aa, num_hidden])
        src_update = self.W_O(src_update)
        return src_update


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

    def forward(self, h_V, h_EV, mask_attend = None, src_key_mask = None):
        
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
        attend_logits = torch.matmul(Q, K).view([n_batch, n_terms, n_nodes, n_neighbors, n_heads]).transpose(-2,-1)
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            mask = mask_attend.unsqueeze(3).expand(-1, -1, -1, n_heads, -1)
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(3,4))
        h_V_update = h_V_update.view([n_batch, n_terms, n_nodes, self.num_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update


class TERMTransformerLayer(nn.Module):
    def __init__(self, hparams):
        super(TERMTransformerLayer, self).__init__()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['transformer_dropout'])
        self.norm = nn.ModuleList([Normalize(hparams['hidden_dim']) for _ in range(2)])

        self.attention = TERMAttention(hparams = self.hparams)
        self.dense = PositionWiseFeedForward(hparams['hidden_dim'], hparams['hidden_dim'] * 4)

    def forward(self, src, src_mask=None, mask_attend=None, checkpoint = False):
        """ Parallel computation of full transformer layer """
        # Self-attention
        if checkpoint:
            dsrc = torch.utils.checkpoint.checkpoint(self.attention, src, mask_attend)
        else:
            dsrc = self.attention(src, mask_attend = mask_attend)
        src = self.norm[0](src + self.dropout(dsrc))

        # Position-wise feedforward
        if checkpoint:
            dsrc = torch.utils.checkpoint.checkpoint(self.dense, src)
        else:
            dsrc = self.dense(src)
        src = self.norm[1](src + self.dropout(dsrc))

        if src_mask is not None:
            src_mask = src_mask.unsqueeze(-1)
            src = src_mask * src
        return src


class S2STERMTransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_heads=4, dropout=0.1):
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
        node_features = hparams['hidden_dim']
        edge_features = hparams['hidden_dim']
        hidden_dim = hparams['hidden_dim']
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
        self.encoder_layers = nn.ModuleList([
            layer(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, V, E, E_idx, mask):
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        return self.W_out(h_V)


class TERMGraphTransformerEncoder(nn.Module):
    def __init__(self, hparams):
        """ Graph labeling network """
        super(TERMGraphTransformerEncoder, self).__init__()

        self.hparams = hparams
        node_features = hparams['hidden_dim']
        edge_features = hparams['hidden_dim']
        hidden_dim = hparams['hidden_dim']
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
        edge_layer = TERMEdgeTransformerLayer
        node_layer = S2STERMTransformerLayer

        # Encoder layers
        self.edge_encoder = nn.ModuleList([
            edge_layer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.node_encoder = nn.ModuleList([
            node_layer(hidden_dim, num_heads, dropout=dropout)
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
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for edge_layer, node_layer in zip(self.edge_encoder, self.node_encoder):
            h_EV_edges = cat_term_edge_endpoints(h_E, h_V, E_idx)
            h_E_new = edge_layer(h_E, h_EV_edges, E_idx, mask_E=mask_attend, mask_attend=mask_attend)

            h_EV_nodes = cat_term_neighbors_nodes(h_V, h_E, E_idx)
            h_V = node_layer(h_V, h_EV_nodes, mask_V = mask, mask_attend = mask_attend)
            h_E = h_E_new

        h_E = self.W_out(h_E)
        h_E = merge_duplicate_term_edges(h_E, E_idx)

        return h_V, h_E


class TERMMatchAttention(nn.Module):
    def __init__(self, hparams):
        super(TERMMatchAttention, self).__init__()
        self.hparams = hparams

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'], bias=False)
        self.W_K = nn.Linear(hparams['hidden_dim']*2, hparams['hidden_dim'], bias=False)
        self.W_V = nn.Linear(hparams['hidden_dim']*2, hparams['hidden_dim'], bias=False)
        self.W_O = nn.Linear(hparams['hidden_dim'], hparams['hidden_dim'], bias=False)

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        mask_attn_dev = mask_attend.device
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).to(mask_attn_dev))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend.float() * attend
        return attend

    def forward(self, h_V, h_T, mask_attend = None, src_key_mask = None):
        n_batches, sum_term_len, n_matches = h_V.shape[:3]

        # append h_T onto h_V to form h_VT
        h_T_expand = h_T.unsqueeze(-2).expand(h_V.shape)
        h_VT = torch.cat([h_V, h_T_expand], dim = -1)
        query = h_V
        key = h_VT
        value = h_VT

        n_heads = self.hparams['term_heads']
        num_hidden = self.hparams['hidden_dim']

        assert num_hidden % n_heads == 0

        d = num_hidden // n_heads
        Q = self.W_Q(query).view([n_batches, sum_term_len, n_matches, n_heads, d]).transpose(2,3)
        K = self.W_K(key).view([n_batches, sum_term_len, n_matches, n_heads, d]).transpose(2,3)
        V = self.W_V(value).view([n_batches, sum_term_len, n_matches, n_heads, d]).transpose(2,3)

        attend_logits = torch.matmul(Q, K.transpose(-2,-1)) / np.sqrt(d)

        if mask_attend is not None:
            # we need to reshape the src key mask for residue-residue attention
            # expand to num_heads
            mask = mask_attend.unsqueeze(2).expand(-1, -1, n_heads, -1).unsqueeze(-1).float()
            mask_t = mask.transpose(-2, -1)
            # perform outer product
            mask = mask @ mask_t
            mask = mask.bool()
            # Masked softmax
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        src_update = torch.matmul(attend, V).transpose(2,3).contiguous()
        src_update = src_update.view([n_batches, sum_term_len, n_matches, num_hidden])
        src_update = self.W_O(src_update)
        return src_update


class TERMMatchTransformerLayer(nn.Module):
    def __init__(self, hparams):
        super(TERMMatchTransformerLayer, self).__init__()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['transformer_dropout'])
        self.norm = nn.ModuleList([Normalize(hparams['hidden_dim']) for _ in range(2)])

        self.attention = TERMMatchAttention(hparams = self.hparams)
        self.dense = PositionWiseFeedForward(hparams['hidden_dim'], hparams['hidden_dim'] * 4)

    def forward(self, src, target, src_mask=None, mask_attend=None, checkpoint = False):
        """ Parallel computation of full transformer layer """
        # Self-attention
        if checkpoint:
            dsrc = torch.utils.checkpoint.checkpoint(self.attention, src, target, mask_attend)
        else:
            dsrc = self.attention(src, target, mask_attend = mask_attend)
        src = self.norm[0](src + self.dropout(dsrc))

        # Position-wise feedforward
        if checkpoint:
            dsrc = torch.utils.checkpoint.checkpoint(self.dense, src)
        else:
            dsrc = self.dense(src)
        src = self.norm[1](src + self.dropout(dsrc))

        if src_mask is not None:
            src_mask = src_mask.unsqueeze(-1)
            src = src_mask * src
        return src


class TERMMatchTransformerEncoder(nn.Module):
    def __init__(self, hparams):
        super(TERMMatchTransformerEncoder, self).__init__()
        self.hparams = hparams
        
        # Hyperparameters
        hidden_dim = hparams['hidden_dim']
        self.hidden_dim = hidden_dim
        num_encoder_layers = hparams['matches_layers']
        num_heads = hparams['matches_num_heads']
        dropout = hparams['transformer_dropout']

        # Embedding layers
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_t = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_pool = nn.Linear(hidden_dim * 2, hidden_dim, bias = True)
        layer = TERMMatchTransformerLayer

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            layer(hparams)
            for _ in range(num_encoder_layers)
        ])

        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # lets try [CLS]-style pooling
        pool_token_init = torch.zeros(1, hidden_dim)
        torch.nn.init.xavier_uniform_(pool_token_init)
        self.pool_token = nn.Parameter(pool_token_init, requires_grad=True)

    def forward(self, V, target, mask):
        dev = V.device
        n_batches, sum_term_len, n_align = V.shape[:3]

        # embed each copy of the pool token with some information about the target ppoe
        pool = self.pool_token.view([1, 1, self.hidden_dim]).expand(n_batches, sum_term_len, -1)
        pool = torch.cat([pool, target], dim = -1)
        pool = self.W_pool(pool)
        pool = pool.unsqueeze(-2)

        V = torch.cat([pool, V], dim = -2)
        
        h_V = self.W_v(V)
        h_T = self.W_t(target)

        # Encoder is unmasked self-attention
        for idx, layer in enumerate(self.encoder_layers):
            h_V = layer(h_V, h_T, mask.unsqueeze(-1).float(), checkpoint = self.hparams['gradient_checkpointing'])

        h_V = self.W_out(h_V)
        return h_V[:, :, 0, :]

# from pytorch docs for 1.5
class TERMTransformer(nn.Module):
    def __init__(self, transformer, hparams):
        super(TERMTransformer, self).__init__()
        self.hparams = hparams
        self.layers = _get_clones(transformer, hparams['term_layers'])

    def forward(self, src, src_mask = None, mask_attend = None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask = src_mask, mask_attend = mask_attend)

        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
