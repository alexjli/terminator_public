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
    def __init__(self, num_hidden, num_in, num_heads=4):
        super(TERMAttention, self).__init__()
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

    def forward(self, src, mask_attend = None, src_key_mask = None):
        query, key, value = src, src, src

        n_batches, n_terms, n_aa = query.shape[:3]
        n_heads = self.num_heads

        assert self.num_hidden % n_heads == 0

        d = self.num_hidden // n_heads
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
            mask = mask.byte()
            # Masked softmax
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        src_update = torch.matmul(attend, V).transpose(2,3).contiguous()
        src_update = src_update.view([n_batches, n_terms, n_aa, self.num_hidden])
        src_update = self.W_O(src_update)
        return src_update

class TERMTransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_heads=4, dropout=0.1):
        super(TERMTransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])

        self.attention = TERMAttention(num_hidden, num_hidden, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, src, src_mask=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # Self-attention
        dsrc = self.attention(src, mask_attend = mask_attend)
        src = self.norm[0](src + self.dropout(dsrc))

        # Position-wise feedforward
        dsrc = self.dense(src)
        src = self.norm[1](src + self.dropout(dsrc))

        if src_mask is not None:
            src_mask = src_mask.unsqueeze(-1)
            src = src_mask * src
        return src

# from pytorch docs for 1.5
class TERMTransformer(nn.Module):
    def __init__(self, transformer, num_layers = 4):
        super(TERMTransformer, self).__init__()
        self.layers = _get_clones(transformer, num_layers)

    def forward(self, src, mask_attend = None):
        output = src

        for mod in self.layers:
            output = mod(output, mask_attend = mask_attend)

        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
