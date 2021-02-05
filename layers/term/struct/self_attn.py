import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy

from layers.s2s_modules import PositionWiseFeedForward, Normalize

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
