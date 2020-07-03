from nets import CondenseMSA
from struct2seq.energies import *
import torch
import torch.nn as nn
import numpy as np

class TERMinator(nn.Module):
    def __init__(self):
        super(TERMinator, self).__init__()
        self.bot = CondenseMSA(hidden_dim = 32, num_features = 7, filter_len = 3, num_blocks = 4, nheads = 4, device = 'cpu')
        self.top = PairEnergies(num_letters = 20, node_features = 32, edge_features = 32, input_dim = 32, hidden_dim = 64, k_neighbors=5)

    def forward(self, msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        out = self.top(condense, X, x_mask)
        return out
