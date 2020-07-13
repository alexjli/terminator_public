from nets import CondenseMSA
from struct2seq.energies import *
import torch
import torch.nn as nn
import numpy as np

class TERMinator(nn.Module):
    def __init__(self, device = 'cuda:0'):
        super(TERMinator, self).__init__()
        self.dev = device
        self.bot = CondenseMSA(hidden_dim = 32, num_features = 7, filter_len = 3, num_blocks = 4, nheads = 4, device = self.dev)
        self.top = PairEnergies(num_letters = 20, node_features = 32, edge_features = 32, input_dim = 32, hidden_dim = 64, k_neighbors=5).to(self.dev)

    ''' Negative log psuedo-likelihood '''
    ''' Averaged nlpl per residue, across batches '''
    def _nlpl(self, etab, E_idx, seqs, x_mask):
        n_batch, L, k, _ = etab.shape
        etab = etab.unsqueeze(-1).view(n_batch, L, k, 22, 22)
        # idx matrix to gather the identity at all other residues given a residue of focus
        E_aa = torch.gather(seqs.unsqueeze(-1).expand(-1, -1, k), 1, E_idx)
        E_aa = E_aa.view(list(E_idx.shape) + [1,1]).expand(-1, -1, -1, 22, -1)
        # gather the 22 energies for each edge based on E_aa
        edge_nrgs = torch.gather(etab, 4, E_aa).squeeze(-1)
        # get the nrg of for 22 possible aa identities at each position
        aa_nrgs = torch.sum(edge_nrgs, dim = 2)
        # convert energies to probabilities
        all_aa_probs = torch.softmax(aa_nrgs, dim = 2)
        # get the probability of the sequence
        seqs_probs = torch.gather(all_aa_probs, 2, seqs.unsqueeze(-1)).squeeze(-1)
        # convert to nlpl
        log_probs = torch.log(seqs_probs) * x_mask # zero out positions that don't have residues
        n_res = torch.sum(x_mask, dim=-1)
        nlpl = torch.sum(log_probs, dim=-1)/n_res
        nlpl = -torch.mean(nlpl)
        return nlpl

    def forward(self,
                msas,
                features,
                seq_lens,
                focuses,
                term_lens,
                src_key_mask,
                X,
                x_mask,
                sequence):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        etab, E_idx = self.top(condense, X, x_mask)
        nlpl = self._nlpl(etab, E_idx, sequence, x_mask)
        return nlpl

    def potts(self,
              msas,
              features,
              seq_lens,
              focuses,
              term_lens,
              src_key_mask,
              X,
              x_mask,
              sparse = False):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        etab, E_idx = self.top(condense, X, x_mask, sparse = sparse)
        return etab, E_idx
