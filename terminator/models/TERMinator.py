import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from terminator.utils.common import int_to_aa
from terminator.utils.loop_utils import nlcpl as _nlcpl
from terminator.utils.loop_utils import nlpl as _nlpl

from .layers.condense import CondenseMSA, MultiChainCondenseMSA_g
from .layers.energies.gvp import GVPPairEnergies
from .layers.energies.s2s import (AblatedPairEnergies,
                                  MultiChainPairEnergies_g, PairEnergies,
                                  PairEnergiesFullGraph)


class TERMinator(nn.Module):
    def __init__(self, hparams, device='cuda:0'):
        super(TERMinator, self).__init__()
        self.dev = device
        self.hparams = hparams
        self.bot = CondenseMSA(hparams=self.hparams, device=self.dev)

        if self.hparams["use_terms"]:
            self.hparams['energies_input_dim'] = self.hparams['term_hidden_dim']
        else:
            self.hparams['energies_input_dim'] = 0

        if self.hparams['struct2seq_linear']:
            self.top = AblatedPairEnergies(hparams=self.hparams).to(self.dev)
        else:
            self.top = PairEnergies(hparams=self.hparams).to(self.dev)

        print(self.bot.hparams['term_hidden_dim'], self.top.hparams['energies_hidden_dim'])

        self.prior = torch.zeros(20).view(1, 1, 20).to(self.dev)

    def update_prior(self, aa_nrgs, alpha=0.1):
        avg_nrgs = aa_nrgs.mean(dim=1).mean(dim=0).view(1, 1, 20)
        self.prior = self.prior * (1 - alpha) + avg_nrgs * alpha
        self.prior = self.prior.detach()

    def forward(self, msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, sequence, max_seq_len,
                ppoe):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len,
                                 ppoe)
        nlcpl, avg_prob, counter = _nlcpl(etab, E_idx, sequence, x_mask)
        return nlcpl, avg_prob, counter

    def potts(self, msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe):
        ''' compute the \'potts model parameters\' for the structure '''
        if self.hparams['use_terms']:
            condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask, max_seq_len, ppoe)
            etab, E_idx = self.top(X, x_mask, V_embed=condense)
        else:
            etab, E_idx = self.top(X, x_mask)
        return etab, E_idx

    def _get_self_etab(self, etab):
        ''' extract self nrgs from etab '''
        self_etab = etab[:, :, 0]
        n_batch, L = self_etab.shape[:2]
        self_etab = self_etab.unsqueeze(-1).view(n_batch, L, 20, 20)
        self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
        return self_nrgs

    def _percent(self, pred_seqs, true_seqs, x_mask):
        ''' with the predicted seq and actual seq, find the percent identity '''
        is_same = (pred_seqs == true_seqs)
        lens = torch.sum(x_mask, dim=-1)
        is_same *= x_mask.bool()
        num_same = torch.sum(is_same.float(), dim=-1)
        percent_same = num_same / lens
        return percent_same

    def _int_to_aa(self, int_seqs):
        ''' Convert int sequence sto its corresponding amino acid identities '''
        vec_i2a = np.vectorize(int_to_aa)
        char_seqs = vec_i2a(int_seqs).tolist()
        char_seqs = [''.join([aa for aa in seq]) for seq in char_seqs]
        return char_seqs


class MultiChainTERMinator_gcnkt(TERMinator):
    def __init__(self, hparams, device='cuda:0'):
        super(MultiChainTERMinator_gcnkt, self).__init__(hparams, device)
        self.dev = device
        self.hparams = hparams
        self.bot = MultiChainCondenseMSA_g(hparams, device=self.dev)

        if self.hparams["use_terms"]:
            self.hparams['energies_input_dim'] = self.hparams['term_hidden_dim']
        else:
            self.hparams['energies_input_dim'] = 0

        if hparams['struct2seq_linear']:
            self.top = AblatedPairEnergies_g(hparams).to(self.dev)
        elif hparams['energies_gvp']:
            self.top = GVPPairEnergies(hparams).to(self.dev)
        elif hparams['energies_full_graph']:
            self.top = PairEnergiesFullGraph(hparams).to(self.dev)
        else:
            self.top = MultiChainPairEnergies_g(hparams).to(self.dev)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, sequence, max_seq_len,
                ppoe, chain_lens, contact_idx):
        etab, E_idx = self.potts(msas,
                                 features,
                                 seq_lens,
                                 focuses,
                                 term_lens,
                                 src_key_mask,
                                 X,
                                 x_mask,
                                 max_seq_len,
                                 ppoe,
                                 chain_lens,
                                 contact_idx=contact_idx)
        if torch.isnan(etab).any() or torch.isnan(E_idx).any():
            raise RuntimeError("nan found in potts model")
        nlcpl, avg_prob, counter = _nlcpl(etab, E_idx, sequence, x_mask)
        if torch.isnan(nlcpl).any() or torch.isnan(avg_prob).any():
            raise RuntimeError("nan when computing nlcpl")
        return nlcpl, avg_prob, counter

    def potts(self, msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe,
              chain_idx, contact_idx):
        ''' compute the \'potts model parameters\' for the structure '''
        if self.hparams['use_terms']:
            node_embeddings, edge_embeddings = self.bot(msas,
                                                        features,
                                                        seq_lens,
                                                        focuses,
                                                        term_lens,
                                                        src_key_mask,
                                                        max_seq_len,
                                                        chain_idx,
                                                        X,
                                                        ppoe,
                                                        contact_idx=contact_idx)
        else:
            node_embeddings, edge_embeddings = 0, 0
        etab, E_idx = self.top(node_embeddings, edge_embeddings, X, x_mask, chain_idx)

        return etab, E_idx
