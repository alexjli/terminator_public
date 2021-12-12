from .layers.condense import *
from .layers.energies.s2s import *
from .layers.energies.gvp import *
#from .struct2seq.self_attention import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from terminator.utils.common import int_to_aa
from terminator.utils.loop_utils import nlpl as _nlpl, nlcpl as _nlcpl

class TERMinator(nn.Module):
    def __init__(self, hparams, device = 'cuda:0'):
        super(TERMinator, self).__init__()
        self.dev = device
        self.hparams = hparams
        self.bot = CondenseMSA(hparams = self.hparams, device = self.dev)

        if self.hparams["use_terms"]:
            self.hparams['energies_input_dim']= self.hparams['term_hidden_dim']
        else:
            self.hparams['energies_input_dim'] = 0

        if self.hparams['struct2seq_linear']:
            self.top = AblatedPairEnergies(hparams = self.hparams).to(self.dev)
        else:
            self.top = PairEnergies(hparams = self.hparams).to(self.dev)

        print(self.bot.hparams['term_hidden_dim'], self.top.hparams['energies_hidden_dim'])

        self.prior = torch.zeros(20).view(1, 1, 20).to(self.dev)

    def update_prior(self, aa_nrgs, alpha = 0.1):
        avg_nrgs = aa_nrgs.mean(dim = 1).mean(dim = 0).view(1, 1, 20)
        self.prior = self.prior * (1-alpha) + avg_nrgs * alpha
        self.prior = self.prior.detach()

    def forward(self,
                msas,
                features,
                seq_lens,
                focuses,
                term_lens,
                src_key_mask,
                X,
                x_mask,
                sequence,
                max_seq_len,
                ppoe):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe)
        nlcpl, avg_prob, counter = _nlcpl(etab, E_idx, sequence, x_mask)
        return nlcpl, avg_prob, counter

    ''' compute the \'potts model parameters\' for the structure '''
    def potts(self,
              msas,
              features,
              seq_lens,
              focuses,
              term_lens,
              src_key_mask,
              X,
              x_mask,
              max_seq_len,
              ppoe):
        if self.hparams['use_terms']:
            condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask, max_seq_len, ppoe)
            etab, E_idx = self.top(X, x_mask, V_embed = condense)
        else:
            etab, E_idx = self.top(X, x_mask)
        return etab, E_idx

    ''' Optimize the sequence using max psuedo-likelihood '''
    def opt_sequence(self,
                     msas,
                     features,
                     seq_lens,
                     focuses,
                     term_lens,
                     src_key_mask,
                     X,
                     x_mask,
                     sequences,
                     max_seq_len,
                     ppoe):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe)
        int_seqs = self._seq(etab, E_idx, x_mask, sequences)
        int_seqs = int_seqs.cpu().numpy()
        char_seqs = self._int_to_aa(int_seqs)
        return char_seqs

    ''' predict the optimal sequence based purely on structure '''
    def pred_sequence(self,
                      msas,
                      features,
                      seq_lens,
                      focuses,
                      term_lens,
                      src_key_mask,
                      X,
                      x_mask,
                      max_seq_len,
                      ppoe):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe)
        self_nrgs = self._get_self_etab(etab)
        init_seqs = self._init_seqs(self_nrgs)
        int_seqs = self._seq(etab, E_idx, x_mask, init_seqs)
        int_seqs = int_seqs.cpu().numpy()
        char_seqs = self._int_to_aa(int_seqs)
        return char_seqs

    ''' compute percent sequence recovery '''
    def percent_recovery(self,
                         msas,
                         features,
                         seq_lens,
                         focuses,
                         term_lens,
                         src_key_mask,
                         X,
                         x_mask,
                         sequences,
                         max_seq_len,
                         ppoe):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe)
        self_nrgs = self._get_self_etab(etab)
        init_seqs = self._init_seqs(self_nrgs)
        pred_seqs = self._seq(etab, E_idx, x_mask, init_seqs)
        p_recov = self._percent(pred_seqs, sequences, x_mask)
        return p_recov

    ''' extract self nrgs from etab '''
    def _get_self_etab(self, etab):
        self_etab = etab[:, :, 0]
        n_batch, L = self_etab.shape[:2]
        self_etab = self_etab.unsqueeze(-1).view(n_batch, L, 20, 20)
        self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
        return self_nrgs

    ''' with the predicted seq and actual seq, find the percent identity '''
    def _percent(self, pred_seqs, true_seqs, x_mask):
        is_same = (pred_seqs == true_seqs)
        lens = torch.sum(x_mask, dim=-1)
        is_same *= x_mask.bool()
        #print(is_same)
        num_same = torch.sum(is_same.float(), dim=-1)
        #print(num_same, lens)
        percent_same = num_same / lens
        return percent_same

    ''' Convert int sequence sto its corresponding amino acid identities '''
    def _int_to_aa(self, int_seqs):
        # vectorize the function so we can apply it over the np array
        vec_i2a = np.vectorize(int_to_aa)
        char_seqs = vec_i2a(int_seqs).tolist()
        char_seqs = [''.join([aa for aa in seq]) for seq in char_seqs]
        return char_seqs

    ''' Guess an initial sequence based on the self energies '''
    def _init_seqs(self, self_etab):
        #self_etab = self.ln(self_etab)
        aa_idx = torch.argmax(-self_etab, dim = -1)
        return aa_idx

    ''' Find the int aa seq with the highest psuedolikelihood using an initial guess from the self energies '''

    def _seq(self, etab, E_idx, x_mask, sequences):
        n_batch, L, k, _ = etab.shape
        etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

        # X is encoded as 20 so lets just add an extra row/col of zeros
        pad = (0, 1, 0, 1)
        etab = F.pad(etab, pad, "constant", 0)

        # separate selfE and pairE since we have to treat selfE differently
        self_etab = etab[:, :, 0:1]
        pair_etab = etab[:, :, 1:]
        # idx matrix to gather the identity at all other residues given a residue of focus
        E_aa = torch.gather(sequences.unsqueeze(-1).expand(-1, -1, k-1), 1, E_idx[:, :, 1:])
        E_aa = E_aa.view(list(E_idx[:,:,1:].shape) + [1,1]).expand(-1, -1, -1, 21, -1)
        # gather the 22 energies for each edge based on E_aa
        pair_nrgs = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
        # gather 22 self energies by taking the diagonal of the etab
        self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
        # concat the two to get a full edge etab
        edge_nrgs = torch.cat((self_nrgs, pair_nrgs), dim=2)
        # get the avg nrg for 22 possible aa identities at each position
        aa_nrgs = torch.sum(edge_nrgs, dim = 2)
        #aa_nrgs -= self.prior
        #aa_nrgs = self.ln(aa_nrgs)
        # get the indexes of the max nrgs
        # these are our predicted aa identities
        aa_idx = torch.argmax(-aa_nrgs, dim = -1)

        return aa_idx

class MultiChainTERMinator_gcnkt(TERMinator):
    def __init__(self, hparams, device = 'cuda:0'):
        super(MultiChainTERMinator_gcnkt, self).__init__(hparams, device)
        self.dev = device
        self.hparams = hparams
        self.bot = MultiChainCondenseMSA_g(hparams, device = self.dev)

        if self.hparams["use_terms"]:
            self.hparams['energies_input_dim']= self.hparams['term_hidden_dim']
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

    def forward(self,
                msas,
                features,
                seq_lens,
                focuses,
                term_lens,
                src_key_mask,
                X,
                x_mask,
                sequence,
                max_seq_len,
                ppoe,
                chain_lens,
                contact_idx):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe, chain_lens, contact_idx = contact_idx)
        if torch.isnan(etab).any() or torch.isnan(E_idx).any():
            raise RuntimeError("nan found in potts model")
        nlcpl, avg_prob, counter = _nlcpl(etab, E_idx, sequence, x_mask)
        if torch.isnan(nlcpl).any() or torch.isnan(avg_prob).any():
            raise RuntimeError("nan when computing nlcpl")
        return nlcpl, avg_prob, counter

    ''' compute the \'potts model parameters\' for the structure '''
    def potts(self,
              msas,
              features,
              seq_lens,
              focuses,
              term_lens,
              src_key_mask,
              X,
              x_mask,
              max_seq_len,
              ppoe,
              chain_idx,
              contact_idx):

        """
        # generate chain_idx from chain_lens
        chain_idx = []
        for c_lens in chain_lens:
            arrs = []
            for i in range(len(c_lens)):
                l = c_lens[i]
                arrs.append(torch.ones(l)*i)
            chain_idx.append(torch.cat(arrs, dim = -1))
        chain_idx = pad_sequence(chain_idx, batch_first = True).to(self.dev)
        """

        if self.hparams['use_terms']:
            node_embeddings, edge_embeddings = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask, max_seq_len, chain_idx, X, ppoe, contact_idx = contact_idx)
        else:
            node_embeddings, edge_embeddings = 0, 0
        etab, E_idx = self.top(node_embeddings, edge_embeddings, X, x_mask, chain_idx)

        return etab, E_idx

    ''' Optimize the sequence using max psuedo-likelihood '''
    def opt_sequence(self,
                     msas,
                     features,
                     seq_lens,
                     focuses,
                     term_lens,
                     src_key_mask,
                     X,
                     x_mask,
                     sequences,
                     max_seq_len,
                     ppoe,
                     chain_lens,
                     contact_idx):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe, chain_lens, contact_idx = contact_idx)
        int_seqs = self._seq(etab, E_idx, x_mask, sequences)
        int_seqs = int_seqs.cpu().numpy()
        char_seqs = self._int_to_aa(int_seqs)
        return char_seqs

    ''' predict the optimal sequence based purely on structure '''
    def pred_sequence(self,
                      msas,
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
                      contact_idx):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe, chain_lens, contact_idx = contact_idx)
        self_nrgs = self._get_self_etab(etab)
        init_seqs = self._init_seqs(self_nrgs)
        int_seqs = self._seq(etab, E_idx, x_mask, init_seqs)
        int_seqs = int_seqs.cpu().numpy()
        char_seqs = self._int_to_aa(int_seqs)
        return char_seqs

    ''' compute percent sequence recovery '''
    def percent_recovery(self,
                         msas,
                         features,
                         seq_lens,
                         focuses,
                         term_lens,
                         src_key_mask,
                         X,
                         x_mask,
                         sequences,
                         max_seq_len,
                         ppoe,
                         chain_lens,
                         contact_idx):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe, chain_lens, contact_idx = contact_idx)
        self_nrgs = self._get_self_etab(etab)
        init_seqs = self._init_seqs(self_nrgs)
        pred_seqs = self._seq(etab, E_idx, x_mask, init_seqs)
        p_recov = self._percent(pred_seqs, sequences, x_mask)
        return p_recov


class MultiChainTERMinator_gstats(TERMinator):
    def __init__(self, hparams, device = 'cuda:0'):
        super(MultiChainTERMinator_gstats, self).__init__(hparams, device)
        self.dev = device
        self.hparams = hparams
        self.bot = MultiChainCondenseMSA_g(hparams, device = self.dev)
        if hparams['energies_gvp']:
            self.top = GVPPairEnergies(hparams).to(self.dev)
        elif hparams['energies_full_graph']:
            self.top = PairEnergiesFullGraph(hparams).to(self.dev)
        else:
            self.top = MultiChainPairEnergies_g(hparams).to(self.dev)


        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                msas,
                features,
                seq_lens,
                focuses,
                term_lens,
                src_key_mask,
                X,
                x_mask,
                sequence,
                max_seq_len,
                ppoe,
                chain_lens,
                sing_stats,
                pair_stats):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe, chain_lens, sing_stats, pair_stats)
        nlcpl, avg_prob, counter = _nlcpl(etab, E_idx, sequence, x_mask)
        return nlcpl, avg_prob, counter

    ''' compute the \'potts model parameters\' for the structure '''
    def potts(self,
              msas,
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
              sing_stats,
              pair_stats):

        # generate chain_idx from chain_lens
        chain_idx = []
        for c_lens in chain_lens:
            arrs = []
            for i in range(len(c_lens)):
                l = c_lens[i]
                arrs.append(torch.ones(l)*i)
            chain_idx.append(torch.cat(arrs, dim = -1))
        chain_idx = pad_sequence(chain_idx, batch_first = True).to(self.dev)

        node_embeddings, edge_embeddings = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask, max_seq_len, chain_idx, X, ppoe, sing_stats = sing_stats, pair_stats = pair_stats)
        etab, E_idx = self.top(node_embeddings, edge_embeddings, X, x_mask, chain_idx)

        return etab, E_idx

    ''' Optimize the sequence using max psuedo-likelihood '''
    def opt_sequence(self,
                     msas,
                     features,
                     seq_lens,
                     focuses,
                     term_lens,
                     src_key_mask,
                     X,
                     x_mask,
                     sequences,
                     max_seq_len,
                     ppoe,
                     chain_lens,
                     sing_stats,
                     pair_stats):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe, chain_lens, sing_stats, pair_stats)
        int_seqs = self._seq(etab, E_idx, x_mask, sequences)
        int_seqs = int_seqs.cpu().numpy()
        char_seqs = self._int_to_aa(int_seqs)
        return char_seqs

    ''' predict the optimal sequence based purely on structure '''
    def pred_sequence(self,
                      msas,
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
                      sing_stats,
                      pair_stats):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe, chain_lens, sing_stats, pair_stats)
        self_nrgs = self._get_self_etab(etab)
        init_seqs = self._init_seqs(self_nrgs)
        int_seqs = self._seq(etab, E_idx, x_mask, init_seqs)
        int_seqs = int_seqs.cpu().numpy()
        char_seqs = self._int_to_aa(int_seqs)
        return char_seqs

    ''' compute percent sequence recovery '''
    def percent_recovery(self,
                         msas,
                         features,
                         seq_lens,
                         focuses,
                         term_lens,
                         src_key_mask,
                         X,
                         x_mask,
                         sequences,
                         max_seq_len,
                         ppoe,
                         chain_lens,
                         sing_stats,
                         pair_stats):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe, chain_lens, sing_stats, pair_stats)
        self_nrgs = self._get_self_etab(etab)
        init_seqs = self._init_seqs(self_nrgs)
        pred_seqs = self._seq(etab, E_idx, x_mask, init_seqs)
        p_recov = self._percent(pred_seqs, sequences, x_mask)
        return p_recov
