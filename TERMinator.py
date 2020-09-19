from nets import CondenseMSA
from struct2seq.energies import *
from struct2seq.self_attention import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from preprocessing.common import int_to_aa


class TERMinator(nn.Module):
    def __init__(self, hidden_dim = 64, resnet_blocks = 1, conv_filter = 9, term_heads = 4, term_layers = 4, k_neighbors = 30, device = 'cuda:0'):
        super(TERMinator, self).__init__()
        self.dev = device
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.bot = CondenseMSA(hidden_dim = hidden_dim, filter_len = conv_filter, num_blocks = resnet_blocks, nheads = term_heads, num_transformers = term_layers, device = self.dev)
        self.top = PairEnergies(num_letters = 20, node_features = hidden_dim, edge_features = hidden_dim, input_dim = hidden_dim, hidden_dim = hidden_dim, k_neighbors=k_neighbors, num_encoder_layers = 3).to(self.dev)

        self.prior = torch.zeros(20).view(1, 1, 20).to(self.dev)

    ''' Negative log psuedo-likelihood '''
    ''' Averaged nlpl per residue, across batches '''
    def _nlpl(self, etab, E_idx, ref_seqs, x_mask):
        n_batch, L, k, _ = etab.shape
        etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)
        
        # X is encoded as 20 so lets just add an extra row/col of zeros
        pad = (0, 1, 0, 1)
        etab = F.pad(etab, pad, "constant", 0)

        isnt_x_aa = (ref_seqs != 20).float().to(self.dev)

        # separate selfE and pairE since we have to treat selfE differently
        self_etab = etab[:, :, 0:1] 
        pair_etab = etab[:, :, 1:]
        # idx matrix to gather the identity at all other residues given a residue of focus
        E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k-1), 1, E_idx[:, :, 1:])
        E_aa = E_aa.view(list(E_idx[:,:,1:].shape) + [1,1]).expand(-1, -1, -1, 21, -1)
        # gather the 22 energies for each edge based on E_aa
        pair_nrgs = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
        # gather 22 self energies by taking the diagonal of the etab
        self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
        # concat the two to get a full edge etab
        edge_nrgs = torch.cat((self_nrgs, pair_nrgs), dim=2)
        # get the avg nrg for 22 possible aa identities at each position
        aa_nrgs = torch.sum(edge_nrgs, dim = 2)

        #prior_loss = ((aa_nrgs - self.prior) ** 2).mean(dim = 1).sum()

        #aa_nrgs -= self.prior
        #self.update_prior(aa_nrgs)
        #aa_nrgs = self.ln(aa_nrgs)


        # convert energies to probabilities
        all_aa_probs = torch.softmax(-aa_nrgs, dim = 2)
        # get the probability of the sequence
        seqs_probs = torch.gather(all_aa_probs, 2, ref_seqs.unsqueeze(-1)).squeeze(-1)
        
        # get average psuedolikelihood per residue
        avg_prob = torch.mean(torch.mean(seqs_probs, dim=-1))

        # convert to nlpl
        log_probs = torch.log(seqs_probs) * x_mask # zero out positions that don't have residues
        log_probs = log_probs * isnt_x_aa # zero out positions where the native sequence is X
        n_res = torch.sum(x_mask * isnt_x_aa, dim=-1)
        nlpl = torch.sum(log_probs, dim=-1)#/n_res
        nlpl = -torch.mean(nlpl)
        return nlpl, avg_prob

    ''' Negative log composite psuedo-likelihood
        Averaged nlcpl per residue, across batches
        p(a_i,m ; a_j,n) = 
            softmax [
                E_s(a_i,m) + E_s(a_j,n) 
                + E_p(a_i,m ; a_j,n)
                + sum_(u != m,n) [
                    E_p(a_i,m; A_u)
                    + E_p(A_u, a_j,n)
                    ]
                ]
        '''
    def _nlcpl(self, etab, E_idx, ref_seqs, x_mask):
        n_batch, L, k, _ = etab.shape
        etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)
        
        # X is encoded as 20 so lets just add an extra row/col of zeros
        pad = (0, 1, 0, 1)
        etab = F.pad(etab, pad, "constant", 0)

        isnt_x_aa = (ref_seqs != 20).float().to(self.dev)

        # separate selfE and pairE since we have to treat selfE differently
        self_etab = etab[:, :, 0:1] 
        pair_etab = etab[:, :, 1:]

        # gather 22 self energies by taking the diagonal of the etab
        self_nrgs_im = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
        self_nrgs_im_expand = self_nrgs_im.expand(-1, -1, k-1, -1)

        # E_idx for all but self
        E_idx_jn = E_idx[:, :, 1:]

        # self Es gathered from E_idx_others
        E_idx_jn_expand = E_idx_jn.unsqueeze(-1).expand(-1, -1, -1, 21)
        self_nrgs_jn = torch.gather(self_nrgs_im_expand, 1, E_idx_jn_expand)

        # idx matrix to gather the identity at all other residues given a residue of focus
        E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k-1), 1, E_idx_jn)
        # expand the matrix so we can gather pair energies
        E_aa = E_aa.view(list(E_idx_jn.shape) + [1,1]).expand(-1, -1, -1, 21, -1)
        # gather the 22 energies for each edge based on E_aa
        pair_nrgs_jn = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
        # sum_(u != n,m) E_p(a_i,n; A_u)
        sum_pair_nrgs_jn = torch.sum(pair_nrgs_jn, dim = 2)
        pair_nrgs_im_u = sum_pair_nrgs_jn.unsqueeze(2).expand(-1, -1, k-1, -1) - pair_nrgs_jn

        # get pair_nrgs_u_jn from pair_nrgs_im_u
        E_idx_imu_to_ujn = E_idx_jn.unsqueeze(-1).expand(pair_nrgs_im_u.shape)
        pair_nrgs_u_jn = torch.gather(pair_nrgs_im_u, 1, E_idx_imu_to_ujn)

        """
        # concat the two to get a full edge etab
        edge_nrgs = torch.cat((self_nrgs, pair_nrgs), dim=2)
        # get the avg nrg for 22 possible aa identities at each position
        aa_nrgs = torch.sum(edge_nrgs, dim = 2)
        """

        # start building this wacky energy table
        self_nrgs_im_expand = self_nrgs_im_expand.unsqueeze(-1).expand(-1, -1, -1, -1, 21)
        self_nrgs_jn_expand = self_nrgs_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 21).transpose(-2, -1)
        pair_nrgs_im_expand = pair_nrgs_im_u.unsqueeze(-1).expand(-1, -1, -1, -1, 21)
        pair_nrgs_jn_expand = pair_nrgs_u_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 21).transpose(-2, -1)

        composite_nrgs = (self_nrgs_im_expand + 
                          self_nrgs_jn_expand + 
                          pair_etab + 
                          pair_nrgs_im_expand + 
                          pair_nrgs_jn_expand)

        # convert energies to probabilities
        composite_nrgs_reshape = composite_nrgs.view(n_batch, L, k-1, 21 * 21, 1)
        composite_prob_dist = torch.softmax(-composite_nrgs_reshape, dim = -2).view(n_batch, L, k-1, 21, 21)

        # get the probability of the sequence
        im_probs = torch.gather(composite_prob_dist, 4, E_aa).squeeze(-1)
        ref_seqs_expand = ref_seqs.view(list(ref_seqs.shape) + [1,1]).expand(-1, -1, k-1, 1)
        edge_probs = torch.gather(im_probs, 3, ref_seqs_expand).squeeze(-1)
        
        # get average composite psuedolikelihood per residue per batch
        avg_prob = torch.mean(torch.mean(torch.mean(edge_probs, dim=-1), dim=-1))

        x_mask = x_mask.unsqueeze(-1)
        isnt_x_aa = isnt_x_aa.unsqueeze(-1)
        # convert to nlcpl
        log_edge_probs = torch.log(edge_probs) * x_mask # zero out positions that don't have residues
        log_edge_probs = log_edge_probs * isnt_x_aa # zero out positions where the native sequence is X
        n_res = torch.sum((x_mask * isnt_x_aa).squeeze(-1), dim=-1)
        log_seq_probs = torch.sum(log_edge_probs, dim=-1)
        nlcpl = torch.sum(log_seq_probs, dim=-1)/(2 * n_res) # we divide by 2 because each probability is duplicated
        nlcpl = -torch.mean(nlcpl)
        return nlcpl, avg_prob

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
                sequence):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        etab, E_idx = self.top(condense, X, x_mask)
        rms = torch.sqrt(torch.mean(etab**2))
        nlpl, avg_prob = self._nlcpl(etab, E_idx, sequence, x_mask)
        return nlpl, rms, avg_prob

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
              sparse = False):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        etab, E_idx = self.top(condense, X, x_mask, sparse = sparse)
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
                     sequences):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        etab, E_idx = self.top(condense, X, x_mask)
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
                      x_mask):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        etab, E_idx = self.top(condense, X, x_mask)
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
                         sequences):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        etab, E_idx = self.top(condense, X, x_mask)
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
        is_same *= x_mask.byte()
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

    ''' Find the int aa seq with the highest psuedolikelihood '''
    def _seq(self, etab, E_idx, x_mask, sequences):
        n_batch, L, k, _ = etab.shape
        etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)
        
        # separate selfE and pairE since we have to treat selfE differently
        self_etab = etab[:, :, 0:1] 
        pair_etab = etab[:, :, 1:]
        # idx matrix to gather the identity at all other residues given a residue of focus
        E_aa = torch.gather(sequences.unsqueeze(-1).expand(-1, -1, k-1), 1, E_idx[:, :, 1:])
        E_aa = E_aa.view(list(E_idx[:,:,1:].shape) + [1,1]).expand(-1, -1, -1, 20, -1)
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


'''
TERMinator, except using different Structured Transformers for self and pair energies
'''
class TERMinator2(nn.Module):
    def __init__(self, hidden_dim = 64, resnet_blocks = 1, conv_filter = 9, term_heads = 4, term_layers = 4, k_neighbors = 30, device = 'cuda:0'):
        super(TERMinator2, self).__init__()
        self.dev = device
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.bot = CondenseMSA(hidden_dim = hidden_dim, filter_len = conv_filter, num_blocks = resnet_blocks, nheads = term_heads, num_transformers = term_layers, device = self.dev)
        self.selfE = SelfEnergies(num_letters = 20, node_features = hidden_dim, edge_features = hidden_dim, input_dim = hidden_dim, hidden_dim = hidden_dim, k_neighbors=k_neighbors, num_encoder_layers = 1).to(self.dev)
        self.pairE = PairEnergies(num_letters = 20, node_features = hidden_dim, edge_features = hidden_dim, input_dim = hidden_dim, hidden_dim = hidden_dim, k_neighbors=k_neighbors, num_encoder_layers = 1).to(self.dev)

    ''' Negative log psuedo-likelihood '''
    ''' Averaged nlpl per residue, across batches '''
    def _nlpl(self, self_etab, etab, E_idx, seqs, x_mask):
        n_batch, L, k, _ = etab.shape
        etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)
        
        # separate selfE and pairE since we have to treat selfE differently 
        pair_etab = etab[:, :, 1:]
        # idx matrix to gather the identity at all other residues given a residue of focus
        E_aa = torch.gather(seqs.unsqueeze(-1).expand(-1, -1, k-1), 1, E_idx[:, :, 1:])
        E_aa = E_aa.view(list(E_idx[:,:,1:].shape) + [1,1]).expand(-1, -1, -1, 20, -1)
        # gather the 22 energies for each edge based on E_aa
        pair_nrgs = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
        # make self energies tensor the proper shape for concatenation
        self_nrgs = self_etab.unsqueeze(2)
        # concat the two to get a full edge etab
        edge_nrgs = torch.cat((self_nrgs, pair_nrgs), dim=2)
        # get the nrg of for 22 possible aa identities at each position
        aa_nrgs = torch.sum(edge_nrgs, dim = 2)
        # convert energies to probabilities
        all_aa_probs = torch.softmax(-aa_nrgs, dim = -1)
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
        self_etab = self.selfE(condense, X, x_mask)
        etab, E_idx = self.pairE(condense, X, x_mask)
        nlpl = self._nlpl(self_etab, etab, E_idx, sequence, x_mask)
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
        etab, E_idx = self.pairE(condense, X, x_mask, sparse = sparse)
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
                     sequences):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        etab, E_idx = self.pairE(condense, X, x_mask)
        self_etab = self.selfE(condense, X, x_mask)
        int_seqs = self._seq(self_etab, etab, E_idx, x_mask, sequences)
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
                      x_mask):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        etab, E_idx = self.pairE(condense, X, x_mask)
        self_nrgs = self.selfE(condense, X, x_mask)
        init_seqs = self._init_seqs(self_nrgs)
        int_seqs = self._seq(self_nrgs, etab, E_idx, x_mask, init_seqs)
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
                         sequences):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        etab, E_idx = self.pairE(condense, X, x_mask)
        self_nrgs = self.selfE(condense, X, x_mask)
        init_seqs = self._init_seqs(self_nrgs)
        pred_seqs = self._seq(self_nrgs, etab, E_idx, x_mask, init_seqs)
        p_recov = self._percent(pred_seqs, sequences, x_mask)
        return p_recov

    ''' with the predicted seq and actual seq, find the percent identity '''
    def _percent(self, pred_seqs, true_seqs, x_mask):
        is_same = (pred_seqs == true_seqs)
        lens = torch.sum(x_mask, dim=-1)
        is_same *= x_mask.byte()
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
        aa_idx = torch.argmax(-self_etab, dim = -1)
        return aa_idx

    ''' Find the int aa seq with the highest psuedolikelihood '''
    def _seq(self, self_etab, etab, E_idx, x_mask, ref_seqs):
        n_batch, L, k, _ = etab.shape

        etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

        # separate selfE and pairE since we have to treat selfE differently 
        pair_etab = etab[:, :, 1:]
        # idx matrix to gather the identity at all other residues given a residue of focus
        E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k-1), 1, E_idx[:, :, 1:])
        E_aa = E_aa.view(list(E_idx[:,:,1:].shape) + [1,1]).expand(-1, -1, -1, 20, -1)
        # gather the 22 energies for each edge based on E_aa
        pair_nrgs = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
        # make self energies tensor the proper shape for concatenation
        self_nrgs = self_etab.unsqueeze(2)
        # concat the two to get a full edge etab
        edge_nrgs = torch.cat((self_nrgs, pair_nrgs), dim=2)
        # get the nrg of for 22 possible aa identities at each position
        aa_nrgs = torch.sum(edge_nrgs, dim = 2)
        # get the indexes of the max nrgs
        # these are our predicted aa identities
        aa_idx = torch.argmax(-aa_nrgs, dim = -1)

        return aa_idx


""" imma try something: don't introduce edges until the end """
class TERMinator3(nn.Module):
    def __init__(self, hidden_dim = 64, resnet_blocks = 1, conv_filter = 9, term_heads = 4, term_layers = 4, k_neighbors = 30, device = 'cuda:0'):
        super(TERMinator3, self).__init__()
        self.dev = device
        self.k_neighbors = k_neighbors
        self.hidden_dim = hidden_dim
        self.bot = CondenseMSA(hidden_dim = hidden_dim, filter_len = conv_filter, num_blocks = resnet_blocks, nheads = term_heads, num_transformers = term_layers, device = self.dev)
        self.top = RawSelfEnergies(num_letters = 20, node_features = hidden_dim, edge_features = hidden_dim, input_dim = hidden_dim, hidden_dim = hidden_dim, k_neighbors=k_neighbors, num_encoder_layers = 3).to(self.dev)

        self.W_out = nn.Linear(hidden_dim*3, 20 * 20)


    ''' Negative log psuedo-likelihood '''
    ''' Averaged nlpl per residue, across batches '''
    def _nlpl(self, etab, E_idx, ref_seqs, x_mask):
        n_batch, L, k, _ = etab.shape
        etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)
        
        # X is encoded as 20 so lets just add an extra row/col of zeros
        pad = (0, 1, 0, 1)
        etab = F.pad(etab, pad, "constant", 0)

        isnt_x_aa = (ref_seqs != 20).float().to(self.dev)

        # separate selfE and pairE since we have to treat selfE differently
        self_etab = etab[:, :, 0:1] 
        pair_etab = etab[:, :, 1:]
        # idx matrix to gather the identity at all other residues given a residue of focus
        E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k-1), 1, E_idx[:, :, 1:])
        E_aa = E_aa.view(list(E_idx[:,:,1:].shape) + [1,1]).expand(-1, -1, -1, 21, -1)
        # gather the 22 energies for each edge based on E_aa
        pair_nrgs = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
        # gather 22 self energies by taking the diagonal of the etab
        self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
        # concat the two to get a full edge etab
        edge_nrgs = torch.cat((self_nrgs, pair_nrgs), dim=2)
        # get the avg nrg for 22 possible aa identities at each position
        aa_nrgs = torch.sum(edge_nrgs, dim = 2)

        #prior_loss = ((aa_nrgs - self.prior) ** 2).mean(dim = 1).sum()

        #aa_nrgs -= self.prior
        #self.update_prior(aa_nrgs)
        #aa_nrgs = self.ln(aa_nrgs)


        # convert energies to probabilities
        all_aa_probs = torch.softmax(-aa_nrgs, dim = 2)

        if torch.isnan(all_aa_probs).any():
            print(aa_nrgs)
            print(all_aa_probs)
            exit()

        # get the probability of the sequence
        seqs_probs = torch.gather(all_aa_probs, 2, ref_seqs.unsqueeze(-1)).squeeze(-1)

        # convert to nlpl
        log_probs = torch.log(seqs_probs) * x_mask # zero out positions that don't have residues
        log_probs = log_probs * isnt_x_aa # zero out positions where the native sequence is X
        n_res = torch.sum(x_mask * isnt_x_aa, dim=-1)
        nlpl = torch.sum(log_probs, dim=-1)/n_res
        nlpl = -torch.mean(nlpl)
        return nlpl

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
                sequence):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses,
                                 term_lens, src_key_mask, X, x_mask)
        rms = torch.sqrt(torch.mean(etab**2))
        nlpl = self._nlpl(etab, E_idx, sequence, x_mask)
        return nlpl, rms

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
              sparse = False):
        condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask)
        h_V, h_E, E_idx = self.top(condense, X, x_mask)
        h_EV = cat_edge_endpoints(h_E, h_V, E_idx)
        etab = self.W_out(h_EV) * x_mask.unsqueeze(-1).unsqueeze(-1)
        if torch.isnan(etab).any():
            print(etab)
            print('etab')
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
                     sequences):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)
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
                      x_mask):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)
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
                         sequences):
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)
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
        is_same *= x_mask.byte()
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

    ''' Find the int aa seq with the highest psuedolikelihood '''
    def _seq(self, etab, E_idx, x_mask, sequences):
        n_batch, L, k, _ = etab.shape
        etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)
        
        # separate selfE and pairE since we have to treat selfE differently
        self_etab = etab[:, :, 0:1] 
        pair_etab = etab[:, :, 1:]
        # idx matrix to gather the identity at all other residues given a residue of focus
        E_aa = torch.gather(sequences.unsqueeze(-1).expand(-1, -1, k-1), 1, E_idx[:, :, 1:])
        E_aa = E_aa.view(list(E_idx[:,:,1:].shape) + [1,1]).expand(-1, -1, -1, 20, -1)
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


class TERMinator4(TERMinator):
    def __init__(self, hidden_dim = 64, resnet_blocks = 1, conv_filter = 9, term_heads = 4, term_layers = 4, k_neighbors = 30, device = 'cuda:0'):
        super(TERMinator4, self).__init__(hidden_dim = 64, resnet_blocks = 1, conv_filter = 9, term_heads = 4, term_layers = 4, k_neighbors = 30, device = 'cuda:0')
        self.bot = CondenseMSA(hidden_dim = hidden_dim, filter_len = conv_filter, num_blocks = resnet_blocks, nheads = term_heads, num_transformers = term_layers, device = self.dev)
        self.top = PairEnergiesFullGraph(num_letters = 20, node_features = hidden_dim, edge_features = hidden_dim, input_dim = hidden_dim, hidden_dim = hidden_dim, k_neighbors=k_neighbors, num_encoder_layers = 3).to(self.dev)


