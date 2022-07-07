""" Loss functions for TERMinator, and a customizable loss function constructor built from the included components.

In order to use the customizable loss function constructor :code:`construct_loss_fn`, loss functions
must have the signature :code:`loss(etab, E_idx, data)`, where
    - :code:`loss` is the name of the loss fn
    - :code:`etab` is the outputted etab from TERMinator
    - :code:`E_idx` is the edge index outputted from TERMinator
    - :code:`data` is the training data dictionary
Additionally, the function must return two outputs :code:`loss_contribution, norm_count`, where
    - :code:`loss_contribution` is the computed loss contribution by the function
    - :code:`norm_count` is a normalizing constant associated with the loss (e.g. when averaging across losses in batches,
the average loss will be :math:`\\frac{\\sum_i loss_contribution}{\\sum_i norm_count}`)
"""
import sys

import torch
import torch.nn.functional as F
import torch.linalg

# pylint: disable=no-member

NOT_LOSS_FNS = ["_get_loss_fn", "construct_loss_fn"]


def nlpl(etab, E_idx, data):
    """ Negative log psuedo-likelihood
        Returns negative log psuedolikelihoods per residue, with padding residues masked """
    ref_seqs = data['seqs']
    x_mask = data['x_mask']
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 1, 0, 1)
    etab = F.pad(etab, pad, "constant", 0)
    isnt_x_aa = (ref_seqs != 20).float().to(etab.device)

    # separate selfE and pairE since we have to treat selfE differently
    self_etab = etab[:, :, 0:1]
    pair_etab = etab[:, :, 1:]
    # idx matrix to gather the identity at all other residues given a residue of focus
    E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k - 1), 1, E_idx[:, :, 1:])
    E_aa = E_aa.view(list(E_idx[:, :, 1:].shape) + [1, 1]).expand(-1, -1, -1, 21, -1)
    # gather the 22 energies for each edge based on E_aa
    pair_nrgs = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
    # concat the two to get a full edge etab
    edge_nrgs = torch.cat((self_nrgs, pair_nrgs), dim=2)
    # get the avg nrg for 22 possible aa identities at each position
    aa_nrgs = torch.sum(edge_nrgs, dim=2)

    # convert energies to probabilities
    log_all_aa_probs = torch.log_softmax(-aa_nrgs, dim=2)
    # get the probability of the sequence
    log_seqs_probs = torch.gather(log_all_aa_probs, 2, ref_seqs.unsqueeze(-1)).squeeze(-1)

    full_mask = x_mask * isnt_x_aa
    n_res = torch.sum(x_mask * isnt_x_aa)

    # convert to nlpl
    log_seqs_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    nlpl_return = -torch.sum(log_seqs_probs) / n_res
    return nlpl_return, int(n_res)


def nlcpl(etab, E_idx, data):
    """ Negative log composite psuedo-likelihood
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

        Returns: log likelihoods per residue, as well as tensor mask
    """
    ref_seqs = data['seqs']
    x_mask = data['x_mask']
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 1, 0, 1)
    etab = F.pad(etab, pad, "constant", 0)

    isnt_x_aa = (ref_seqs != 20).float().to(etab.device)

    # separate selfE and pairE since we have to treat selfE differently
    self_etab = etab[:, :, 0:1]
    pair_etab = etab[:, :, 1:]

    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs_im = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
    self_nrgs_im_expand = self_nrgs_im.expand(-1, -1, k - 1, -1)

    # E_idx for all but self
    E_idx_jn = E_idx[:, :, 1:]

    # self Es gathered from E_idx_others
    E_idx_jn_expand = E_idx_jn.unsqueeze(-1).expand(-1, -1, -1, 21)
    self_nrgs_jn = torch.gather(self_nrgs_im_expand, 1, E_idx_jn_expand)

    # idx matrix to gather the identity at all other residues given a residue of focus
    E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k - 1), 1, E_idx_jn)
    # expand the matrix so we can gather pair energies
    E_aa = E_aa.view(list(E_idx_jn.shape) + [1, 1]).expand(-1, -1, -1, 21, -1)
    # gather the 22 energies for each edge based on E_aa
    pair_nrgs_jn = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
    # sum_(u != n,m) E_p(a_i,n; A_u)
    sum_pair_nrgs_jn = torch.sum(pair_nrgs_jn, dim=2)
    pair_nrgs_im_u = sum_pair_nrgs_jn.unsqueeze(2).expand(-1, -1, k - 1, -1) - pair_nrgs_jn

    # get pair_nrgs_u_jn from pair_nrgs_im_u
    E_idx_imu_to_ujn = E_idx_jn.unsqueeze(-1).expand(pair_nrgs_im_u.shape)
    pair_nrgs_u_jn = torch.gather(pair_nrgs_im_u, 1, E_idx_imu_to_ujn)

    # start building this wacky energy table
    self_nrgs_im_expand = self_nrgs_im_expand.unsqueeze(-1).expand(-1, -1, -1, -1, 21)
    self_nrgs_jn_expand = self_nrgs_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 21).transpose(-2, -1)
    pair_nrgs_im_expand = pair_nrgs_im_u.unsqueeze(-1).expand(-1, -1, -1, -1, 21)
    pair_nrgs_jn_expand = pair_nrgs_u_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 21).transpose(-2, -1)

    composite_nrgs = (self_nrgs_im_expand + self_nrgs_jn_expand + pair_etab + pair_nrgs_im_expand +
                      pair_nrgs_jn_expand)

    # convert energies to probabilities
    composite_nrgs_reshape = composite_nrgs.view(n_batch, L, k - 1, 21 * 21, 1)
    log_composite_prob_dist = torch.log_softmax(-composite_nrgs_reshape, dim=-2).view(n_batch, L, k - 1, 21, 21)
    # get the probability of the sequence
    im_probs = torch.gather(log_composite_prob_dist, 4, E_aa).squeeze(-1)
    ref_seqs_expand = ref_seqs.view(list(ref_seqs.shape) + [1, 1]).expand(-1, -1, k - 1, 1)
    log_edge_probs = torch.gather(im_probs, 3, ref_seqs_expand).squeeze(-1)

    # reshape masks
    x_mask = x_mask.unsqueeze(-1)
    isnt_x_aa = isnt_x_aa.unsqueeze(-1)
    full_mask = x_mask * isnt_x_aa
    n_edges = torch.sum(full_mask.expand(log_edge_probs.shape))

    # convert to nlcpl
    log_edge_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    nlcpl_return = -torch.sum(log_edge_probs) / n_edges
    return nlcpl_return, int(n_edges)


def nlcpl_test(etab, E_idx, data):
    """ Alias of nlcpl_full """
    return nlcpl_full(etab, E_idx, data)


def nlcpl_full(etab, E_idx, data):
    """ Negative log composite psuedo-likelihood
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

        Returns: averaged log likelihood per residue pair,
        as well as the number of edges considered
    """

    ref_seqs = data['seqs']
    x_mask = data['x_mask']
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 1, 0, 1)
    etab = F.pad(etab, pad, "constant", 0)

    isnt_x_aa = (ref_seqs != 20).float().to(etab.device)

    # separate selfE and pairE since we have to treat selfE differently
    self_etab = etab[:, :, 0:1]
    pair_etab = etab[:, :, 1:]

    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs_im = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
    self_nrgs_im_expand = self_nrgs_im.expand(-1, -1, k - 1, -1)

    # E_idx for all but self
    E_idx_jn = E_idx[:, :, 1:]

    # self Es gathered from E_idx_others
    E_idx_jn_expand = E_idx_jn.unsqueeze(-1).expand(-1, -1, -1, 21)
    self_nrgs_jn = torch.gather(self_nrgs_im_expand, 1, E_idx_jn_expand)

    # idx matrix to gather the identity at all other residues given a residue of focus
    E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k - 1), 1, E_idx_jn)
    # expand the matrix so we can gather pair energies
    E_aa = E_aa.view(list(E_idx_jn.shape) + [1, 1]).expand(-1, -1, -1, 21, -1)
    # gather the 22 energies for each edge based on E_aa
    pair_nrgs_jn = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
    # sum_(u != n,m) E_p(a_i,n; A_u)
    sum_pair_nrgs_jn = torch.sum(pair_nrgs_jn, dim=2)
    pair_nrgs_im_u = sum_pair_nrgs_jn.unsqueeze(2).expand(-1, -1, k - 1, -1) - pair_nrgs_jn

    # get pair_nrgs_u_jn from pair_nrgs_im_u
    E_idx_imu_to_ujn = E_idx_jn.unsqueeze(-1).expand(pair_nrgs_im_u.shape)
    pair_nrgs_u_jn = torch.gather(pair_nrgs_im_u, 1, E_idx_imu_to_ujn)

    # start building this wacky energy table
    self_nrgs_im_expand = self_nrgs_im_expand.unsqueeze(-1).expand(-1, -1, -1, -1, 21)
    self_nrgs_jn_expand = self_nrgs_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 21).transpose(-2, -1)
    pair_nrgs_im_expand = pair_nrgs_im_u.unsqueeze(-1).expand(-1, -1, -1, -1, 21)
    pair_nrgs_jn_expand = pair_nrgs_u_jn.unsqueeze(-1).expand(-1, -1, -1, -1, 21).transpose(-2, -1)

    composite_nrgs = (self_nrgs_im_expand + self_nrgs_jn_expand + pair_etab + pair_nrgs_im_expand +
                      pair_nrgs_jn_expand)

    # convert energies to probabilities
    composite_nrgs_reshape = composite_nrgs.view(n_batch, L, k - 1, 21 * 21, 1)
    log_composite_prob_dist = torch.log_softmax(-composite_nrgs_reshape, dim=-2).view(n_batch, L, k - 1, 21, 21)
    # get the probability of the sequence
    im_probs = torch.gather(log_composite_prob_dist, 4, E_aa).squeeze(-1)
    ref_seqs_expand = ref_seqs.view(list(ref_seqs.shape) + [1, 1]).expand(-1, -1, k - 1, 1)
    log_edge_probs = torch.gather(im_probs, 3, ref_seqs_expand).squeeze(-1)

    # reshape masks
    x_mask = x_mask.unsqueeze(-1)
    isnt_x_aa = isnt_x_aa.unsqueeze(-1)
    full_mask = x_mask * isnt_x_aa
    n_self = torch.sum(full_mask)
    n_edges = torch.sum(full_mask.expand(log_edge_probs.shape))

    # compute probabilities for self edges
    self_nrgs = torch.diagonal(etab[:, :, 0], offset=0, dim1=-2, dim2=-1)
    log_self_probs_dist = torch.log_softmax(-self_nrgs, dim=-1) * full_mask
    log_self_probs = torch.gather(log_self_probs_dist, 2, ref_seqs.unsqueeze(-1))

    # convert to nlcpl
    log_edge_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    nlcpl_return = -(torch.sum(log_self_probs) + torch.sum(log_edge_probs)) / (n_self + n_edges)
    return nlcpl_return, int(n_self + n_edges)


# pylint: disable=unused-argument
def etab_norm_penalty(etab, E_idx, data):
    """ Take the norm of all etabs and scale it by the total number of residues involved """
    seq_lens = data['seq_lens']
    etab_norm = torch.linalg.norm(etab.view([-1]))
    return etab_norm / seq_lens.sum(), int(seq_lens.sum())


# pylint: disable=unused-argument
def mindren_etab_norm_penalty(etab, E_idx, data):
    """ Take the norm of all etabs and scale it by the total number of residues involved """
    seq_lens = data['seq_lens']
    # etab_norm = torch.linalg.norm(etab.view([-1]))
    # return etab_norm / seq_lens.sum(), int(seq_lens.sum())
    etab_norm = torch.mean(torch.linalg.norm(etab, dim=(1,2,3)) / seq_lens)
    return etab_norm, int(seq_lens.sum())


# pylint: disable=unused-argument
def etab_l1_norm_penalty(etab, E_idx, data):
    """ Take the norm of all etabs and scale it by the total number of residues involved """
    seq_lens = data['seq_lens']
    # etab_norm = torch.linalg.norm(etab.view([-1]))
    # return etab_norm / seq_lens.sum(), int(seq_lens.sum())
    etab_norm = torch.mean(torch.sum(torch.abs(etab), dim=(1,2,3)) / seq_lens)
    return etab_norm, int(seq_lens.sum())


# pylint: disable=unused-argument
def pair_self_energy_ratio(etab, E_idx, data):
    """ Return the ratio of the scaled norm of pair energies vs self energies in an etab """
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)
    self_etab = etab[:, :, 0:1]
    pair_etab = etab[:, :, 1:]

    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
    # compute an "avg" by taking the mean of the magnitude of the values
    # then sqrt to get approx right scale for the energies
    self_nrgs_avg = self_nrgs[self_nrgs != 0].square().mean().sqrt()
    pair_nrgs_avg = pair_etab[pair_etab != 0].square().mean().sqrt()

    return pair_nrgs_avg / self_nrgs_avg, n_batch


# Loss function construction


def _get_loss_fn(fn_name):
    """ Retrieve a loss function from this file given the function name """
    try:
        if fn_name in NOT_LOSS_FNS:  # prevent recursive and unexpected behavior
            raise NameError
        return getattr(sys.modules[__name__], fn_name)
    except NameError as ne:
        raise ValueError(f"Loss fn {fn_name} not found in {__name__}") from ne


def construct_loss_fn(hparams):
    """ Construct a combined loss function based on the inputted hparams

    Args
    ----
    hparams : dict
        The fully constructed hparams (see :code:`terminator/utils/model/default_hparams.py`). It should
        contain an entry for 'loss_config' in the format {loss_fn_name : scaling_factor}. For example,
        .. code-block :
            {
                'nlcpl': 1,
                'etab_norm_penalty': 0.01
            }

    Returns
    -------
    _loss_fn
        The constructed loss function
    """
    loss_config = hparams['loss_config']

    def _loss_fn(etab, E_idx, data):
        """ The returned loss function """
        loss_dict = {}
        for loss_fn_name, scaling_factor in loss_config.items():
            subloss_fn = _get_loss_fn(loss_fn_name)
            loss, count = subloss_fn(etab, E_idx, data)
            loss_dict[loss_fn_name] = {"loss": loss, "count": count, "scaling_factor": scaling_factor}
        return loss_dict

    return _loss_fn
