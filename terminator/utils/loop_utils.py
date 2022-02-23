import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm


def _to_dev(data_dict, dev):
    """ Push all tensor objects in the dictionary to the given device.

    Args
    ----
    data_dict : dict 
        Dictionary of input features to TERMinator
    dev : str 
        Device to load tensors onto
    """
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(dev)
        if key == 'gvp_data':
            data_dict['gvp_data'] = [data.to(dev) for data in data_dict['gvp_data']]


def run_epoch(model, dataloader, optimizer=None, scheduler=None, grad=False, test=False, dev="cuda:0", scaler=None):
    # arg checking
    if test:
        assert not grad, "grad should not be on for test set"
    if grad:
        assert optimizer is not None, "require an optimizer if using grads"

    running_loss = 0
    running_prob = 0
    global_count = 0

    # set grads properly
    if grad:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    # record inference outputs if necessary
    if test:
        dump = []

    progress = tqdm(total=len(dataloader))
    for i, data in enumerate(dataloader):
        # a small hack for DataParallel to know which device got which proteins
        data['scatter_idx'] = torch.arange(len(data['seq_lens']))
        _to_dev(data, dev)
        max_seq_len = max(data['seq_lens'].tolist())
        ids = data['ids']

        try:
            if scaler:
                with autocast():
                    etab, E_idx = model(data, max_seq_len)
                    loss, prob, batch_count = nlcpl(etab, E_idx, data['seqs'], data['x_mask'])
                    if model.hparams['regularize_etab'] != 0:
                        loss += model.hparams['regularize_etab']*etab.norm()
            else:
                etab, E_idx = model(data, max_seq_len)
                loss, prob, batch_count = nlcpl(etab, E_idx, data['seqs'], data['x_mask'])
                if model.hparams['regularize_etab'] != 0:
                    loss += model.hparams['regularize_etab']*etab.norm()
        except Exception as e:
            print(ids)
            raise e

        if torch.cuda.device_count() > 1:
            loss = (loss * batch_count).sum() / batch_count.sum()
            prob = (prob * batch_count).sum() / batch_count.sum()
            batch_count = batch_count.sum()

        running_loss += loss.item() * batch_count.item()
        running_prob += prob.item() * batch_count.item()
        global_count += batch_count.item()

        if grad:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if test:
            n_batch, l, n = etab.shape[:3]
            dump.append({
                'out': etab.view(n_batch, l, n, 20, 20).cpu().numpy(),
                'idx': E_idx.cpu().numpy(),
                'ids': ids
            })

        avg_loss = running_loss / (global_count)
        avg_prob = running_prob / (global_count)
        term_mask_eff = int((~data['src_key_mask']).sum().item() / data['src_key_mask'].numel() * 100)
        res_mask_eff = int(data['x_mask'].sum().item() / data['x_mask'].numel() * 100)

        progress.update(1)
        progress.refresh()
        progress.set_description_str('avg loss {} | avg prob {} | eff 0.{},0.{}'.format(
            avg_loss, avg_prob, term_mask_eff, res_mask_eff))

    progress.close()
    epoch_loss = running_loss / (global_count)
    avg_prob = running_prob / (global_count)

    if scheduler:
        scheduler.step(epoch_loss)

    torch.set_grad_enabled(True)  # just to be safe
    if test:
        return epoch_loss, avg_prob, dump
    else:
        return epoch_loss, avg_prob


# TODO: fix this
def nlpl(etab, E_idx, ref_seqs, x_mask):
    ''' Negative log psuedo-likelihood
        Returns negative log psuedolikelihoods per residue, with padding residues masked'''
    etab_device = etab.device
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 1, 0, 1)
    etab = F.pad(etab, pad, "constant", 0)
    isnt_x_aa = (ref_seqs != 20).float().to(etab_device)

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

    # get average psuedolikelihood per residue
    avg_prob = (torch.exp(log_seqs_probs) * full_mask).sum() / n_res

    # convert to nlpl
    log_seqs_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    nlpl = -torch.sum(log_seqs_probs) / n_res
    return nlpl, avg_prob, n_res


# TODO: do we zero out the energy between a non-X residue and an X residue?
def nlcpl(etab, E_idx, ref_seqs, x_mask):
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

        Returns: log likelihoods per residue, as well as tensor mask
    '''
    etab_device = etab.device
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 1, 0, 1)
    etab = F.pad(etab, pad, "constant", 0)  # TODO: should i be padding w something else?

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

    # get average composite psuedolikelihood per residue per batch
    edge_probs = torch.exp(log_edge_probs) * full_mask
    # per residue prob for entire batch rather than averaged over proteins
    avg_prob = torch.sum(edge_probs) / n_edges

    # convert to nlcpl
    log_edge_probs *= full_mask  # zero out positions that don't have residues or where the native sequence is X
    nlcpl = -torch.sum(log_edge_probs) / n_edges
    return nlcpl, avg_prob, n_edges
