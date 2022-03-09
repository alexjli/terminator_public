""" Utilities for running training and evaluation loops """
import torch
from tqdm import tqdm

# pylint: disable=no-member


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


def _compute_loss(loss_dict):
    """ Compute the total loss given a loss dictionary

    Args
    ----
    loss_dict : dict
        Dictionary with keys :code:`loss_fn_name` and values of dictionaries with
        - :code:`loss` corresponding to the loss value
        - :code:`count` corresponding to the normalizing factor
        - :code:`scaling_factor` corresponding to the scaling coefficient in the loss function

    Returns
    -------
    loss : torch.Tensor of size 1
        Computed loss
    """
    loss = 0
    for subloss_dict in loss_dict.values():
        loss += subloss_dict["loss"] * subloss_dict["scaling_factor"]
    return loss


def _sum_loss_dicts(total_ld, batch_ld):
    """ Add all values in :code:`batch_ld` into the corresponding values in :code:`total_ld`

    Args
    ----
    total_ld, batch_ld : dict
        Dictionary with keys :code:`loss_fn_name` and values of dictionaries with
        - :code:`loss` corresponding to the loss value
        - :code:`count` corresponding to the normalizing factor
        - :code:`scaling_factor` corresponding to the scaling coefficient in the loss function

    Returns
    -------
    combined_ld : dict
        Combined loss dictionary with the same structure as the input dictionaries
    """
    def _weighted_loss_sum(ld1, ld2):
        c1, c2 = ld1["count"], ld2["count2"]
        return (ld1["loss"] * c1 + ld2["loss"] * c2)/(c1 + c2)
    # combine the two loss dictionaries
    combined_ld = total_ld.copy()
    for loss_fn_name, batch_subld in batch_ld.items():
        if loss_fn_name not in combined_ld.keys():
            combined_ld[loss_fn_name] = batch_subld
        else:
            combined_subld = combined_ld[loss_fn_name]
            assert combined_subld["scaling_factor"] == batch_subld["scaling_factor"]
            combined_subld["loss"] = _weighted_loss_sum(combined_subld, batch_subld)
            combined_subld["count"] += batch_subld["count"]
    return combined_ld


def run_epoch(model, dataloader, loss_fn, optimizer=None, scheduler=None, grad=False, test=False, dev="cuda:0"):
    """ Run :code:`model` on one epoch of :code:`dataloader`

    Args
    ----
    model : terminator.model.TERMinator.TERMinator
        An instance of TERMinator
    dataloader : torch.utils.data.DataLoader
        A torch DataLoader that wraps either terminator.data.data.TERMDataLoader or terminator.data.data.TERMLazyDataLoader
    loss_fn : function
        Loss function with signature :code:`loss_fn(etab, E_idx, data)` and returns :code`loss, batch_count`,
        where
        - :code:`etab, E_idx` is the outputted Potts Model
        - :code:`data` is the input data produced by :code:`dataloader`
        - :code:`hparams` is the model hyperparameters
        - :code:`loss` is the loss value
        - :code:`batch_count` is the averaging factor
    optimizer : torch optimizer or None
        An optimizer for :code:`model`. Used when :code:`grad=True, test=False`
    scheduler : torch scheduler or None
        The associted scheduler for the given optimizer
    grad : bool
        Whether or not to compute gradients. :code:`True` to train the model, :code:`False` to use model in evaluation mode.
    test : bool
        Whether or not to save the outputs of the model. Requires :code:`grad=False`.
    dev : str, default="cuda:0"
        What device to compute on

    Returns
    -------
    epoch_loss : float
        Loss on the run epoch
    avg_prob : float
        Average probability of native residue pairs, averaged across all kNN edges
    dump : list of dicts, conditionally present
        Outputs the model. Present when :code:`test=True`
    """
    # arg checking
    if test:
        assert not grad, "grad should not be on for test set"
    if grad:
        assert optimizer is not None, "require an optimizer if using grads"
    if scheduler is not None:
        assert optimizer is not None, "using a scheduler requires an optimizer"

    running_loss_dict = {}

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
    for data in dataloader:
        # a small hack for DataParallel to know which device got which proteins
        data['scatter_idx'] = torch.arange(len(data['seq_lens']))
        _to_dev(data, dev)
        max_seq_len = max(data['seq_lens'].tolist())
        ids = data['ids']

        try:
            etab, E_idx = model(data, max_seq_len)
            batch_loss_dict = loss_fn(etab, E_idx, data)
            loss = _compute_loss(batch_loss_dict)
        except Exception as e:
            print(ids)
            raise e

        running_loss_dict = _sum_loss_dicts(running_loss_dict, batch_loss_dict)

        if grad:
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

        avg_loss = _compute_loss(running_loss_dict)
        term_mask_eff = int((~data['src_key_mask']).sum().item() / data['src_key_mask'].numel() * 100)
        res_mask_eff = int(data['x_mask'].sum().item() / data['x_mask'].numel() * 100)

        progress.update(1)
        progress.refresh()
        progress.set_description_str(f'avg loss {avg_loss} | eff 0.{term_mask_eff},0.{res_mask_eff}')

    progress.close()
    epoch_loss = _compute_loss(running_loss_dict)

    if scheduler is not None:
        scheduler.step(epoch_loss)

    torch.set_grad_enabled(True)  # just to be safe
    if test:
        return epoch_loss, dump
    return epoch_loss, None
