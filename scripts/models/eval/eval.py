"""Perform inference with a trained TERMinator model.

The resulting evaluated proteins will be dumped in :code:`<output_dir>` via
a pickle file :code:`net.out`.

Usage:
    .. code-block::

        python eval.py \\
            --dataset <dataset_dir> \\
            --model_dir <trained_model_dir> \\
            --output_dir <output_dir> \\
            [--subset <data_subset_file>] \\
            [--dev <device>]

    If :code:`subset` is not provided, the entire dataset :code:`dataset` will
    be evaluated.

See :code:`python eval.py --help` for more info.
"""

import argparse
import json
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from terminator.data.data import TERMLazyDataset, TERMLazyBatchSampler
from terminator.models.TERMinator import TERMinator
from terminator.utils.model.loop_utils import run_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval TERMinator Psuedoperplexity')
    parser.add_argument('--dataset', help='input folder .features files in proper directory structure', required=True)
    parser.add_argument('--model_dir', help='trained model folder', required=True)
    parser.add_argument('--output_dir', help='where to dump net.out', required=True)
    parser.add_argument('--subset',
                        help=('file specifiying subset of dataset to evaluate. '
                              'if none provided, the whole dataset folder will be evaluated'))
    parser.add_argument('--dev', help='device to train on', default='cuda:0')
    args = parser.parse_args()

    dev = args.dev
    if args.subset:
        test_ids = []
        with open(os.path.join(args.subset), 'r') as f:
            for line in f:
                test_ids += [line[:-1]]
    else:
        test_ids = None

    test_dataset = TERMLazyDataset(args.dataset, pdb_ids=test_ids)
    test_batch_sampler = TERMLazyBatchSampler(test_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_sampler=test_batch_sampler,
                                 collate_fn=test_batch_sampler._package)

    with open(os.path.join(args.model_dir, "hparams.json")) as fp:
        hparams = json.load(fp)

    # backwards compatability
    if "cov_features" not in hparams.keys():
        hparams["cov_features"] = False
    if "term_use_mpnn" not in hparams.keys():
        hparams["term_use_mpnn"] = False
    if "matches" not in hparams.keys():
        hparams["matches"] = "resnet"
    if "struct2seq_linear" not in hparams.keys():
        hparams['struct2seq_linear'] = False
    if "energies_gvp" not in hparams.keys():
        hparams['energies_gvp'] = False
    if "num_sing_stats" not in hparams.keys():
        hparams['num_sing_stats'] = 0
    if "num_pair_stats" not in hparams.keys():
        hparams['num_pair_stats'] = 0
    if "contact_idx" not in hparams.keys():
        hparams['contact_idx'] = False
    if "fe_dropout" not in hparams.keys():
        hparams['fe_dropout'] = 0.1
    if "fe_max_len" not in hparams.keys():
        hparams['fe_max_len'] = 1000
    if "cie_dropout" not in hparams.keys():
        hparams['cie_dropout'] = 0.1

    terminator = TERMinator(hparams=hparams, device=dev)
    terminator = nn.DataParallel(terminator)

    best_checkpoint_state = torch.load(os.path.join(args.model_dir, 'net_best_checkpoint.pt'), map_location=dev)
    best_checkpoint = best_checkpoint_state['state_dict']
    terminator.module.load_state_dict(best_checkpoint)
    terminator.to(dev)

    test_sum = 0
    test_weights = 0

    terminator.eval()
    # test
    test_loss, test_prob, dump = run_epoch(terminator, test_dataloader, grad=False, test=True, dev=dev)
    print(f"test loss {test_loss} test prob {test_prob}")

    # save etab outputs for dTERMen runs
    with open(os.path.join(args.output_dir, 'net.out'), 'wb') as fp:
        pickle.dump(dump, fp)
