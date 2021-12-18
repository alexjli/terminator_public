from terminator.models.TERMinator import *
from terminator.data.data import *
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import os
import sys
import copy
import json
from terminator.utils.loop_utils import run_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval TERMinator Psuedoperplexity')
    parser.add_argument('--dataset', help='input folder .features files in proper directory structure')
    parser.add_argument('--subset', help='file specifiying subset of dataset to evaluate', default='test.in')
    parser.add_argument('--output_dir', help='where to dump net.out')
    parser.add_argument('--model_dir', help='trained model folder')
    parser.add_argument('--dev', help='device to train on', default='cuda:0')
    args = parser.parse_args()

    dev = args.dev
    test_ids = []
    with open(os.path.join(args.dataset, args.subset), 'r') as f:
        for line in f:
            test_ids += [line[:-1]]
    test_dataset = LazyDataset(args.dataset, pdb_ids=test_ids)
    test_batch_sampler = TERMLazyDataLoader(test_dataset, batch_size=1, shuffle=False)
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

    terminator = MultiChainTERMinator_gcnkt(hparams=hparams, device=dev)
    terminator = nn.DataParallel(terminator)

    best_checkpoint_state = torch.load(
        os.path.join(args.model_dir, 'net_best_checkpoint.pt'),
        map_location=dev
    )
    best_checkpoint = best_checkpoint_state['state_dict']
    terminator.module.load_state_dict(best_checkpoint)
    terminator.to(dev)

    test_sum = 0
    test_weights = 0

    terminator.eval()
    # test
    test_loss, test_prob, dump = run_epoch(
        terminator,
        test_dataloader,
        grad=False,
        test=True,
        dev=dev
    )
    print(f"test loss {test_loss} test prob {test_prob}")

    # save etab outputs for dTERMen runs
    with open(os.path.join(args.output_dir, 'net.out'), 'wb') as fp:
        pickle.dump(dump, fp)
