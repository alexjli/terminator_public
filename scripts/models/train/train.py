"""Train TERMinator model.

Usage:
    .. code-block::

        python train.py \\
            --dataset <dataset_dir> \\
            --hparams <hparams_file> \\
            --run_dir <run_dir> \\
            [--train <train_split_file>] \\
            [--validation <val_split_file>] \\
            [--test <test_split_file>] \\
            [--out_dir <out_dir>] \\
            [--dev <device>] \\
            [--epochs <num_epochs>]
            [--lazy]

    If :code:`--out_dir <out_dir>` is not set, :code:`net.out` will be dumped
    into :code:`<run_dir>`.

    For any of the split files, if the option is not provided, :code:`train.py` will
    look for them within :code:`<dataset_dir>`.

See :code:`python train.py --help` for more info.
"""

import argparse
import copy
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from terminator.data.data import (LazyDataset, TERMDataLoader, TERMDataset,
                                  TERMLazyDataLoader)
from terminator.models.TERMinator import MultiChainTERMinator_gcnkt
from terminator.utils.loop_utils import run_epoch

# for autosummary import purposes
sys.path.insert(0, os.path.dirname(__file__))
from default_hparams import DEFAULT_HPARAMS
from noam_opt import get_std_opt

try:
    import horovod.torch as hvd
except ImportError:
    pass

torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=1000)
torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(profile="full")


def main(args):
    dev = args.dev
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    train_dataloader, val_dataloader, test_dataloader = None, None, None
    kwargs = {}
    kwargs['num_workers'] = 16

    # load hparams
    hparams = json.load(open(args.hparams, 'r'))
    for key in DEFAULT_HPARAMS:
        if key not in hparams:
            hparams[key] = DEFAULT_HPARAMS[key]

    hparams_path = os.path.join(run_dir, 'hparams.json')
    if os.path.isfile(hparams_path):
        previous_hparams = json.load(open(hparams_path, 'r'))
        if previous_hparams != hparams:
            raise Exception('Given hyperparameters do not agree with previous hyperparameters.')
    else:
        json.dump(hparams, open(hparams_path, 'w'))

    # set up dataloaders
    train_ids = []
    with open(args.train, 'r') as f:
        for line in f:
            train_ids += [line[:-1]]
    validation_ids = []
    with open(args.validation, 'r') as f:
        for line in f:
            validation_ids += [line[:-1]]
    test_ids = []
    with open(args.test, 'r') as f:
        for line in f:
            test_ids += [line[:-1]]
    if args.lazy:
        train_dataset = LazyDataset(args.dataset, pdb_ids=train_ids)
        val_dataset = LazyDataset(args.dataset, pdb_ids=validation_ids)
        test_dataset = LazyDataset(args.dataset, pdb_ids=test_ids)

        train_batch_sampler = TERMLazyDataLoader(train_dataset,
                                                 batch_size=hparams['train_batch_size'],
                                                 shuffle=hparams['shuffle'],
                                                 semi_shuffle=hparams['semi_shuffle'],
                                                 sort_data=hparams['sort_data'],
                                                 term_matches_cutoff=hparams['term_matches_cutoff'],
                                                 max_term_res=hparams['max_term_res'],
                                                 max_seq_tokens=hparams['max_seq_tokens'],
                                                 term_dropout=hparams['term_dropout'])
        test_term_matches_cutoff = hparams[
            'test_term_matches_cutoff'] if 'test_term_matches_cutoff' in hparams else hparams['term_matches_cutoff']
        val_batch_sampler = TERMLazyDataLoader(val_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               term_matches_cutoff=test_term_matches_cutoff)
        test_batch_sampler = TERMLazyDataLoader(test_dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                term_matches_cutoff=test_term_matches_cutoff)
    else:
        train_dataset = TERMDataset(args.dataset, pdb_ids=train_ids)
        val_dataset = TERMDataset(args.dataset, pdb_ids=validation_ids)
        test_dataset = TERMDataset(args.dataset, pdb_ids=test_ids)

        train_batch_sampler = TERMDataLoader(train_dataset,
                                             batch_size=hparams['train_batch_size'],
                                             shuffle=hparams['shuffle'],
                                             semi_shuffle=hparams['semi_shuffle'],
                                             sort_data=hparams['sort_data'])
        val_batch_sampler = TERMDataLoader(val_dataset, batch_size=1, shuffle=False)
        test_batch_sampler = TERMDataLoader(test_dataset, batch_size=1, shuffle=False)

    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_batch_sampler,
                                  collate_fn=train_batch_sampler._package,
                                  pin_memory=True,
                                  **kwargs)
    val_dataloader = DataLoader(val_dataset,
                                batch_sampler=val_batch_sampler,
                                collate_fn=val_batch_sampler._package,
                                pin_memory=True,
                                **kwargs)
    test_dataloader = DataLoader(test_dataset,
                                 batch_sampler=test_batch_sampler,
                                 collate_fn=test_batch_sampler._package,
                                 **kwargs)

    terminator = MultiChainTERMinator_gcnkt(hparams=hparams, device=dev)
    print(terminator)
    print("hparams", terminator.hparams)

    if hparams['finetune']: # freeze all but the last output layer
        for (name, module) in terminator.named_children():
            if name == "top":
                for (n, m) in module.named_children():
                    if n == "W_out":
                        m.requires_grad = True
                        print("top.{} unfrozen".format(n))
                    else:
                        m.requires_grad = False
                        print("top.{} frozen".format(n))
            else:
                module.requires_grad = False
                print("{} frozen".format(name))

    if torch.cuda.device_count() > 1 and dev != "cpu":
        terminator = nn.DataParallel(terminator)
        terminator_module = terminator.module
    else:
        terminator_module = terminator
    terminator.to(dev)
    """
    optimizer = optim.Adam(terminator.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience = 5, verbose = True)
    """
    optimizer = get_std_opt(terminator.parameters(),
                            d_model=hparams['energies_hidden_dim'],
                            regularization=hparams['regularization'])
    scheduler = None

    save = []

    # load checkpoint
    if os.path.isfile(os.path.join(run_dir, 'net_best_checkpoint.pt')):
        best_checkpoint_state = torch.load(os.path.join(run_dir, 'net_best_checkpoint.pt'))
        last_checkpoint_state = torch.load(os.path.join(run_dir, 'net_last_checkpoint.pt'))
        best_checkpoint = best_checkpoint_state['state_dict']
        best_validation = best_checkpoint_state['val_prob']
        start_epoch = last_checkpoint_state['epoch'] + 1
        terminator_module.load_state_dict(last_checkpoint_state['state_dict'])
        optimizer.load_state_dict(last_checkpoint_state['optimizer_state'])
        with open(os.path.join(run_dir, 'training_curves.pk'), 'rb') as fp:
            save = pickle.load(fp)
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'), purge_step=start_epoch + 1)
    else:
        best_checkpoint = None
        best_validation = -1
        start_epoch = 0
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))

    try:
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(start_epoch, args.epochs):
            print('epoch', epoch)

            epoch_loss, avg_prob = run_epoch(terminator, train_dataloader, optimizer=optimizer, grad=True, dev=dev)

            print('epoch loss', epoch_loss, '| approx epoch prob', avg_prob)
            writer.add_scalar('training loss', epoch_loss, epoch)
            writer.add_scalar('approx training prob', avg_prob, epoch)

            # validate
            val_loss, val_prob = run_epoch(terminator, val_dataloader, scheduler=scheduler, grad=False, dev=dev)

            print('val loss', val_loss, '| approx val prob', val_prob)
            writer.add_scalar('val loss', val_loss, epoch)
            writer.add_scalar('val prob', val_prob, epoch)
            save.append([epoch_loss, val_loss])

            if val_prob > best_validation:
                best_validation = val_prob
                best_checkpoint = copy.deepcopy(terminator_module.state_dict())
                checkpoint_state = {
                    'epoch': epoch,
                    'state_dict': best_checkpoint,
                    'best_model': True,
                    'val_prob': best_validation,
                    'optimizer_state': optimizer.state_dict()
                }
                torch.save(checkpoint_state, os.path.join(run_dir, 'net_best_checkpoint.pt'))
                torch.save(checkpoint_state, os.path.join(run_dir, 'net_last_checkpoint.pt'))
            else:
                checkpoint_state = {
                    'epoch': epoch,
                    'state_dict': terminator_module.state_dict(),
                    'best_model': False,
                    'val_prob': val_prob,
                    'optimizer_state': optimizer.state_dict()
                }
                torch.save(checkpoint_state, os.path.join(run_dir, 'net_last_checkpoint.pt'))

            with open(os.path.join(run_dir, 'training_curves.pk'), 'wb') as fp:
                pickle.dump(save, fp)

    except KeyboardInterrupt:
        pass

    print(save)
    torch.save(terminator_module.state_dict(), os.path.join(run_dir, 'net_last.pt'))
    torch.save(best_checkpoint, os.path.join(run_dir, 'net_best.pt'))
    with open(os.path.join(run_dir, 'training_curves.pk'), 'wb') as fp:
        pickle.dump(save, fp)

    # test
    terminator_module.load_state_dict(best_checkpoint)
    test_loss, test_prob, dump = run_epoch(terminator, test_dataloader, grad=False, test=True, dev=dev)
    print(f"test loss {test_loss} test prob {test_prob}")

    if args.out_dir:
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)
        net_out_path = os.path.join(args.out_dir, "net.out")
    else:
        net_out_path = os.path.join(run_dir, "net.out")

    # save etab outputs for dTERMen runs
    with open(net_out_path, 'wb') as fp:
        pickle.dump(dump, fp)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train TERMinator!')
    parser.add_argument('--dataset',
                        help='input folder .features files in proper directory structure.',
                        required=True)
    parser.add_argument('--hparams',
                        help='hparams file path',
                        required=True)
    parser.add_argument('--run_dir',
                        help='path to place folder to store model files',
                        required=True)
    parser.add_argument('--train',
                        help='file with training dataset split')
    parser.add_argument('--validation',
                        help='file with validation dataset split')
    parser.add_argument('--test',
                        help='file with test dataset split')
    parser.add_argument('--out_dir',
                        help='path to place test set eval results (e.g. net.out). If not set, default to --run_dir')
    parser.add_argument('--dev',
                        help='device to train on',
                        default='cuda:0')
    parser.add_argument('--epochs',
                        help='number of epochs to train for',
                        default=100,
                        type=int)
    parser.add_argument('--lazy',
                        help="use lazy data loading",
                        type=bool,
                        default=True)
    args = parser.parse_args()

    # by default, if no splits are provided, read the splits from the dataset folder
    if args.train is None:
        args.train = os.path.join(args.dataset, 'train.in')
    if args.validation is None:
        args.validation = os.path.join(args.dataset, 'validation.in')
    if args.test is None:
        args.test = os.path.join(args.dataset, 'test.in')

    main(args)
