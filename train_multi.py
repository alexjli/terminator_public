from TERMinator import *
from data import *
from train_utils import *
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
from struct2seq.noam_opt import *
import argparse
import os
import sys
import copy
import json
from hparams.default import DEFAULT_HPARAMS
try:
    import horovod.torch as hvd
except ImportError:
    pass

INPUT_DATA = '/home/gridsan/alexjli/keatinglab_shared/alexjli/TERMinator/'
OUTPUT_DIR = '/home/gridsan/alexjli/keatinglab_shared/alexjli/TERMinator_runs/'
torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=1000)
torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(profile="full")

def main(args):
    dev = args.dev
    run_output_dir = os.path.join(OUTPUT_DIR, args.run_name)
    if not os.path.isdir(run_output_dir):
        os.makedirs(run_output_dir)
    train_dataloader, val_dataloader, test_dataloader = None, None, None
    kwargs = {}
    kwargs['num_workers'] = 16

    # load hparams
    hparams = json.load(open(args.hparams, 'r'))
    for key in DEFAULT_HPARAMS:
        if key not in hparams:
            hparams[key] = DEFAULT_HPARAMS[key]

    hparams_path = os.path.join(run_output_dir, 'hparams.json')
    if os.path.isfile(hparams_path):
        previous_hparams = json.load(open(hparams_path, 'r'))
        if previous_hparams != hparams:
            raise Exception('Given hyperparameters do not agree with previous hyperparameters.')
    else:
        json.dump(hparams, open(hparams_path, 'w'))

    # set up dataloaders
    train_ids = []
    with open(os.path.join(INPUT_DATA, args.dataset, args.train), 'r') as f:
        for line in f:
            train_ids += [line[:-1]]
    validation_ids = []
    with open(os.path.join(INPUT_DATA, args.dataset, args.validation), 'r') as f:
        for line in f:
            validation_ids += [line[:-1]]
    test_ids = []
    with open(os.path.join(INPUT_DATA, args.dataset, args.test), 'r') as f:
        for line in f:
            test_ids += [line[:-1]]
    train_dataset = LazyDataset(os.path.join(INPUT_DATA, args.dataset), pdb_ids = train_ids) 
    val_dataset = LazyDataset(os.path.join(INPUT_DATA, args.dataset), pdb_ids = validation_ids)
    test_dataset = LazyDataset(os.path.join(INPUT_DATA, args.dataset), pdb_ids = test_ids)
    
    train_batch_sampler = TERMLazyDataLoader(train_dataset, batch_size=hparams['train_batch_size'], shuffle=hparams['shuffle'], semi_shuffle = hparams['semi_shuffle'],  sort_data=hparams['sort_data'])
    val_batch_sampler = TERMLazyDataLoader(val_dataset, batch_size=1, shuffle=False)
    test_batch_sampler = TERMLazyDataLoader(test_dataset, batch_size=1, shuffle=False)

    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler = train_batch_sampler,
                                  collate_fn = train_batch_sampler._package,
                                  pin_memory=True,
                                  **kwargs)
    val_dataloader = DataLoader(val_dataset, 
                                batch_sampler = val_batch_sampler,
                                collate_fn = val_batch_sampler._package,
                                pin_memory=True,
                                **kwargs)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_sampler = test_batch_sampler,
                                 collate_fn = test_batch_sampler._package,
                                 **kwargs)

    terminator = MultiChainTERMinator_gcnkt(hparams = hparams, device = dev)
    print(terminator)
    print("hparams", terminator.hparams)

    if torch.cuda.device_count() > 1:
        terminator = nn.DataParallel(terminator)
        terminator_module = terminator.module
    else:
        terminator_module = terminator
    terminator.to(dev)

    """
    optimizer = optim.Adam(terminator.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience = 5, verbose = True)
    """
    optimizer = get_std_opt(terminator.parameters(), d_model = hparams['energies_hidden_dim'], regularization = hparams['regularization'])
    scheduler = None
    

    save = []
    
    # load checkpoint
    if os.path.isfile(os.path.join(run_output_dir, 'net_best_checkpoint.pt')):
        best_checkpoint_state = torch.load(os.path.join(run_output_dir, 'net_best_checkpoint.pt'))
        last_checkpoint_state = torch.load(os.path.join(run_output_dir, 'net_last_checkpoint.pt'))
        best_checkpoint = best_checkpoint_state['state_dict']
        best_validation = best_checkpoint_state['val_prob']
        start_epoch = last_checkpoint_state['epoch'] + 1
        terminator_module.load_state_dict(last_checkpoint_state['state_dict'])
        optimizer.load_state_dict(last_checkpoint_state['optimizer_state'])
        with open(os.path.join(run_output_dir, 'training_curves.pk'), 'rb') as fp:
            save = pickle.load(fp)
        writer = SummaryWriter(log_dir = os.path.join(run_output_dir, 'tensorboard'), purge_step = start_epoch+1)
    else:
        best_checkpoint = None
        best_validation = -1  
        start_epoch = 0
        writer = SummaryWriter(log_dir = os.path.join(run_output_dir, 'tensorboard'))

    try:
        #torch.autograd.set_detect_anomaly(True)
        for epoch in range(start_epoch, args.epochs):
            print('epoch', epoch)

            epoch_loss, avg_prob = run_epoch(terminator, train_dataloader, optimizer = optimizer, grad = True, dev = dev)

            print('epoch loss', epoch_loss, '| approx epoch prob', avg_prob)
            writer.add_scalar('training loss', epoch_loss, epoch)
            writer.add_scalar('approx training prob', avg_prob, epoch)

            # validate
            val_loss, val_prob = run_epoch(terminator, val_dataloader, scheduler = scheduler, grad = False, dev = dev)

            print('val loss', val_loss, '| approx val prob', val_prob)
            writer.add_scalar('val loss', val_loss, epoch)
            writer.add_scalar('val prob', val_prob, epoch)
            save.append([epoch_loss, val_loss])

            if val_prob > best_validation:
                best_validation = val_prob
                best_checkpoint = copy.deepcopy(terminator_module.state_dict())
                checkpoint_state = {'epoch': epoch, 'state_dict': best_checkpoint, 'best_model': True, 'val_prob': best_validation, 'optimizer_state': optimizer.state_dict()}
                torch.save(checkpoint_state, os.path.join(run_output_dir, 'net_best_checkpoint.pt'))
                torch.save(checkpoint_state, os.path.join(run_output_dir, 'net_last_checkpoint.pt'))
            else:
                checkpoint_state = {'epoch': epoch, 'state_dict': terminator_module.state_dict(), 'best_model': False, 'val_prob': val_prob, 'optimizer_state': optimizer.state_dict()}
                torch.save(checkpoint_state, os.path.join(run_output_dir, 'net_last_checkpoint.pt'))

            with open(os.path.join(run_output_dir, 'training_curves.pk'), 'wb') as fp:
                pickle.dump(save, fp)

    except KeyboardInterrupt:
        pass

    print(save)
    torch.save(terminator_module.state_dict(), os.path.join(run_output_dir, 'net_last.pt'))
    torch.save(best_checkpoint, os.path.join(run_output_dir, 'net_best.pt'))
    with open(os.path.join(run_output_dir, 'training_curves.pk'), 'wb') as fp:
        pickle.dump(save, fp)

    # test
    terminator_module.load_state_dict(best_checkpoint)
    test_loss, test_prob, dump = run_epoch(terminator, test_dataloader, grad = False, test = True, dev = dev)
    print(f"test loss {test_loss} test prob {test_prob}")
    # save etab outputs for dTERMen runs
    with open(run_output_dir + '/net.out', 'wb') as fp:
        pickle.dump(dump, fp)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train TERMinator!')
    parser.add_argument('--dataset', help = 'input folder .features files in proper directory structure. prefix is $INPUT_DATA/', default = "features_singlechain")
    parser.add_argument('--dev', help = 'device to train on', default = 'cuda:0')
    parser.add_argument('--train', help = 'file with training dataset', default = 'train.in')
    parser.add_argument('--validation', help = 'file with validation dataset', default = 'validation.in')
    parser.add_argument('--test', help = 'file with test dataset', default = 'test.in')
    # parser.add_argument('--shuffle_splits', help = 'shuffle dataset before creating train, validate, test splits', default = False, type=bool)
    parser.add_argument('--run_name', help = 'name for run, to use for output subfolder', default = 'test_run')
    parser.add_argument('--epochs', help = 'number of epochs to train for', default = 100, type=int)
    parser.add_argument('--hparams', help = 'hparams file name', default = 'hparams/default.json')
    args = parser.parse_args()
    main(args)
 
