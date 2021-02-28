from TERMinator import *
from data import *
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
try:
    import horovod.torch as hvd
except ImportError:
    pass

INPUT_DATA = '/nobackup/users/alexjli/TERMinator/'
OUTPUT_DIR = '/nobackup/users/alexjli/TERMinator/'

DEFAULT_HPARAMS = {
            'hidden_dim': 32,
            'gradient_checkpointing': True,
            'cov_features': True,
            'resnet_blocks': 4,
            'term_layers': 4,
            'conv_filter': 3,
            'matches_layers': 4,
            'matches_num_heads': 4,
            'term_heads': 4,
            'k_neighbors': 30,
            'fe_dropout': 0.1,
            'fe_max_len': 1000,
            'transformer_dropout': 0.1,
            'energies_num_letters': 20,
            'energies_encoder_layers': 3,
            'energies_decoder_layers': 3,
            'energies_vocab': 20,
            'energies_protein_features': 'full',
            'energies_augment_eps': 0,
            'energies_dropout': 0.1,
            'energies_forward_attention_decoder': True,
            'energies_use_mpnn': False,
            'energies_output_dim': 20*20,
            'resnet_linear': False,
            'transformer_linear': False,
            'struct2seq_linear': False,
            'use_terms': True,
            'train_batch_size': 16,
            'regularization': 0,
            'num_features': len(['sin_phi', 'sin_psi', 'sin_omega', 'cos_phi', 'cos_psi', 'cos_omega', 'env', 'rmsd', 'term_len'])
        }

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
    if args.horovod:
        hvd.init()
        torch.set_num_threads(1)
        torch.cuda.manual_seed(0)
        if hasattr(mp, '_supports_context') and mp._supports_context and 'forkserver' in mp.get_all_start_methods():
            kwargs['multiprocessing_context'] = 'forkserver'
            kwargs['num_workers'] = 1
    else:
        kwargs['num_workers'] = 16

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

    if args.lazy:
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
        
        if args.horovod:
            train_batch_sampler = TERMLazyDistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True, batch_size=hparams['train_batch_size'])
            val_batch_sampler = TERMLazyDistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False, batch_size=1)
            test_batch_sampler = TERMLazyDistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False, batch_size=1)
        else:
            train_batch_sampler = TERMLazyDataLoader(train_dataset, batch_size=hparams['train_batch_size'], shuffle=True, sort_data=True)
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
    else:
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
        train_dataset = TERMDataset(os.path.join(INPUT_DATA, args.dataset), pdb_ids = train_ids) 
        val_dataset = TERMDataset(os.path.join(INPUT_DATA, args.dataset), pdb_ids = validation_ids)
        test_dataset = TERMDataset(os.path.join(INPUT_DATA, args.dataset), pdb_ids = test_ids)

        train_dataloader = TERMDataLoader(train_dataset, batch_size=hparams['train_batch_size'], shuffle = True)
        val_dataloader = TERMDataLoader(val_dataset, batch_size=1, shuffle=False)
        test_dataloader = TERMDataLoader(test_dataset, batch_size=1, shuffle=False)


    terminator = MultiChainTERMinator_g(hparams = hparams, device = dev)
    if torch.cuda.device_count() > 1:
        if args.horovod:
            torch.cuda.set_device(hvd.local_rank())
            terminator_module = terminator
        else:
            terminator = nn.DataParallel(terminator)
            terminator_module = terminator.module
    else:
        terminator_module = terminator
    terminator.to(dev)

    if args.horovod:
        lr_multiplier = hvd.size()
    else:
        lr_multiplier = 1

    optimizer = get_std_opt(terminator.parameters(), d_model = hparams['hidden_dim'], lr_multiplier=lr_multiplier, regularization = hparams['regularization'])

    if args.horovod:
        hvd.broadcast_parameters(terminator.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=terminator.named_parameters())

    lam = 0.1
    max_grad_norm = 1

    save = []
    
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
        for epoch in range(start_epoch, 50):
            print('epoch', epoch)

            # train
            if args.horovod:
                train_batch_sampler.set_epoch(epoch)
            running_loss = 0
            running_prob = 0
            count = 0
            terminator.train()
            
            train_progress = tqdm(total=len(train_dataloader))
            for i, data in enumerate(train_dataloader):
                msas = data['msas'].to(dev)
                features = data['features'].to(dev).float()
                focuses = data['focuses'].to(dev)
                src_key_mask = data['src_key_mask'].to(dev)
                X = data['X'].to(dev)
                x_mask = data['x_mask'].to(dev)
                seqs = data['seqs'].to(dev)
                seq_lens = data['seq_lens'].to(dev)
                term_lens = data['term_lens'].to(dev)
                max_seq_len = max(seq_lens.tolist())
                chain_lens = data['chain_lens']
                ppoe = data['ppoe'].to(dev).float()


                loss, rms, prob = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs, max_seq_len, ppoe, chain_lens)

                if torch.cuda.device_count() > 1 and not args.horovod:
                    loss = loss.mean()
                    prob = prob.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(terminator.parameters(), max_grad_norm)
                optimizer.step()

                running_loss += loss.item()
                running_prob += prob.item()
                count = i

                avg_loss = running_loss / (count + 1)
                avg_prob = running_prob / (count + 1)
                train_progress.update(1)
                train_progress.refresh()
                train_progress.set_description_str('avg loss {} | avg prob {} '.format(avg_loss, avg_prob))

            train_progress.close()
            epoch_loss = running_loss / (count+1)
            avg_prob = running_prob / (count+1)
            print('epoch loss', epoch_loss, '| approx epoch prob', avg_prob)
            writer.add_scalar('training loss', epoch_loss, epoch)
            writer.add_scalar('approx training prob', avg_prob, epoch)

            # validate
            with torch.no_grad():
                running_loss = 0
                running_prob = 0
                count = 0
                terminator.eval()
                recovery = []

                val_progress = tqdm(total=len(val_dataloader))
                for i, data in enumerate(val_dataloader):
                    msas = data['msas'].to(dev)
                    features = data['features'].to(dev).float()
                    focuses = data['focuses'].to(dev)
                    src_key_mask = data['src_key_mask'].to(dev)
                    X = data['X'].to(dev)
                    x_mask = data['x_mask'].to(dev)
                    seqs = data['seqs'].to(dev)
                    seq_lens = data['seq_lens'].to(dev)
                    term_lens = data['term_lens'].to(dev)
                    max_seq_len = max(seq_lens.tolist())
                    chain_lens = data['chain_lens']
                    ppoe = data['ppoe'].to(dev).float()


                    loss, rms, prob = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs, max_seq_len, ppoe, chain_lens)
                    if torch.cuda.device_count() > 1 and not args.horovod:
                        loss = loss.mean()
                        prob = prob.mean()
                    running_loss += loss.item()
                    running_prob += prob.item()
                    count = i

                    p_recov = terminator_module.percent_recovery(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs, max_seq_len, ppoe, chain_lens)
                    recovery.append(p_recov)
                    
                    val_progress.update(1)
                    val_progress.refresh()
                    val_progress.set_description_str('point loss {} | point prob {}'.format(loss.item(), prob.item()))

                val_progress.close()
                val_loss = running_loss / (count+1)
                val_prob = running_prob / (count+1)
                print('val loss', val_loss, '| approx val prob', val_prob)
                writer.add_scalar('val loss', val_loss, epoch)
                writer.add_scalar('approx val prob', val_prob, epoch)
                writer.add_scalar('val p recov', torch.stack(recovery).mean().item(), epoch)
                print('val p recov', torch.stack(recovery).mean().item())
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

    dump = []
    recovery = []
    terminator_module.load_state_dict(best_checkpoint)

    terminator_module.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            print('datapoint', i)
            msas = data['msas'].to(dev)
            features = data['features'].to(dev).float()
            focuses = data['focuses'].to(dev)
            src_key_mask = data['src_key_mask'].to(dev)
            X = data['X'].to(dev)
            x_mask = data['x_mask'].to(dev)
            seqs = data['seqs'].to(dev)
            seq_lens = data['seq_lens'].to(dev)
            ids = data['ids']
            term_lens = data['term_lens'].to(dev)
            max_seq_len = max(seq_lens.tolist())
            chain_lens = data['chain_lens']
            ppoe = data['ppoe'].to(dev).float()

            loss, rms, prob = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs, max_seq_len, ppoe, chain_lens)
            if torch.cuda.device_count() > 1 and not args.horovod:
                loss = loss.mean()
                prob = prob.mean()
            
            print(loss.item(), prob.item())

            output, E_idx = terminator_module.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe, chain_lens)
            
            pred_seqs = terminator_module.pred_sequence(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe, chain_lens)
            opt_seqs = terminator_module.opt_sequence(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs, max_seq_len, ppoe, chain_lens)
            print(ids)
            p_recov = terminator_module.percent_recovery(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs, max_seq_len, ppoe, chain_lens)
            print('pred', pred_seqs)
            print('opt', opt_seqs)
            print('p recov', p_recov)
            print('prob', prob)
            recovery.append(p_recov)

            n_batch, l, n = output.shape[:3]
            """
            print('out', output.view(n_batch, l, n, 22, 22)[0, 0, 0, ...])
            print('idx', E_idx[0,0, ...])
            print('label', etab.to_dense().view(n_batch, l, l, 22, 22)[0, 0, 0, ...])
            print()
            """
            dump.append({'out': output.view(n_batch, l, n, 20, 20).cpu().numpy(),
                         'idx': E_idx.cpu().numpy(),
                         'ids': ids
                        })

    print('avg p recov:', torch.stack(recovery).mean())

    with open(run_output_dir + '/net.out', 'wb') as fp:
        pickle.dump(dump, fp)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train TERMinator!')
    parser.add_argument('--dataset', help = 'input folder .features files in proper directory structure. prefix is $INPUT_DATA/', default = "features_singlechain")
    parser.add_argument('--dev', help = 'device to train on', default = 'cuda:0')
    parser.add_argument('--lazy', help = 'use lazy data loading', default = True, type = bool)
    parser.add_argument('--train', help = 'file with training dataset', default = 'train.in')
    parser.add_argument('--validation', help = 'file with validation dataset', default = 'validation.in')
    parser.add_argument('--test', help = 'file with test dataset', default = 'test.in')
    # parser.add_argument('--shuffle_splits', help = 'shuffle dataset before creating train, validate, test splits', default = False, type=bool)
    parser.add_argument('--run_name', help = 'name for run, to use for output subfolder', default = 'test_run')
    parser.add_argument('--hparams', help = 'hparams file name', default = 'hparams/default.json')
    parser.add_argument('--horovod', help = 'use Horovod for parallelization', default = False, type = bool)
    args = parser.parse_args()
    main(args)
 
