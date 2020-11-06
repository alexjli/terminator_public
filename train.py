from TERMinator import *
from data import *
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from struct2seq.noam_opt import *
import argparse
import os

INPUT_DATA = '/nobackup1c/users/vsundar/TERMinator/'
OUTPUT_DIR = '/nobackup1c/users/vsundar/TERMinator/'

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

        train_batch_sampler = TERMLazyDataLoader(train_dataset, batch_size=6, shuffle=True)
        val_batch_sampler = TERMLazyDataLoader(val_dataset, batch_size=1, shuffle=False)
        test_batch_sampler = TERMLazyDataLoader(test_dataset, batch_size=1, shuffle=False)

        train_dataloader = DataLoader(train_dataset,
                                      batch_sampler = train_batch_sampler,
                                      num_workers = 2,
                                      collate_fn = train_batch_sampler._package,
                                      pin_memory=True)
        val_dataloader = DataLoader(val_dataset, 
                                      batch_sampler = val_batch_sampler,
                                      num_workers = 2,
                                      collate_fn = val_batch_sampler._package,
                                      pin_memory=True)
        test_dataloader = DataLoader(test_dataset, 
                                      batch_sampler = test_batch_sampler,
                                      num_workers = 2,
                                      collate_fn = test_batch_sampler._package)
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

        train_dataloader = TERMDataLoader(train_dataset, batch_size=6, shuffle=True)
        val_dataloader = TERMDataLoader(val_dataset, batch_size=1, shuffle=False)
        test_dataloader = TERMDataLoader(test_dataset, batch_size=1, shuffle=False)


    terminator = TERMinator(hidden_dim = 32, resnet_blocks = 4, term_layers = 4, conv_filter=3, device = dev)
    terminator.to(dev)

    optimizer = get_std_opt(terminator.parameters(), d_model=32)

    lam = 0.1
    max_grad_norm = 1

    save = []
    
    if os.path.isfile(os.path.join(run_output_dir, 'net_best_checkpoint.pt')):
        best_checkpoint_state = torch.load(os.path.join(run_output_dir, 'net_best_checkpoint.pt'))
        last_checkpoint_state = torch.load(os.path.join(run_output_dir, 'net_last_checkpoint.pt'))
        best_checkpoint = best_checkpoint_state['state_dict']
        best_validation = best_checkpoint_state['val_prob']
        start_epoch = last_checkpoint_state['epoch']
        terminator.load_state_dict(last_checkpoint_state['state_dict'])
        optimizer.load_state_dict(last_checkpoint_state['optimizer_state'])
        with open(os.path.join(run_output_dir, 'training_curves.pk'), 'rb') as fp:
            save = pickle.load(fp)
        writer = SummaryWriter(log_dir = os.path.join(run_output_dir, 'tensorboard'), purge_step = start_epoch+1)
    else:
        best_checkpoint = None
        best_validation = 999
        start_epoch = 0
        writer = SummaryWriter(log_dir = os.path.join(run_output_dir, 'tensorboard'))

    try:
        for epoch in range(start_epoch, 100):
            print('epoch', epoch)

            # train
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
                seq_lens = data['seq_lens']
                term_lens = data['term_lens']


                loss, rms, prob = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)

                torch.nn.utils.clip_grad_norm_(terminator.parameters(), max_grad_norm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_prob += prob.item()
                count = i

                avg_loss = running_loss / (count + 1)
                avg_prob = running_prob / (count + 1)
                train_progress.update(1)
                train_progress.refresh()
                train_progress.set_description_str('loss {} | prob {} '.format(avg_loss, avg_prob))

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
                    seq_lens = data['seq_lens']
                    term_lens = data['term_lens']


                    loss, rms, prob = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
                    running_loss += loss.item()
                    running_prob += prob.item()
                    count = i

                    p_recov = terminator.percent_recovery(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
                    recovery.append(p_recov)
                    
                    val_progress.update(1)

                val_progress.close()
                val_loss = running_loss / (count+1)
                val_prob = running_prob / (count+1)
                print('val loss', val_loss, '| approx val prob', val_prob)
                writer.add_scalar('val loss', val_loss, epoch)
                writer.add_scalar('approx val prob', val_prob, epoch)
                writer.add_scalar('val p recov', torch.stack(recovery).mean().item(), epoch)
                print('val p recov', torch.stack(recovery).mean().item())
                save.append([epoch_loss, val_loss])

            if val_prob < best_validation:
                best_validation = val_prob
                best_checkpoint = terminator.state_dict()
                checkpoint_state = {'epoch': epoch, 'state_dict': best_checkpoint, 'best_model': True, 'val_prob': best_validation, 'optimizer_state': optimizer.state_dict()}
                torch.save(checkpoint_state, os.path.join(run_output_dir, 'net_best_checkpoint.pt'))
                torch.save(checkpoint_state, os.path.join(run_output_dir, 'net_last_checkpoint.pt'))
            else:
                checkpoint_state = {'epoch': epoch, 'state_dict': terminator.state_dict(), 'best_model': False, 'val_prob': val_prob, 'optimizer_state': optimizer.state_dict()}
                torch.save(checkpoint_state, os.path.join(run_output_dir, 'net_last_checkpoint.pt'))

            with open(os.path.join(run_output_dir, 'training_curves.pk'), 'wb') as fp:
                pickle.dump(save, fp)

    except KeyboardInterrupt:
        pass

    print(save)

    torch.save(terminator.state_dict(), os.path.join(run_output_dir, 'net_last.pt'))
    torch.save(best_checkpoint, os.path.join(run_output_dir, 'net_best.pt'))
 

    with open(os.path.join(run_output_dir, 'training_curves.pk'), 'wb') as fp:
        pickle.dump(save, fp)

    dump = []
    recovery = []
    terminator.load_state_dict(best_checkpoint)

    terminator.eval()
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
            seq_lens = data['seq_lens']
            term_lens = data['term_lens']

            loss, rms, prob = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
            
            print(loss.item(), prob.item())

            output, E_idx = terminator.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)
            
            pred_seqs = terminator.pred_sequence(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)
            opt_seqs = terminator.opt_sequence(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
            print(ids)
            p_recov = terminator.percent_recovery(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
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
    parser.add_argument('--dataset', help = 'input folder .features files in proper directory structure. prefix is $INPUT_DATA/', default = "features_7000")
    parser.add_argument('--dev', help = 'device to train on', default = 'cuda:0')
    parser.add_argument('--lazy', help = 'use lazy data loading', default = True, type = bool)
    parser.add_argument('--train', help = 'file with training dataset', default = 'train.in')
    parser.add_argument('--validation', help = 'file with validation dataset', default = 'validation.in')
    parser.add_argument('--test', help = 'file with test dataset', default = 'test.in')
    # parser.add_argument('--shuffle_splits', help = 'shuffle dataset before creating train, validate, test splits', default = False, type=bool)
    parser.add_argument('--run_name', help = 'name for run, to use for output subfolder', default = 'test_run')
    args = parser.parse_args()
    main(args)
 
