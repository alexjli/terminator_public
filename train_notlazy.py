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


ifsdata = '/home/ifsdata/scratch/grigoryanlab/alexjli/'

torch.set_printoptions(threshold=10000)
torch.set_printoptions(linewidth=1000)
torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(profile="full")
dev = 'cuda:0'
#dev = 'cpu'
writer = SummaryWriter(log_dir = ifsdata + 'runs')

dataset = TERMDataset(ifsdata + 'features_600')
#dataset.shuffle()

# idxs at which to split the dataset
train_val, val_test = (len(dataset) * 8) // 10, (len(dataset) * 9) // 10
train_dataset = dataset[:train_val]
val_dataset = dataset[train_val:val_test]
test_dataset = dataset[val_test:]

train_dataloader = TERMDataLoader(train_dataset, batch_size=12, shuffle=True)
val_dataloader = TERMDataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataloader = TERMDataLoader(test_dataset, batch_size=1, shuffle=False)

terminator = TERMinator(hidden_dim = 32, resnet_blocks = 3, term_layers = 3, conv_filter=3, device = dev)
terminator.to(dev)

optimizer = get_std_opt(terminator.parameters(), d_model=32)

lam = 0.1
max_grad_norm = 1

save = []

best_checkpoint = None
best_validation = 999

try:
    for epoch in range(1000):
        print('epoch', epoch)

        # train
        running_loss = 0
        running_prob = 0
        count = 0
        terminator.train()
        
        train_progress = tqdm(total=len(train_dataloader))
        for i, data in enumerate(train_dataloader):
            #print('datapoint', i)
            msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, seqs, ids = data
            #print(ids)
            msas = msas.to(dev)
            features = features.to(dev).float()
            #print(features.shape)
            focuses = focuses.to(dev)
            src_key_mask = src_key_mask.to(dev)
            X = X.to(dev)
            x_mask = x_mask.to(dev)
            seqs = seqs.to(dev)

            loss, rms, prob = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
            #loss = loss + rms

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
                msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, seqs, ids = data
                msas = msas.to(dev)
                features = features.to(dev).float()
                focuses = focuses.to(dev)
                src_key_mask = src_key_mask.to(dev)
                X = X.to(dev)
                x_mask = x_mask.to(dev)
                seqs = seqs.to(dev)

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

        if epoch % 6 == 0 and val_prob < best_validation:
            best_validation = val_prob
            best_checkpoint = terminator.state_dict()

        """
        for name, param in terminator.named_parameters():
            if 'bn' not in name:
                try:
                    writer.add_histogram(name, param.grad, epoch)
                    if name == 'bot.embedding.embedding.weight':
                        writer.add_embedding(param, metadata=['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x'], global_step=epoch)
                except:
                    writer.add_histogram(name, torch.zeros(1), epoch)
        #writer.flush()
        """

except KeyboardInterrupt:
    pass

print(save)

dump = []

recovery = []
terminator.load_state_dict(best_checkpoint)
terminator.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print('datapoint', i)
        msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, seqs, ids = data
        msas = msas.to(dev)
        features = features.to(dev).float()
        focuses = focuses.to(dev)
        src_key_mask = src_key_mask.to(dev)
        X = X.to(dev)
        x_mask = x_mask.to(dev)
        seqs = seqs.to(dev)
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

import pickle
with open(ifsdata + 'net.out', 'wb') as fp:
    pickle.dump(dump, fp)

with open(ifsdata+'training_curves.pk', 'wb') as fp:
    pickle.dump(save, fp)

torch.save(terminator.state_dict(), ifsdata + 'net_last.pt')
torch.save(best_checkpoint, ifsdata + 'net_best.pt')
writer.close()
