from TERMinator import TERMinator, TERMinator2
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


dataset = LazyDataset(ifsdata + 'features_2000')
#dataset.shuffle()
#dataset = dataset[:100]

# idxs at which to split the dataset
train_val, val_test = (len(dataset) * 8) // 10, (len(dataset) * 9) // 10
train_dataset = dataset[:train_val]
val_dataset = dataset[train_val:val_test]
test_dataset = dataset[val_test:]

train_batch_sampler = TERMLazyDataLoader(train_dataset, batch_size=12, shuffle=True)
val_batch_sampler = TERMLazyDataLoader(val_dataset, batch_size=1, shuffle=False)
test_batch_sampler = TERMLazyDataLoader(test_dataset, batch_size=1, shuffle=False)

train_dataloader = DataLoader(train_dataset,
                              batch_sampler = train_batch_sampler,
                              num_workers = 4,
                              collate_fn = train_batch_sampler._package)
val_dataloader = DataLoader(val_dataset, 
                              batch_sampler = val_batch_sampler,
                              num_workers = 4,
                              collate_fn = val_batch_sampler._package)
test_dataloader = DataLoader(test_dataset, 
                              batch_sampler = test_batch_sampler,
                              num_workers = 4,
                              collate_fn = test_batch_sampler._package)
terminator = TERMinator(hidden_dim = 32, resnet_blocks = 1, term_layers = 3, device = dev)
terminator.to(dev)

optimizer = optim.SGD(terminator.parameters(), lr=0.001, momentum=0.7)
optimizer = optim.Adagrad(terminator.parameters(), lr=0.001, lr_decay=0.01)
optimizer = optim.Adam(terminator.parameters(), lr = 5e-4, weight_decay=0)
optimizer = get_std_opt(terminator.parameters(), d_model=32)

lam = 0.1
max_grad_norm = 1

save = []

try:
    for epoch in range(1000):
        print('epoch', epoch)

        # train
        running_loss = 0
        count = 0
        terminator.train()
        
        train_progress = tqdm(total=len(train_dataloader))
        for i, data in enumerate(train_dataloader):
            #print('datapoint', i)
            msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, seqs, ids = data
            print(ids)
            msas = msas.to(dev)
            features = features.to(dev).float()
            #print(features.shape)
            focuses = focuses.to(dev)
            src_key_mask = src_key_mask.to(dev)
            X = X.to(dev)
            x_mask = x_mask.to(dev)
            seqs = seqs.to(dev)

            loss = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
            #torch.nn.utils.clip_grad_norm_(terminator.parameters(), max_grad_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count = i

            avg_loss = running_loss / (count + 1)
            train_progress.update(1)
            train_progress.refresh()
            train_progress.set_description_str('loss {} | prob {} '.format(avg_loss, np.exp(-avg_loss)))

        train_progress.close()
        epoch_loss = running_loss / (count+1)
        print('epoch loss', epoch_loss, '| approx epoch prob', np.exp(-epoch_loss))
        writer.add_scalar('training loss', epoch_loss, epoch)
        writer.add_scalar('approx training prob', np.exp(-epoch_loss), epoch)

        # validate
        with torch.no_grad():
            running_loss = 0
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

                loss = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
                running_loss += loss.item()
                count = i

                p_recov = terminator.percent_recovery(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
                recovery.append(p_recov)
                
                val_progress.update(1)

            val_progress.close()
            val_loss = running_loss / (count+1)
            print('val loss', val_loss, '| approx val prob', np.exp(-val_loss))
            writer.add_scalar('val loss', val_loss, epoch)
            writer.add_scalar('approx val prob', np.exp(-val_loss), epoch)
            writer.add_scalar('val p recov', torch.stack(recovery).mean().item(), epoch)
            print('val p recov', torch.stack(recovery).mean().item())
            save.append([epoch_loss, val_loss])

        for name, param in terminator.named_parameters():
            if 'bn' not in name:
                try:
                    writer.add_histogram(name, param.grad, epoch)
                    """
                    if name == 'bot.embedding.embedding.weight':
                        writer.add_embedding(param, metadata=['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x'], global_step=epoch)
                    """
                except:
                    writer.add_histogram(name, torch.zeros(1), epoch)
        #writer.flush()

except KeyboardInterrupt:
    pass

print(save)

dump = []

recovery = []
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
        loss = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
        
        print(loss.item(), torch.exp(-loss).item())

        output, E_idx = terminator.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)
        
        pred_seqs = terminator.pred_sequence(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)
        opt_seqs = terminator.opt_sequence(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
        print(ids)
        p_recov = terminator.percent_recovery(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
        print('pred', pred_seqs)
        print('opt', opt_seqs)
        print('p recov', p_recov)
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

torch.save(terminator.state_dict(), ifsdata + 'net.pt')
writer.close()