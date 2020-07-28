from TERMinator import TERMinator
from data import *
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm

ifsdata = '/home/ifsdata/scratch/grigoryanlab/alexjli/'

torch.set_printoptions(threshold=5000)
torch.set_printoptions(linewidth=500)
torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)
dev = 'cuda:0'
#dev = 'cpu'
dataset = Dataset(ifsdata + 'tmp')
dataset.shuffle()

# idxs at which to split the dataset
train_val, val_test = (len(dataset) * 8) // 10, (len(dataset) * 9) // 10
train_dataset = dataset[:train_val]
val_dataset = dataset[train_val:val_test]
test_dataset = dataset[val_test:]

train_dataloader = TERMDataLoader(train_dataset, batch_size=4, shuffle = True)
val_dataloader = TERMDataLoader(val_dataset, batch_size=1, shuffle = False)
test_dataloader = TERMDataLoader(test_dataset, batch_size=1, shuffle = False)
terminator = TERMinator(device = dev)
terminator.to(dev)

optimizer = optim.SGD(terminator.parameters(), lr=0.001, momentum=0.7)
optimizer = optim.Adagrad(terminator.parameters(), lr=0.001, lr_decay=0.01)

save = []

for epoch in range(1500):
    print('epoch', epoch)

    # train
    running_loss = 0
    count = 0
    terminator.train()
    
    train_progress = tqdm(total=len(train_dataloader))
    for i, data in enumerate(train_dataloader):
        #print('datapoint', i)
        msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, etab, seqs, ids = data
        print(ids)
        msas = msas.to(dev)
        features = features.to(dev).float()
        focuses = focuses.to(dev)
        src_key_mask = src_key_mask.to(dev)
        X = X.to(dev)
        x_mask = x_mask.to(dev)
        seqs = seqs.to(dev)

        loss = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)

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
    print('epoch loss', epoch_loss, '| epoch prob', np.exp(-epoch_loss))

    # validate
    with torch.no_grad():
        running_loss = 0
        count = 0
        terminator.eval()

        val_progress = tqdm(total=len(val_dataloader))
        for i, data in enumerate(val_dataloader):
            msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, etab, seqs, ids = data
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
            
            val_progress.update(1)

        val_progress.close()
        val_loss = running_loss / (count+1)
        print('val loss', val_loss, '| val prob', np.exp(-val_loss))
        save.append([epoch_loss, val_loss])

print(save)

dump = []

terminator.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print('datapoint', i)
        msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, etab, seqs, ids = data
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
        
        #seqs = terminator.predict_sequence(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)
        print(ids)
        #print(seqs)
        

        n_batch, l, n = output.shape[:3]
        """
        print('out', output.view(n_batch, l, n, 22, 22)[0, 0, 0, ...])
        print('idx', E_idx[0,0, ...])
        print('label', etab.to_dense().view(n_batch, l, l, 22, 22)[0, 0, 0, ...])
        print()
        """
        dump.append({'out': output.view(n_batch, l, n, 20, 20).cpu().numpy(),
                     'idx': E_idx.cpu().numpy(),
                     'label': etab.to_dense().view(n_batch, l, l, 22, 22).numpy()
                    })

import pickle
with open(ifsdata + 'net.out', 'wb') as fp:
    pickle.dump(dump, fp)

with open(ifsdata+'training_curves.pk', 'wb') as fp:
    pickle.dump(save, fp)

torch.save(terminator.state_dict(), ifsdata + 'net.pt')
