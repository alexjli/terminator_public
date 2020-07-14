from TERMinator import TERMinator
from data import *
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

torch.set_printoptions(threshold=5000)
dev = 'cuda:0'
dev = 'cpu'
dataset = Dataset('../MST_workspace/features')
dataset.shuffle()

# idxs at which to split the dataset
train_val, val_test = (len(dataset) * 8) // 10, (len(dataset) * 9) // 10
train_dataset = dataset[:train_val]
val_dataset = dataset[train_val:val_test]
test_dataset = dataset[val_test:]

train_dataloader = TERMDataLoader(train_dataset, batch_size=2, shuffle = False)
val_dataloader = TERMDataLoader(val_dataset, batch_size=2, shuffle = False)
test_dataloader = TERMDataLoader(test_dataset, batch_size=2, shuffle = False)
terminator = TERMinator()
terminator.to(dev)

optimizer = optim.SGD(terminator.parameters(), lr=0.003, momentum=0.9)

for epoch in range(1):
    print('epoch', epoch)

    # train
    running_loss = 0
    count = 0
    terminator.train()
    for i, data in enumerate(train_dataloader):
        #print('datapoint', i)
        msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, etab, seqs = data
        msas = msas.to(dev)
        features = features.to(dev).float()
        focuses = focuses.to(dev)
        src_key_mask = src_key_mask.to(dev)
        X = X.to(dev)
        x_mask = x_mask.to(dev)

        loss = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count = i
    epoch_loss = running_loss / (count+1)
    print('epoch loss', epoch_loss, '| epoch prob', np.exp(-epoch_loss))

    # validate
    with torch.no_grad():
        terminator.eval()
        running_loss = 0
        count = 0
        terminator.eval()
        for i, data in enumerate(val_dataloader):
            msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, etab, seqs = data
            msas = msas.to(dev)
            features = features.to(dev).float()
            focuses = focuses.to(dev)
            src_key_mask = src_key_mask.to(dev)
            X = X.to(dev)
            x_mask = x_mask.to(dev)

            loss = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
            running_loss += loss.item()
            count = i

        val_loss = running_loss / (count+1)
        print('val loss', val_loss, '| val prob', np.exp(-val_loss))

"""
terminator.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print('datapoint', i)
        msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, etab, seqs = data
        msas = msas.to(dev)
        features = features.to(dev).float()
        focuses = focuses.to(dev)
        src_key_mask = src_key_mask.to(dev)
        loss = terminator(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
        print(loss)
        print(loss.item())

        output, E_idx = terminator.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)

        print('output', output[0, 0, 0, ...])
        print('idx', E_idx[0,0, ...])
        print('label', etab[0, 0, 0, ...])
        print()
"""
