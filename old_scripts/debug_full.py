from TERMinator import TERMinator
from data import *
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

def SparseMSE(output, label):
    diff = output - label
    residual = diff ** 2
    nnz = residual._nnz()
    sum = torch.sparse.sum(residual)
    error = sum / nnz
    return error

ifsdata = '/home/ifsdata/scratch/grigoryanlab/alexjli/'

torch.set_printoptions(threshold=5000)
dev = 'cuda:0'
dataset = Dataset(ifsdata + 'features')
dataloader = TERMDataLoader(dataset, batch_size=2, shuffle = False)
condense = TERMinator(device = 'cuda:0')
condense.to(dev)

#criterion = nn.MSELoss()
criterion = SparseMSE
# dummy optimizer, replace with adam or smth later
optimizer = optim.SGD(condense.parameters(), lr=0.003, momentum=0.9)

start = time.time()

for epoch in range(40):
    print('epoch', epoch)
    running_loss = 0
    count = 0
    for i, data in enumerate(dataloader):
        print('datapoint', i)
        #print(data)
        msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, etab, seqs, ids = data
        msas = msas.to(dev)
        features = features.to(dev).float()
        focuses = focuses.to(dev)
        src_key_mask = src_key_mask.to(dev)
        #selfEs = selfEs.to(dev).float()
        X = X.to(dev)
        x_mask = x_mask.to(dev)
        seqs = seqs.to(dev)
        print(seqs.shape)
        output = condense(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)

        optimizer.zero_grad()

        #loss = criterion(output, etab).float()
        loss = output
        print('loss', loss.item())
        print('prob', torch.exp(-loss).item())
        if loss.item() == float('inf') or loss.item() == float('nan'):
            exit()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count = i
    print('epoch loss', running_loss / (count+1))
    print('epoch prob', np.exp(-(running_loss / (count + 1))))
    print()

print(time.time() - start, 'seconds have elapsed')

condense.eval()
condense.to(dev)
with torch.no_grad():
    for i, data in enumerate(dataloader):
        print('datapoint', i)
        msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, etab, seqs, ids = data
        msas = msas.to(dev)
        features = features.to(dev).float()
        focuses = focuses.to(dev)
        src_key_mask = src_key_mask.to(dev)
        #selfEs = selfEs.to(dev).float()
        X = X.to(dev)
        x_mask = x_mask.to(dev)
        seqs = seqs.to(dev)
        loss = condense(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, seqs)
        print(loss)
        print(loss.item())

        output, E_idx = condense.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)

        print('output', output[0, 0, 0, ...])
        print('idx', E_idx[0,0, ...])
        print('label', etab.to_dense()[0, 0, 0, ...])
        print()


