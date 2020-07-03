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

dev = 'cuda:0'
dev = 'cpu'
dataset = Dataset('../MST_workspace/features')
dataloader = TERMDataLoader(dataset, batch_size=2)
condense = TERMinator()
condense.to(dev)

#criterion = nn.MSELoss()
criterion = SparseMSE
# dummy optimizer, replace with adam or smth later
optimizer = optim.SGD(condense.parameters(), lr=0.003, momentum=0.9)

for epoch in range(40):
    print('epoch', epoch)
    running_loss = 0
    for i, data in enumerate(dataloader):
        print('datapoint', i)
        msas, features, seq_lens, focuses, src_key_mask, selfEs, term_lens, X, x_mask, etab = data
        msas = msas.to(dev)
        features = features.to(dev).float()
        focuses = focuses.to(dev)
        src_key_mask = src_key_mask.to(dev)
        selfEs = selfEs.to(dev).float()
        X = X.to(dev)
        x_mask = x_mask.to(dev)
        output = condense(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)

        optimizer.zero_grad()

        loss = criterion(output, etab).float()
        print(loss)
        print(loss.item())
        loss.backward()
        optimizer.step()
        exit()

        running_loss += loss.item()
    print('epoch loss', running_loss / (i+1))
    print()

condense.eval()
with torch.no_grad():
    for i, data in enumerate(dataloader):
        print('datapoint', i)
        msas, features, seq_lens, focuses, src_mask, src_key_mask, selfEs, term_lens = data
        msas = msas.to(dev)
        features = features.to(dev).float()
        focuses = focuses.to(dev)
        src_mask = src_mask.to(dev)
        src_key_mask = src_key_mask.to(dev)
        selfEs = selfEs.to(dev).float()
        output = condense(msas, features, seq_lens, focuses, term_lens, src_mask, src_key_mask)

        print('output', output[:, 0, ...])
        print('label', selfEs[:, 0, ...])
        print()
