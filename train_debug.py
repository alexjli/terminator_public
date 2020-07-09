from nets import *
from data import *
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

dev = 'cuda:0'
dev = 'cpu'
dataset = Dataset('../MST_workspace/features')
dataloader = TERMDataLoader(dataset, batch_size=2)
condense = Wrapper(hidden_dim = 32, num_blocks = 2, device = dev)
condense.to(dev)

criterion = nn.MSELoss()
# dummy optimizer, replace with atom later
optimizer = optim.SGD(condense.parameters(), lr=0.003, momentum=0.9)

for epoch in range(40):
    print('epoch', epoch)
    running_loss = 0
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

        optimizer.zero_grad()

        loss = criterion(output, selfEs)
        print(loss.item())
        loss.backward()
        optimizer.step()

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
