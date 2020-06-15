from nets import *
from data import *
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

dev = 'cuda:0'
dataset = Dataset('../MST_workspace/features')
dataloader = TERMDataLoader(dataset, batch_size=1)
condense = CondenseMSA(hidden_dim = 32, num_blocks=2, device = dev).double()
reshape = DummyLSTM().to(dev).double()
#print(condense)
condense.to(dev)

criterion = nn.MSELoss()
# dummy optimizer, replace with atom later
optimizer = optim.SGD(condense.parameters(), lr=0.001, momentum=0.9)

for i, data in enumerate(dataloader):
    print('datapoint', i)
    msas, features, seq_lens, focus, src_mask, src_key_mask, selfEs = data
    msas = msas.to(dev)
    features = features.to(dev)
    focus = focus.to(dev)
    src_mask = src_mask.to(dev)
    src_key_mask = src_key_mask.to(dev)
    selfEs = selfEs.to(dev)
    output = condense(msas, features, seq_lens, focus, src_mask, src_key_mask)
    output = reshape(output)

    optimizer.zero_grad()

    loss = criterion(output, selfEs)
    print(loss.item())
    loss.backward()
    optimizer.step()
