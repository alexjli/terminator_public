from nets import *
from data import *
import pickle
import torch
import numpy as np
import time

dataloader = TERMDataLoader('temp.features', batch_size=2)
condense = CondenseMSA(hidden_dim = 32, num_blocks=2)
#print(condense)
condense.cuda()

for data in dataloader
    msas, features, seq_lens, focus, src_mask, src_key_mask = data
    msas = msas.cuda().long()
    features = features.cuda().float()
    focus = focus.cuda()
    src_mask = src_mask.cuda()
    src_key_mask = src_key_mask.cuda()
    output = condense(msas, features, seq_lens, focus, src_mask, src_key_mask)
