from nets import *
from data import *
import pickle
import torch
import numpy as np
import time

def convert(tensor):
    return torch.from_numpy(tensor)

dataloader = TERMDataLoader('temp.features', batch_size=2)
condense = CondenseMSA(hidden_dim = 8, num_blocks=2)
#print(condense)
#condense.cuda()

for data in dataloader:
    start = time.time()
    msas, features, seq_lens, focus, src_mask, src_key_mask = data
    print(src_key_mask.shape)
    # msas = msas.cuda().long()
    # features = features.cuda().float()
    # focus = focus.cuda()
    # src_mask = src_mask.cuda()
    # src_key_mask = src_key_mask.cuda()
    msas = msas.long()
    features = features.float()
    output = condense(msas, features, seq_lens, focus, src_mask, src_key_mask)
    print(output.shape)
    end = time.time()
    print(end - start)
    del msas, features, seq_lens, focus, src_mask, src_key_mask
    torch.cuda.empty_cache()
