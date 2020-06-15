from nets import *
from data import *
import pickle
import torch
import numpy as np
import time

def convert(tensor):
    return torch.from_numpy(tensor)

"""
with open('data/2JOF_full.features', 'rb') as fp:
    data = pickle.load(fp)
with open('data/2JOF.features', 'rb') as fp:
    data2 = pickle.load(fp)
with open('data/5FG0.features', 'rb') as fp:
    data3 = pickle.load(fp)
with open('data/2FGO.features', 'rb') as fp:
    data4 = pickle.load(fp)
with open('temp.features', 'wb') as fp:
    pickle.dump([data, data2, data3, data4], fp)
"""



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
