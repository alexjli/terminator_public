from nets import *
import pickle
import torch
import numpy as np
import time
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

def convert(tensor):
    return torch.from_numpy(tensor)

with open('preprocessing/2JOF_full.features', 'rb') as fp:
    data1 = pickle.load(fp)

with open('preprocessing/2JOF.features', 'rb') as fp:
    data2 = pickle.load(fp)

features1 = convert(data1['features']).float().cuda()
features2 = convert(data2['features']).float().cuda()
features = pad_sequence([features1.squeeze(0).transpose(0,1), features2.squeeze(0).transpose(0,1)], batch_first=True)
features = features.transpose(1,2)
print(features.shape)

msas1 = convert(data1['msas']).cuda()
msas2 = convert(data2['msas']).cuda()
msas = pad_sequence([msas1.squeeze(0).transpose(0,1), msas2.squeeze(0).transpose(0,1)], batch_first=True)
msas = msas.transpose(1,2)
print(msas.shape)

len1 = features1.shape[2]
len2 = features2.shape[2]

#term_lens = data['term_lens'].tolist()

seq_len1 = data1['seq_len']
seq_len2 = data2['seq_len']
seq_lens = [seq_len1, seq_len2]

feat_len1 = features1.shape[2]
feat_len2 = features2.shape[2]
feat_lens = [feat_len1, feat_len2]
src_key_mask = pad_sequence([torch.zeros(l) for l in feat_lens], batch_first=True, padding_value=1).bool().cuda()
print(src_key_mask.shape)

focus1 = convert(data1['focuses'])
focus2 = convert(data2['focuses'])
focus = pad_sequence([focus1, focus2], batch_first=True)
print(focus.shape)

from scipy.linalg import block_diag
src_mask1 = convert(data1['mask'])
diff = features1.shape[2]-features2.shape[2]
src_mask2 = convert(block_diag(data2['mask'], np.zeros((diff,diff))))
src_mask = torch.stack([src_mask1, src_mask2])
# convert to boolean
# invert because 1 = mask, 0 = no mask
src_mask = ~src_mask.bool().cuda()
print(src_mask.shape)

condense = CondenseMSA()
condense.cuda()

start = time.time()
output = condense(msas, features, seq_lens, focus, src_mask, src_key_mask)
