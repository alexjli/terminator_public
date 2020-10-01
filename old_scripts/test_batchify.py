import torch

from batched_term_transformer.term_attn import *
from data import *

dataset = Dataset('../MST_workspace/features')
dataloader = TERMDataLoader(dataset, batch_size=2)
iterable = iter(dataloader)
data = next(iterable)

msas, features, seq_lens, focuses, padded_src_masks, src_key_mask, selfEs, term_lens = data
print([set(focus) for focus in focuses.tolist()])
term_total_len_1 = sum(term_lens[0])
term_total_len_2 = sum(term_lens[1])
if term_total_len_1 > term_total_len_2:
    max_term_len = max(term_lens[1])
    diff = term_total_len_1 - term_total_len_2
    term_lens[1] += [max_term_len] * (diff // max_term_len)
    term_lens[1].append(diff % max_term_len)
else:
    max_term_len = max(term_lens[0])
    diff = term_total_len_2 - term_total_len_1
    term_lens[0] += [max_term_len] * (diff // max_term_len)
    term_lens[0].append(diff % max_term_len)

features = features.sum(dim = 1)
batchify = BatchifyTERM().cuda()
transformer = TERMTransformer(7, 7, 1).double().cuda()
features = features.cuda()
import time
start = time.time()

b = batchify(features, term_lens)
f = batchify(focuses, term_lens)
#print(f.shape)
max_seq_len = max(seq_lens)
n_batches = 2

b = transformer(b)
layer = torch.arange(n_batches).unsqueeze(-1).unsqueeze(-1).expand(f.shape).long().cuda()
aggregate = torch.zeros(n_batches, max_seq_len, 7).double().cuda()
aggregate = aggregate.index_put((layer,f), b, accumulate=True)

print(time.time() - start)

print(aggregate)
print(aggregate.shape)
print(b.shape)

#torch.Size([1, 50, 862, 7])
