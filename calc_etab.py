import torch
from TERMinator import TERMinator
from data import TERMDataLoader
from utils.packageTensors import dumpTrainingTensors
from to_etab import *

dev = "cuda:0"
state_dict = '/home/ifsdata/scratch/grigoryanlab/alexjli/run_7000_cpl_term1_tanh/net_last.pt'
in_path = "/home/ironfs/scratch/grigoryanlab/alexjli/dTERMen_speedtest200_clique1/1BUO/1BUO"
out_path = "/home/ironfs/scratch/grigoryanlab/alexjli/etabs_7000_cpl_term1_tanh/1BUO.etab"
datapoint = dumpTrainingTensors(in_path, cutoff=50, save=False)
dataloader = TERMDataLoader([datapoint])

data = next(iter(dataloader))
msas = data['msas'].to(dev)
features = data['features'].to(dev).float()
focuses = data['focuses']
focuses.to(dev)
src_key_mask = data['src_key_mask'].to(dev)
X = data['X'].to(dev)
x_mask = data['x_mask'].to(dev)
seqs = data['seqs'].to(dev)
seq_lens = data['seq_lens']
term_lens = data['term_lens']

terminator = TERMinator(hidden_dim = 32, resnet_blocks = 4, term_layers = 4, conv_filter=3, device = dev
)
terminator.load_state_dict(torch.load(state_dict))
terminator.to(dev)
terminator.eval()
etab, E_idx = terminator.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask)

n_batch, l, n = etab.shape[:3]
etab = etab.view(1, l, n, 20, 20).squeeze(0).detach().cpu().numpy()
E_idx = E_idx.squeeze(0).detach().cpu().numpy()
print(etab.shape)
idx_dict = get_idx_dict(in_path+".red.pdb")
to_etab_file(etab, E_idx, idx_dict, out_path)

