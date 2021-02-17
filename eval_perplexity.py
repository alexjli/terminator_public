from TERMinator import *
from data import *
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import os
import sys
import copy
import json

INPUT_DATA = '/home/gridsan/alexjli/keatinglab_shared/alexjli/TERMinator/'
OUTPUT_DIR = '/home/gridsan/alexjli/keatinglab_shared/alexjli/TERMinator_runs/'

''' Negative log psuedo-likelihood '''
''' Averaged nlpl per residue, across batches '''
def nlpl(etab, E_idx, ref_seqs, x_mask):
    etab_device = etab.device
    n_batch, L, k, _ = etab.shape
    etab = etab.unsqueeze(-1).view(n_batch, L, k, 20, 20)

    # X is encoded as 20 so lets just add an extra row/col of zeros
    pad = (0, 1, 0, 1)
    etab = F.pad(etab, pad, "constant", 0)
    isnt_x_aa = (ref_seqs != 20).float().to(etab_device)

    # separate selfE and pairE since we have to treat selfE differently
    self_etab = etab[:, :, 0:1]
    pair_etab = etab[:, :, 1:]
    # idx matrix to gather the identity at all other residues given a residue of focus
    E_aa = torch.gather(ref_seqs.unsqueeze(-1).expand(-1, -1, k-1), 1, E_idx[:, :, 1:])
    E_aa = E_aa.view(list(E_idx[:,:,1:].shape) + [1,1]).expand(-1, -1, -1, 21, -1)
    # gather the 22 energies for each edge based on E_aa
    pair_nrgs = torch.gather(pair_etab, 4, E_aa).squeeze(-1)
    # gather 22 self energies by taking the diagonal of the etab
    self_nrgs = torch.diagonal(self_etab, offset=0, dim1=-2, dim2=-1)
    # concat the two to get a full edge etab
    edge_nrgs = torch.cat((self_nrgs, pair_nrgs), dim=2)
    # get the avg nrg for 22 possible aa identities at each position
    aa_nrgs = torch.sum(edge_nrgs, dim = 2)

    #prior_loss = ((aa_nrgs - self.prior) ** 2).mean(dim = 1).sum()

    #aa_nrgs -= self.prior
    #self.update_prior(aa_nrgs)
    #aa_nrgs = self.ln(aa_nrgs)


    # convert energies to probabilities
    log_all_aa_probs = torch.log_softmax(-aa_nrgs, dim = 2)
    # get the probability of the sequence
    log_probs = torch.gather(log_all_aa_probs, 2, ref_seqs.unsqueeze(-1)).squeeze(-1)

    # convert to nlpl
    log_probs *= x_mask # zero out positions that don't have residues
    log_probs *= isnt_x_aa # zero out positions where the native sequence is X
    n_res = torch.sum(x_mask * isnt_x_aa, dim=-1)
    nlpl = -torch.sum(log_probs, dim=-1)#/n_res
    return nlpl, n_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval TERMinator Psuedoperplexity')
    parser.add_argument('--dataset', help = 'input folder .features files in proper directory structure. prefix is $INPUT_DATA/', default = "features_singlechain")
    parser.add_argument('--dev', help = 'device to train on', default = 'cuda:0')
    parser.add_argument('--test', help = 'file with test dataset', default = 'test.in')
    # parser.add_argument('--shuffle_splits', help = 'shuffle dataset before creating train, validate, test splits', default = False, type=bool)
    parser.add_argument('--run_name', help = 'name for run, to use for output subfolder', default = 'test_run')
    args = parser.parse_args()

    run_output_dir = os.path.join(OUTPUT_DIR, args.run_name)
    dev = args.dev

    test_ids = []
    with open(os.path.join(INPUT_DATA, args.dataset, args.test), 'r') as f:
        for line in f:
            test_ids += [line[:-1]]
    test_dataset = LazyDataset(os.path.join(INPUT_DATA, args.dataset), pdb_ids = test_ids)
    test_batch_sampler = TERMLazyDataLoader(test_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_sampler = test_batch_sampler,
                                 collate_fn = test_batch_sampler._package)

    with open(os.path.join(run_output_dir, "hparams.json")) as fp:
        hparams = json.load(fp)

    # backwards compatability
    if "cov_features" not in hparams.keys():
        hparams["cov_features"] = False
    if "term_use_mpnn" not in hparams.keys():
        hparams["term_use_mpnn"] = False
    if "matches" not in hparams.keys():
        hparams["matches"] = "resnet"
    if "struct2seq_linear" not in hparams.keys():
        hparams['struct2seq_linear'] = False
    
    #terminator = TERMinator(hparams = hparams, device = dev)
    terminator = MultiChainTERMinator_g(hparams = hparams, device = dev)
    

    best_checkpoint_state = torch.load(os.path.join(run_output_dir, 'net_best_checkpoint.pt'))
    best_checkpoint = best_checkpoint_state['state_dict']
    terminator.load_state_dict(best_checkpoint)
    terminator.to(dev)

    test_sum = 0
    test_weights = 0

    terminator.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader)):
            msas = data['msas'].to(dev)
            features = data['features'].to(dev).float()
            focuses = data['focuses'].to(dev)
            src_key_mask = data['src_key_mask'].to(dev)
            X = data['X'].to(dev)
            x_mask = data['x_mask'].to(dev)
            seqs = data['seqs'].to(dev)
            seq_lens = data['seq_lens'].to(dev)
            ids = data['ids']
            term_lens = data['term_lens'].to(dev)
            max_seq_len = max(seq_lens.tolist())
            chain_lens = data['chain_lens']
            ppoe = data['ppoe'].to(dev).float()

            etab, E_idx = terminator.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe, chain_lens)
            #etab, E_idx = terminator.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe)
            
            loss, n_res = nlpl(etab, E_idx, seqs, x_mask)
            
            # Accumulate
            test_sum += torch.sum(loss).cpu().data.numpy()
            test_weights += n_res.cpu().data.numpy()

    test_loss = test_sum / test_weights
    test_perplexity = np.exp(test_loss)
    print('Perplexity\tTest:{}'.format(test_perplexity))

    with open(os.path.join(run_output_dir,"perplexity.log"), 'w') as fp:
        fp.write(f"Perplexity on Test set:\n{test_perplexity[0]}\n")







    
