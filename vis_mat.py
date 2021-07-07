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
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_DATA = '/home/gridsan/alexjli/keatinglab_shared/alexjli/TERMinator/'
OUTPUT_DIR = '/home/gridsan/alexjli/keatinglab_shared/alexjli/TERMinator_runs/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval TERMinator Psuedoperplexity')
    parser.add_argument('--dev', help = 'device to train on', default = 'cuda:0')
    # parser.add_argument('--shuffle_splits', help = 'shuffle dataset before creating train, validate, test splits', default = False, type=bool)
    parser.add_argument('--run_name', help = 'name for run, to use for output subfolder', default = 'test_run')
    args = parser.parse_args()

    run_output_dir = os.path.join(OUTPUT_DIR, args.run_name)
    dev = args.dev
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
    if "energies_gvp" not in hparams.keys():
        hparams['energies_gvp'] = False
    if "num_stats" not in hparams.keys():
        hparams['num_stats'] = 0
    
    best_checkpoint_state = torch.load(os.path.join(run_output_dir, 'net_best_checkpoint.pt'), map_location=torch.device('cpu'))
    best_checkpoint = best_checkpoint_state['state_dict']
    W_sing = best_checkpoint['bot.W_ppoe.weight'][:, :5].cpu().detach().numpy()
    sns.heatmap(W_sing, cmap="YlGnBu", center=0)
    plt.show()


