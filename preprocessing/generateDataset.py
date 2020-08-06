import numpy as np
import os
import json
import pickle
import argparse
from scipy.linalg import block_diag
import glob
import multiprocessing as mp

from parseTERM import parseTERMdata
from parseEtab import parseEtab
from parseCoords import parseCoords
from mmtf_util import *
from util import *

cath_base_url = 'http://download.cathdb.info/cath/releases/latest-release/'

def dumpTrainingTensors(in_path, out_path = None, cutoff = 1000, save=True):
    coords = parseCoords(in_path + '.red.pdb', save=False)
    data = parseTERMdata(in_path + '.dat')
    etab, self_etab = parseEtab(in_path + '.etab', save=False)

    selection = data['selection']

    term_msas = []
    term_features = []
    term_focuses = []
    term_lens = []
    for term_data in data['terms']:
        focus = term_data['focus']
        # only take data for residues that are in the selection
        take = [i for i in range(len(focus)) if focus[i] in selection]

        # cutoff MSAs at top N
        msa = term_data['labels'][:cutoff]
        # apply take
        term_msas.append(np.take(msa, take, axis=-1))

        # add focus
        focus_take = [item for item in focus if item in selection]
        term_focuses += focus_take
        # append term len, the len of the focus
        term_lens.append(len(focus_take))

        # cutoff ppoe at top N
        ppoe = term_data['ppoe']
        term_len = ppoe.shape[2]
        num_alignments = ppoe.shape[0]
        ppoe = ppoe[:cutoff]

        ppo_rads = ppoe[:, :3]/180*np.pi
        sin_ppo = np.sin(ppo_rads)
        cos_ppo = np.cos(ppo_rads)
        env = ppoe[:, 3:]

        # apply take
        ppoe = np.take(ppoe, take, axis=-1)

        # cutoff rmsd at top N
        rmsd = np.expand_dims(term_data['rmsds'][:cutoff], 1)
        rmsd_arr = np.concatenate([rmsd for _ in range(term_len)], axis=1)
        rmsd_arr = np.expand_dims(rmsd_arr, 1)
        term_len_arr = np.zeros((cutoff, 1, term_len))
        term_len_arr += term_len
        num_alignments_arr = np.zeros((cutoff, 1, term_len))
        num_alignments_arr += num_alignments
        #features = np.concatenate([ppoe, rmsd_arr, term_len_arr, num_alignments_arr], axis=1)
        """
        for arr in [sin_ppo, cos_ppo, env, rmsd_arr, term_len_arr, num_alignments_arr]:
            print(arr.shape)
        """
        features = np.concatenate([sin_ppo, cos_ppo, env, rmsd_arr, term_len_arr, num_alignments_arr], axis=1)

        # pytorch does row vector computation
        # swap rows and columns
        features = features.transpose(0, 2, 1)
        term_features.append(features)

    msa_tensor = np.concatenate(term_msas, axis = -1)

    features_tensor = np.concatenate(term_features, axis = 1)

    len_tensor = np.array(term_lens)
    term_focuses = np.array(term_focuses)

    # check that sum of term lens is as long as the feature tensor
    assert sum(len_tensor) == features_tensor.shape[1]

    pdb = in_path.split('/')[-1]

    coords_tensor = None
    if len(coords) == 1:
        chain = next(iter(coords.keys()))
        coords_tensor = coords[chain]

    output = {
        'pdb': pdb,
        'coords': coords_tensor,
        'features': features_tensor,
        'msas': msa_tensor,
        'focuses': term_focuses,
        'term_lens': len_tensor,
        'sequence': np.array(data['sequence']),
        'seq_len': len(data['selection']),
        'etab': etab,
        'selfE': self_etab
    }


    if save:
        if not out_path:
            out_path = ''

        with open(out_path + '.features', 'wb') as fp:
            pickle.dump(output, fp)

    print('Done with', pdb)

    return output

def generateDataset(in_folder, out_folder, cutoff = 1000):
    # make folder where the dataset files are gonna be placed
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # generate absolute paths so i dont have to think about relative references
    in_folder = os.path.abspath(in_folder)
    out_folder = os.path.abspath(out_folder)

    dataset = []
    os.chdir(in_folder)

    # process folder by folder
    for folder in glob.glob("*"):
        # folders that aren't directories aren't folders!
        if not os.path.isdir(folder):
            continue

        # for every file in the folder
        for file in glob.glob(folder + '/*.dat'):
            name = os.path.splitext(file)[0]
            full_folder_path = os.path.join(out_folder, folder)
            if not os.path.exists(full_folder_path):
                os.mkdir(full_folder_path)
            out_file = os.path.join(out_folder, name)
            print('out file', out_file)
            dumpTrainingTensors(name, out_path = out_file, cutoff = cutoff, chain_lookup = chain_lookup)
            #output = dumpTrainingTensors(name, out_path = out_file, cutoff = cutoff)
            #dataset.append(output)

    return dataset


def generateDatasetParallel(in_folder, out_folder, cutoff = 1000, num_cores = 1):
    print('num cores', num_cores)
    # make folder where the dataset files are gonna be placed
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # generate absolute paths so i dont have to think about relative references
    in_folder = os.path.abspath(in_folder)
    out_folder = os.path.abspath(out_folder)

    os.chdir(in_folder)

    pool = mp.Pool(num_cores)
    # process folder by folder
    for folder in glob.glob("*"):
        # folders that aren't directories aren't folders!
        if not os.path.isdir(folder):
            continue

        full_folder_path = os.path.join(out_folder, folder)
        if not os.path.exists(full_folder_path):
            os.mkdir(full_folder_path)
            
        for idx, file in enumerate(glob.glob(folder+'/*.dat')):
            res = pool.apply_async(dataGen, args=(file, folder, out_folder, cutoff), error_callback = raise_error)
    pool.close()
    pool.join()

def raise_error(error):
    raise error

# inner loop we wanna parallize
def dataGen(file, folder, out_folder, cutoff):
    name = os.path.splitext(file)[0]
    out_file = os.path.join(out_folder, name)
    print('out file', out_file)
    dumpTrainingTensors(name, out_path = out_file, cutoff = cutoff)





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate features data files from dTERMen .dat files')
    parser.add_argument('in_folder', help = 'input folder containing .dat files in proper directory structure', default='dTERMen_data')
    parser.add_argument('out_folder', help = 'folder where features will be placed', default='features')
    parser.add_argument('--cutoff', dest='cutoff', help = 'max number of MSA entries per TERM', default = 1000, type=int)
    parser.add_argument('-n', dest='num_cores', help = 'number of cores to use', default = 1, type = int)
    args = parser.parse_args()
    generateDatasetParallel(args.in_folder, args.out_folder, cutoff = args.cutoff, num_cores = args.num_cores)
