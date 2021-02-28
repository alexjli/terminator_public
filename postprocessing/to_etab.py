import numpy as np
import pickle
import os
import argparse
from shutil import copyfile
import multiprocessing as mp
import sys
import time
import traceback
from tqdm import tqdm
import json

from filepaths import *

# zero is used as padding
AA_to_int = {
'A' : 1, 'ALA' : 1,
'C' : 2, 'CYS' : 2,
'D' : 3, 'ASP' : 3,
'E' : 4, 'GLU' : 4,
'F' : 5, 'PHE' : 5,
'G' : 6, 'GLY' : 6,
'H' : 7, 'HIS' : 7,
'I' : 8, 'ILE' : 8,
'K' : 9, 'LYS' : 9,
'L' : 10, 'LEU' : 10,
'M' : 11, 'MET' : 11,
'N' : 12, 'ASN' : 12,
'P' : 13, 'PRO' : 13,
'Q' : 14, 'GLN' : 14,
'R' : 15, 'ARG' : 15,
'S' : 16, 'SER' : 16,
'T' : 17, 'THR' : 17,
'V' : 18, 'VAL' : 18,
'W' : 19, 'TRP' : 19,
'Y' : 20, 'TYR' : 20,
'X' : 21
}

AA_to_int = {key: val-1 for key, val in AA_to_int.items()}
int_to_AA = {y:x for x,y in AA_to_int.items() if len(x) == 3}

# print to stderr
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def to_etab_file_wrapper(etab_matrix, E_idx, idx_dict, out_path, ingraham_dict=None):
    try:
        return to_etab_file(etab_matrix, E_idx, idx_dict, out_path, ingraham_dict)
    except Exception as e:
        eprint(out_path)
        eprint(idx_dict)
        traceback.print_exc()
        return False, out_path

# should work for multi-chain proteins now
def to_etab_file(etab_matrix, E_idx, idx_dict, out_path, ingraham_dict=None):
    out_file = open(out_path, 'w')

    # etab matrix: l x k x 20 x 20
    self_etab = etab_matrix[:, 0]
    pair_etab = etab_matrix[:, 1:]
    E_idx = E_idx[:, 1:]

    # l x 20
    self_nrgs = np.diagonal(self_etab, offset=0, axis1=-2, axis2=-1)
    for aa_idx, aa_nrgs in enumerate(self_nrgs):
        try:
            chain, resid = idx_dict[aa_idx]
        except Exception as e:
            eprint("num residues: ", len(self_nrgs))
            eprint(out_path)
            eprint(idx_dict)
            traceback.print_exc()
            return False, out_path
        for aa_int_id, nrg in enumerate(aa_nrgs):
            aa_3lt_id = int_to_AA[aa_int_id]
            out_file.write('{},{} {} {}\n'.format(chain, resid, aa_3lt_id, nrg))

    num_aa = self_nrgs.shape[0]
    pair_nrgs = {}

    # l x k-1 x 20 x 20
    for i_idx, nrg_slice in enumerate(pair_etab):
        for k, k_slice in enumerate(nrg_slice):
            j_idx = E_idx[i_idx][k]
            chain_i, i_resid = idx_dict[i_idx]
            # this is just for ingraham proteins
            # if this triggers otherwise, there's a bug
            if ingraham_dict:
                j_idx = ingraham_dict[int(j_idx)]
            chain_j, j_resid = idx_dict[j_idx]

            for i, i_slice in enumerate(k_slice):
                i_3lt_id = int_to_AA[i]
                for j, nrg in enumerate(i_slice):
                    j_3lt_id = int_to_AA[j]
                    
                    # every etab has two entries i, j and j, i
                    # average these nrgs
                    key = [(chain_i, i_resid, i_3lt_id), (chain_j, j_resid, j_3lt_id)]
                    key.sort(key = lambda x: x[1])
                    key = tuple(key)
                    if key not in pair_nrgs.keys():
                        pair_nrgs[key] = nrg
                    else:
                        current_nrg = pair_nrgs[key]
                        pair_nrgs[key] = (current_nrg + nrg)/2

    for key, nrg in sorted(pair_nrgs.items(), key = lambda pair: pair[0][0][1]):
        chain_i, i_resid, i_3lt_id = key[0]
        chain_j, j_resid, j_3lt_id = key[1]
        out_file.write('{},{} {},{} {} {} {}\n'.format(chain_i, i_resid,
                                                       chain_j, j_resid,
                                                       i_3lt_id,
                                                       j_3lt_id,
                                                       nrg))

    out_file.close()
    return True, out_path

def get_idx_dict(pdb, chain_filter=None): 
    idx_dict = {}
    with open(pdb, 'r') as fp:
        current_idx = 0
        for line in fp:
            data = line.strip()
            if data == 'TER' or data == 'END':
                continue 
            try:
                chain = data[21]
                """
                residx = int(data[22:26].strip())
                icode = data[26]
                if icode != ' ':
                    residx = str(residx) + icode
                """
                residx = data[22:27].strip() #rip i didn't know about icodes

            except Exception as e:
                print(data)
                raise e

            if chain_filter:
                if chain != chain_filter:
                    continue

            if (chain, residx) not in idx_dict.values():
                idx_dict[current_idx] = (chain, residx)
                current_idx += 1

    return idx_dict

def load_jsonl(jsonl_file):
    alphabet='ACDEFGHIKLMNPQRSTVWY'
    alphabet_set = set([a for a in alphabet])
    with open(jsonl_file) as f:
        data = {}

        lines = f.readlines()
        start = time.time()
        for i, line in enumerate(lines):
            entry = json.loads(line)
            seq = entry['seq']
            name = entry['name']

            # Convert raw coords to np arrays
            for key, val in entry['coords'].items():
                entry['coords'][key] = np.asarray(val)

            # Check if in alphabet
            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                data[name] = entry

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                print('{} entries ({} loaded) in {:.1f} s'.format(len(data), i+1, elapsed))

    return data

def gen_mask(batch):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)

        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        S[i, :l] = indices

    # Mask
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3)))
    return mask[0]

def filter_etab(etab, E_idx, chain_set, pdb):
    data = [chain_set[pdb]]
    mask = gen_mask(data)

    count = 0
    ingraham_dict = {}
    for idx, item in enumerate(mask):
        if item:
            ingraham_dict[idx] = count
            count += 1
            
    return etab[mask], E_idx[mask], ingraham_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate etabs')
    parser.add_argument('--output_dir', help = 'output directory', default = 'test_run')
    parser.add_argument('--num_cores', help = 'number of processes for parallelization', default=1)
    parser.add_argument('-u', dest = 'update', help = 'flag for force updating etabs', default=False, action = 'store_true')
    args = parser.parse_args()

    output_dir = os.path.join(OUTPUT_DIR, args.output_dir)

    p1 = os.path.join(INPUT_DATA, 'dTERMen_speedtest200_clique1/')
    p2 = os.path.join(INPUT_DATA, 'dTERMen_speedtest200_clique1_p2/')
    p3 = os.path.join(INPUT_DATA, 'dTERMen_speedtest200_clique1_p3/')
    p4 = os.path.join(INPUT_DATA, 'monomer_DB_1/')
    p5 = os.path.join(INPUT_DATA, 'monomer_DB_2/')
    p6 = os.path.join(INPUT_DATA, 'monomer_DB_3/')
    p7 = os.path.join(INPUT_DATA, 'seq_id_50_resid_500/')
    p8 = os.path.join(INPUT_DATA, 'ingraham_db/PDB/')

    if not os.path.isdir(os.path.join(output_dir, 'etabs')):
        os.mkdir(os.path.join(output_dir, 'etabs'))

    with open(os.path.join(output_dir, 'net.out'), 'rb') as fp:
        dump = pickle.load(fp)

    pool = mp.Pool(int(args.num_cores))
    start = time.time()
    pbar = tqdm(total = len(dump))
    not_worked = []

    chain_set = load_jsonl(os.path.join(INPUT_DATA, "chain_set.jsonl"))

    for data in dump:
        pdb = data['ids'][0]
        E_idx = data['idx'][0].copy()
        etab = data['out'][0].copy()
        ingraham_dict = None

        print(pdb)
        idx_dict = None
        if os.path.isdir(p1 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p1, pdb, pdb))
            # copyfile(f'{p1}{pdb}/{pdb}.red.pdb', os.path.join(output_dir, 'etabs', f'{pdb}.red.pdb'))
        elif os.path.isdir(p2 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p2, pdb, pdb))
            # copyfile(f'{p2}{pdb}/{pdb}.red.pdb', os.path.join(output_dir, 'etabs', f'{pdb}.red.pdb'))
        elif os.path.isdir(p3 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p3, pdb, pdb))
            # copyfile(f'{p3}{pdb}/{pdb}.red.pdb', os.path.join(output_dir, 'etabs', f'{pdb}.red.pdb'))
        elif os.path.isdir(p4 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p4, pdb, pdb))
            # copyfile(f'{p4}{pdb}/{pdb}.red.pdb', os.path.join(output_dir, 'etabs', f'{pdb}.red.pdb'))
        elif os.path.isdir(p5 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p5, pdb, pdb))
            # copyfile(f'{p5}{pdb}/{pdb}.red.pdb', os.path.join(output_dir, 'etabs', f'{pdb}.red.pdb'))
        elif os.path.isdir(p6 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p6, pdb, pdb))
            # copyfile(f'{p6}{pdb}/{pdb}.red.pdb', os.path.join(output_dir, 'etabs', f'{pdb}.red.pdb'))
        elif os.path.isdir(p7 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p7, pdb, pdb))
        elif os.path.exists(f"{p8}{pdb[1:3]}/{pdb[:-2].upper()}_{pdb[-1]}.pdb"): # yes i know its bad
            etab, E_idx, ingraham_dict = filter_etab(etab, E_idx, chain_set, pdb)
            pdb, chain = pdb[:-2].upper(), pdb[-1]
            mid = pdb[1:3].lower()
            idx_dict = get_idx_dict('{}{}/{}_{}.pdb'.format(p8, mid, pdb, chain), chain)
            pdb = f"{pdb}_{chain}"
        else:
            raise Exception('umwhat')
        out_path = os.path.join(output_dir, 'etabs/' + pdb + '.etab')

        if os.path.exists(out_path) and not args.update:
            print(f"{pdb} already exists, skipping")
            pbar.update()
            continue

        def check_worked(res):
            worked, out_path = res
            pbar.update()
            if not worked:
                not_worked.append(out_path)

        def raise_error(error):
            raise error
        res = pool.apply_async(to_etab_file_wrapper, args = (etab, E_idx, idx_dict, out_path, ingraham_dict), callback = check_worked, error_callback = raise_error) 

    pool.close()
    pool.join()
    pbar.close()
    print(f"errors in {not_worked}")
    for path in not_worked:
        os.remove(path)
    end = time.time()
    print(f"done, took {end - start} seconds")

