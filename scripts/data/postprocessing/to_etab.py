import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from shutil import copyfile

import numpy as np
from search_utils import find_dtermen_folder
from tqdm import tqdm

from terminator.utils.common import AA_to_int, int_to_AA


# print to stderr
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def to_etab_file_wrapper(etab_matrix, E_idx, idx_dict, out_path):
    try:
        return to_etab_file(etab_matrix, E_idx, idx_dict, out_path)
    except Exception as e:
        eprint(out_path)
        eprint(idx_dict)
        traceback.print_exc()
        return False, out_path


# should work for multi-chain proteins now
def to_etab_file(etab_matrix, E_idx, idx_dict, out_path):
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
            chain_j, j_resid = idx_dict[j_idx]

            for i, i_slice in enumerate(k_slice):
                i_3lt_id = int_to_AA[i]
                for j, nrg in enumerate(i_slice):
                    j_3lt_id = int_to_AA[j]

                    # every etab has two entries i, j and j, i
                    # average these nrgs
                    key = [(chain_i, i_resid, i_3lt_id), (chain_j, j_resid, j_3lt_id)]
                    key.sort(key=lambda x: x[1])
                    key = tuple(key)
                    if key not in pair_nrgs.keys():
                        pair_nrgs[key] = nrg
                    else:
                        current_nrg = pair_nrgs[key]
                        pair_nrgs[key] = (current_nrg + nrg) / 2

    for key, nrg in sorted(pair_nrgs.items(), key=lambda pair: pair[0][0][1]):
        chain_i, i_resid, i_3lt_id = key[0]
        chain_j, j_resid, j_3lt_id = key[1]
        out_file.write(
            '{},{} {},{} {} {} {}\n'.format(
                chain_i, i_resid,
                chain_j, j_resid,
                i_3lt_id, j_3lt_id,
                nrg)
        )

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
                residx = data[22:27].strip()  # rip i didn't know about icodes

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate etabs')
    parser.add_argument('--output_dir',
                        help='output directory',
                        default='test_run')
    parser.add_argument('--num_cores',
                        help='number of processes for parallelization',
                        default=1)
    parser.add_argument("--dtermen_data",
                        help="Root directory for all dTERMen runs")
    parser.add_argument('-u',
                        dest='update',
                        help='flag for force updating etabs',
                        default=False,
                        action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(os.path.join(args.output_dir, 'etabs')):
        os.mkdir(os.path.join(args.output_dir, 'etabs'))
    print("made etabs dir")

    with open(os.path.join(args.output_dir, 'net.out'), 'rb') as fp:
        dump = pickle.load(fp)
    print("loaded dump")

    pool = mp.Pool(int(args.num_cores))
    start = time.time()
    pbar = tqdm(total=len(dump))
    not_worked = []

    print("starting etab dump")
    for data in dump:
        pdb = data['ids'][0]
        E_idx = data['idx'][0].copy()
        etab = data['out'][0].copy()

        print(pdb)
        pdb_path = find_dtermen_folder(pdb, args.dtermen_data)
        idx_dict = get_idx_dict(os.path.join(pdb_path, f'{pdb}.red.pdb'))

        out_path = os.path.join(args.output_dir, 'etabs/' + pdb + '.etab')

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

        res = pool.apply_async(to_etab_file_wrapper,
                               args=(etab, E_idx, idx_dict, out_path),
                               callback=check_worked,
                               error_callback=raise_error)

    pool.close()
    pool.join()
    pbar.close()
    print(f"errors in {not_worked}")
    for path in not_worked:
        os.remove(path)
    end = time.time()
    print(f"done, took {end - start} seconds")
