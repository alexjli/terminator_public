import numpy as np
import pickle
import os
from preprocessing.common import AA_to_int

int_to_AA = {y:x for x,y in AA_to_int.items() if len(x) is 3}

ifsdata = '/home/ifsdata/scratch/grigoryanlab/alexjli'

# currently only works for single chain etabs
def to_etab_file(etab_matrix, E_idx, idx_dict, out_path):
    chain = None
    if len(idx_dict.keys()) == 1:
        chain = next(iter(idx_dict.keys()))
        idx_dict = idx_dict[chain]

    out_file = open(out_path, 'w')

    # etab matrix: l x k x 20 x 20
    self_etab = etab_matrix[:, 0]
    pair_etab = etab_matrix[:, 1:]
    E_idx = E_idx[:, 1:]

    # l x 20
    self_nrgs = np.diagonal(self_etab, offset=0, axis1=-2, axis2=-1)
    for aa_idx, aa_nrgs in enumerate(self_nrgs):
        resid = idx_dict[aa_idx]
        for aa_int_id, nrg in enumerate(aa_nrgs):
            aa_3lt_id = int_to_AA[aa_int_id]
            out_file.write('{},{} {} {}\n'.format(chain, resid, aa_3lt_id, nrg))

    pair_nrgs = {}

    # l x k-1 x 20 x 20
    for i_idx, nrg_slice in enumerate(pair_etab):
        for k, k_slice in enumerate(nrg_slice):
            j_idx = E_idx[i_idx][k]
            i_resid, j_resid = idx_dict[i_idx], idx_dict[j_idx]
            for i, i_slice in enumerate(k_slice):
                i_3lt_id = int_to_AA[i]
                for j, nrg in enumerate(i_slice):
                    j_3lt_id = int_to_AA[j]
                    
                    # every etab has two entries i, j and j, i
                    # average these nrgs
                    key = [(chain, i_resid, i_3lt_id), (chain, j_resid, j_3lt_id)]
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

def get_idx_dict(pdb):
    chain_dict = {}
    with open(pdb, 'r') as fp:
        current_idx = 0
        for line in fp:
            data = line.strip()
            if data == 'TER' or data == 'END':
                continue 
            try:
                chain = data[21]
                idx = int(data[22:26].strip())
            except Exception as e:
                print(data)
                raise e

            if chain not in chain_dict.keys():
                chain_dict[chain] = {}
                current_idx = 0

            if idx not in chain_dict[chain].values():
                chain_dict[chain][current_idx] = idx
                current_idx += 1

    return chain_dict

if __name__ == '__main__':
    ironfs = '/home/ironfs/scratch/grigoryanlab/alexjli/'
    os.chdir(ifsdata)
    p1 = ironfs + 'dTERMen_speedtest200_clique1/'
    p2 = ironfs + 'dTERMen_speedtest200_clique1_p2/'
    p3 = ironfs + 'dTERMen_speedtest200_clique1_p3/'
    p4 = ironfs + 'monomer_DB/'
    p5 = ironfs + 'monomer_DB_2/'

    if not os.path.isdir('etabs'):
        os.mkdir('etabs')

    with open('net.out', 'rb') as fp:
        dump = pickle.load(fp)

    for data in dump:
        pdb = data['ids'][0]
        print(pdb)
        idx_dict = None
        if os.path.isdir(p1 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p1, pdb, pdb))
        elif os.path.isdir(p2 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p2, pdb, pdb))
        elif os.path.isdir(p3 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p3, pdb, pdb))
        elif os.path.isdir(p4 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p4, pdb, pdb))
        elif os.path.isdir(p5 + pdb):
            idx_dict = get_idx_dict('{}{}/{}.red.pdb'.format(p5, pdb, pdb))
        else:
            raise Exception('umwhat')

        E_idx = data['idx'][0]
        etab = data['out'][0]

        out_path = 'etabs/' + pdb + '.etab'

        to_etab_file(etab, E_idx, idx_dict, out_path)


