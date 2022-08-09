import argparse
import glob
import os
import re
import numpy as np
import multiprocessing as mp
import traceback

"""This script is used to condense Bcl-2 family energy tables for a protein-peptide complex into one just for the 20-amino-acid length peptide. This is done by adding all pair energies between the peptide and protein to the self-energies of the peptide energy table values."""



myAmino = ["R","H","K","D","E","S","T","N","Q","C","G","P","A","V","I","L","M","F","Y","W"]
FullAmino = ["ARG","HIS","LYS","ASP","GLU","SER","THR","ASN","GLN","CYS","GLY","PRO","ALA","VAL","ILE","LEU","MET","PHE","TYR","TRP"]
aminos = {FullAmino[i]:myAmino[i] for i in range(len(myAmino))}
CHAIN_LENGTH = 20

def get_full_native(model):
    pdb = "/home/gridsan/mlu/keatinglab_shared/mlu/bcl2/clean_pdb/{}.pdb".format(model)
    lines = [line.rstrip('\n') for line in open(pdb)]

    seq = {"A": "", "B": ""}
    index = ""
    chain = ""
    for l in lines:
        sp = re.split("\s+",l)
        if sp[0] != "ATOM":
            continue
        if sp[5] != index or sp[4] != chain:
            index = sp[5]
            chain = sp[4]
            seq[chain] += aminos[sp[3]]
    
    return seq

# zero is used as padding
AA_to_int = {
    'A': 1,
    'ALA': 1,
    'C': 2,
    'CYS': 2,
    'D': 3,
    'ASP': 3,
    'E': 4,
    'GLU': 4,
    'F': 5,
    'PHE': 5,
    'G': 6,
    'GLY': 6,
    'H': 7,
    'HIS': 7,
    'I': 8,
    'ILE': 8,
    'K': 9,
    'LYS': 9,
    'L': 10,
    'LEU': 10,
    'M': 11,
    'MET': 11,
    'N': 12,
    'ASN': 12,
    'P': 13,
    'PRO': 13,
    'Q': 14,
    'GLN': 14,
    'R': 15,
    'ARG': 15,
    'S': 16,
    'SER': 16,
    'T': 17,
    'THR': 17,
    'V': 18,
    'VAL': 18,
    'W': 19,
    'TRP': 19,
    'Y': 20,
    'TYR': 20,
    'X': 21
}
## amino acid to integer
atoi = {key: val - 1 for key, val in AA_to_int.items()}
## integer to amino acid
iota = {y: x for x, y in atoi.items() if len(x) == 1}


def condense_etab(etab, out_folder):
    #Get Model name
    model = re.search("([A-Za-z0-9_]+)\.etab.npy",etab).group(1)

    print("now condensing:", model)
    # obtain native sequence of protein
    native_seq = get_full_native(model)

    e = np.load(etab)
    protein = native_seq["A"]
    peptide = native_seq["B"]
    new = np.zeros((len(peptide), len(peptide), 22, 22))

    # copy all self and pair energies
    for i in range(len(peptide)):
        for j in range(len(peptide)):
            for k in range(22):
                for l in range(22):
                    energy = e[len(protein)+i][len(protein)+j][k][l]
                    new[i][j][k][l] = energy
                    new[j][i][l][k] = energy
    
    # add peptide's pair energies with the protein into its self energies
    for i in range(len(protein)):
        for j in range(len(peptide)):
            for k in range(22):
                energy = e[i][len(protein)+j][atoi[protein[i]]][k]
                new[j][j][k][k] += energy
    
    fname = out_folder + os.path.basename(etab)
    np.save(fname, new)

def _raise_error(error):
    """Wrapper for error handling without crashing"""
    traceback.print_exception(Exception, error, None)

def condenseEtabs(in_folder, out_folder, num_cores=1):
    print(in_folder+"/*.npy")
    etabs = glob.glob(in_folder+"/*.npy")
    
    for etab in etabs:
        condense_etab(etab, out_folder)
    #pool = mp.Pool(num_cores, maxtasksperchild=10)

    #for etab in etabs:
    #    res = pool.apply_async(condense_etab, args=(etab, out_folder), error_callback=_raise_error)
    #pool.close()
    #pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Condense numpy etabs')
    parser.add_argument('--in_folder',
                        help='folder containing etabs to be condensed',
                        required=True)
    parser.add_argument('--out_folder', help='folder where numpy etabs will be placed', required=True)
    parser.add_argument('-n', dest='num_cores', help='number of proceses to use', type=int, default=1)
    args = parser.parse_args()

    condenseEtabs(args.in_folder, args.out_folder, num_cores=args.num_cores)

