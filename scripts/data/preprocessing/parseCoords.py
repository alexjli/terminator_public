"""Functions to parse :code:`.red.pdb` files"""
import pickle

import numpy as np

from terminator.utils.common import aa_three_to_one


def parseCoords(filename, save=True):
    """ Parse coordinates from :code:`.red.pdb` files, and dump in
    files if specified.

    Args
    ====
    filename : str
        path to :code:`.red.pdb` file

    save : bool, default=True
        whether or not to dump the results

    Returns
    =======
    chain_tensors : dict
        Dictionary mapping chain IDs to arrays of atomic coordinates.

    seq : str
        Sequence of all chains concatenated.
    """
    chain_dict = {}
    with open(filename, 'r') as fp:
        for line in fp:
            data = line.strip()
            if data[:3] == 'TER' or data[:3] == 'END':
                continue
            try:
                element = data[13:16].strip()
                residue = data[17:20].strip()
                residx = data[22:27].strip()
                chain = data[21]
                x = data[30:38].strip()
                y = data[38:46].strip()
                z = data[46:54].strip()
                coords = [float(coord) for coord in [x, y, z]]
                # print(element, chain, coords)
            except Exception as e:
                print(data)
                raise e

            if chain not in chain_dict.keys():
                chain_dict[chain] = {element: [] for element in ['N', 'CA', 'C', 'O']}
                chain_dict[chain]["seq_dict"] = {}

            # naively model terminal carboxylate as a single O atom
            # (i cant find the two oxygens so im just gonna use OXT)
            if element == 'OXT':
                element = 'O'
            chain_dict[chain][element].append(coords)

            seq_dict = chain_dict[chain]["seq_dict"]
            if residx not in seq_dict.keys():
                if residue == 'MSE':  # convert MSE (seleno-met) to MET
                    residue = 'MET'
                elif residue == 'SEP':  # convert SEP (phospho-ser) to SER
                    residue = 'SER'
                elif residue == 'TPO':  # convert TPO (phospho-thr) to THR
                    residue = 'THR'
                elif residue == 'PTR':  # convert PTR (phospho-tyr) to TYR
                    residue = 'TYR'
                elif residue == 'CSO':  # convert CSO (hydroxy-cys) to CYS
                    residue = 'CYS'
                elif residue == 'SEC':  # convert SEC (seleno-cys) to CYS
                    residue = 'CYS'
                seq_dict[residx] = aa_three_to_one(residue)

    chain_tensors = {}
    seq = ""
    for chain in chain_dict.keys():
        coords = [chain_dict[chain][element] for element in ['N', 'CA', 'C', 'O']]
        chain_tensors[chain] = np.stack(coords, 1)
        seq_dict = chain_dict[chain]["seq_dict"]
        chain_seq = "".join([seq_dict[i] for i in seq_dict.keys()])
        assert len(chain_seq) == chain_tensors[chain].shape[0], (chain_seq, chain_tensors[chain].shape, filename)
        seq += "".join([seq_dict[i] for i in seq_dict.keys()])

    if save:
        with open(filename[:-8] + '.coords', 'wb') as fp:
            pickle.dump(chain_tensors, fp)
        with open(filename[:-8] + '.seq', 'w') as fp:
            fp.write(seq)

    return chain_tensors, seq
