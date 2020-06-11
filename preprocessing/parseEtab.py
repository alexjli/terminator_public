import numpy as np
from common import *
import argparse

"""
Given an etab file, parse the corresponding Potts model parameters
This assumes that the full protein etab is provided
Will not work with etabs computed on partial chains
"""
def parseEtab(filename):

    if filename.split('.')[-1] != 'etab':
        raise ValueError('Input file is not an etab file!')

    selfE = []
    pairE = []
    id_to_resid = {}
    fp = open(filename, 'r')
    # this loop requires that all self energies
    # occur before pair energies
    for idx, line in enumerate(fp):
        l = line.strip().split(' ')

        # self energies
        if len(l) == 3:
            id = l[0]
            resid = idx//20
            id_to_resid[id] = resid

            residue = aa_to_int(l[1])
            E = float(l[2])

            selfE.append({'resid': resid, 'residue': residue, 'E': E})

        # couplings
        elif len(l) == 5:
            id0 = l[0]
            id1 = l[1]
            resid0 = id_to_resid[id0]
            resid1 = id_to_resid[id1]

            residue0 = aa_to_int(l[2])
            residue1 = aa_to_int(l[3])
            E = float(l[4])

            pairE.append({'resid0': resid0, 'resid1': resid1,
                          'residue0': residue0, 'residue1': residue1, 'E': E})
        else:
            raise ValueError("Something doesn't look right at line %d: %s" % (idx, line))

    fp.close()

    # number of amino acids in all chains
    L = max(id_to_resid.values()) + 1

    potts_selfE = np.zeros((L, 22))
    potts = np.zeros((L, L, 20, 20))

    for data in selfE:
        resid = data['resid']
        residue = data['residue']
        slice = potts[resid][resid]
        slice[residue][residue] = data['E']

        potts_selfE[resid][residue] = data['E']

    for data in pairE:
        resid0 = data['resid0']
        resid1 = data['resid1']
        residue0 = data['residue0']
        residue1 = data['residue1']
        slice0 = potts[resid0][resid1]
        slice0[residue0][residue1] = data['E']
        slice1 = potts[resid1][resid0]
        slice1[residue1][residue0] = data['E']

    np.save(filename, potts)
    np.save(filename[:-5] + '_selfE', potts_selfE)

    # testing that the values in the potts parameters is correct
    """
    np.set_printoptions(threshold=np.inf, precision=2, linewidth=300)
    print(potts[0][0])
    """

    return potts, potts_selfE

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert an etab into a numpy array of Potts model parameters')
    parser.add_argument('etab', metavar='f', help = 'input etab file')
    args = parser.parse_args()
    parseEtab(args.etab)

"""
def parseEtab(filename):
    selfE = []
    pairE = []
    chainLens = {}
    fp = open(filename, 'r')
    # this loop requires that all self energies
    # occur before pair energies
    for idx, line in enumerate(fp):
        l = line.strip().split(' ')

        # self energies
        if len(l) == 3:
            chain, resid = l[0].split(',')
            resid = int(resid)

            # count how many residue are in each chain
            if chain in chainLens.keys():
                if resid > chainLens[chain]:
                    chainLens[chain] = resid
            else:
                chainLens[chain] = 1

            residue = aa_to_int(l[1])
            E = float(l[2])

            selfE.append([chain, resid, residue, E])
        elif len(l) == 5:
            chain1, resid1 = l[0].split(',')
            chain2, resid2 = l[1].split(',')
            resid1 = int(resid1)
            resid2 = int(resid2)

            residue1 = aa_to_int(l[2])
            residue2 = aa_to_int(l[3])
            E = float(l[4])

            pairE.append([chain1, chain2, resid1, resid2, residue1, residue2, E])
        else:
            raise ValueError("Something doesn't look right at line %d: %s" % (idx, line))

    fp.close()

    offset = [0]
    chains = sorted(chainLens.keys())
    for c in chains:
        offset.append(sum(offset) + chainLens[c])
    offset.pop()
    offset = {chains[i]: offset[i] for i in range(len(chains))}

    # update all resids
    for e_data in selfE:
        chain = e_data[0]
        e_data[1] += offset[chain]
        e_data.pop(0)

    for e_data in pairE:
        chain1 = e_data[0]
        chain2 = e_data[1]
        e_data[2] += offset[chain1] - 1
        e_data[3] += offset[chain2] - 1
        e_data.pop(1)
        e_data.pop(0)

    #potts = np.zeros()
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(selfE)
    #print(selfE[0], selfE[20])
    #print(pairE[0], pairE[20])
"""
