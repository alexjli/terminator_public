import argparse
import json
import os
import pickle

import numpy as np
from parseEtab import parseEtab
from scipy.linalg import block_diag

from terminator.utils.common import seq_to_ints

HEAD_LEN = len('* TERM ')


def parseTERMdata(filename):
    '''
    Function that parses all relavent data from TERM data dumps

    Returns the sequence numerically encoded, the selection,
    full sequence ppoe, and all TERMs found
    '''
    fp = open(filename, 'r')

    # parse initial PDB parameters
    # PDB sequence and selected residues
    seq = fp.readline().strip()
    seq = seq_to_ints(seq)
    selection = fp.readline().strip().split(' ')
    selection = [int(i) for i in selection]

    # parse phi, psi, omega, and environ vals
    # keep track of chain len based on phi=999
    ppoe = []
    chain_lens = []
    current_chain_len = 0

    current_line = fp.readline()
    while current_line[0] != '*':
        data = current_line.strip().split(' ')
        data = [float(i) for i in data]
        # if phi = 999. start new chain
        if data[0] == 999:
            chain_lens.append(current_chain_len)
            current_chain_len = 0
        ppoe.append(data)
        current_chain_len += 1
        current_line = fp.readline()

    # append last chain len
    chain_lens.append(current_chain_len)

    # the first chain len will always be 0, so pop that off
    chain_lens.pop(0)
    ppoe = np.array(ppoe)

    assert sum(chain_lens) == len(seq), "sum of chain lens != total seq len"

    # parse TERMs from rest of file
    terms = []
    while current_line != '':
        term, current_line = parseTERM(fp, current_line)
        terms.append(term)

    fp.close()
    output = {}
    output['sequence'] = seq
    output['selection'] = selection
    output['ppoe'] = ppoe
    output['terms'] = terms
    output['chain_lens'] = chain_lens
    return output


def parseTERM(fp, lastline):
    term_dict = {}
    # idx: index of TERM
    term_dict['idx'] = int(lastline.strip().split(' ')[-1])
    # print(term_dict['idx'])

    # focus: residues in TERM
    focus = fp.readline().strip().split()
    focus = [int(i) for i in focus]
    term_dict['focus'] = focus
    term_dict['contact_idx'] = contact_idx(focus)
    focus_len = len(focus)

    # parse each individual structure match, append to term
    term_labels = []
    term_rmsds = []
    term_ppoe = []

    current_line = fp.readline().strip()
    while current_line != '' and current_line[0] != '*':
        data = current_line.split(' ')
        label, rmsd, ppoe = seq_to_ints(data[0]), float(data[1]), [float(i) for i in data[2:]]
        ppoe = np.array(ppoe).reshape((4, focus_len))
        term_labels.append(label)
        term_rmsds.append(rmsd)
        term_ppoe.append(ppoe)

        current_line = fp.readline().strip()

    # reshape as numpy arrays
    term_dict['labels'] = np.concatenate([term_labels])
    term_dict['rmsds'] = np.concatenate([term_rmsds])
    term_dict['ppoe'] = np.concatenate([term_ppoe])
    return term_dict, current_line


def contact_idx(focus):
    """
    Assign an index based on how close you are to the central element used to create the TERM
    We set 0 to the central element, increment as you go N->C, decrement as you go C->N
    Central element is middle residue for a first order TERM, central contact for second order TERM
    """
    l = len(focus)
    # if all residues are consecutive, first order TERM
    if focus[-1] - focus[0] + 1 == l:
        if l % 2 == 1:  # if it's odd we can easily make this
            return [i - l // 2 for i in range(l)]
        else:  # if it's even we assign both center elements 0
            tail_list = [i for i in range(l // 2)]
            head_list = [-i for i in reversed(tail_list)]
            return head_list + tail_list
    else:  # otherwise, second order TERM
        breakpoint = 0
        for i in range(1, l):
            if focus[i] - focus[i - 1] != 1:
                breakpoint = i
                break
        first_chain = focus[:breakpoint]
        second_chain = focus[breakpoint:]
        return contact_idx(first_chain) + contact_idx(second_chain)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Convert a dTERMen output .dat file and .etab into a pickle file with data in a python-friendly format')
    parser.add_argument('dat',
                        metavar='f',
                        help='input .etab/.dat file basename')
    parser.add_argument('--out',
                        dest='out',
                        help='output path (no file extension)')
    parser.add_argument('--cutoff',
                        dest='cutoff',
                        help='max number of MSAs per TERM',
                        default=1000)
    args = parser.parse_args()
    dumpTrainingTensors(args.dat, out_path=args.out, cutoff=args.cutoff)
