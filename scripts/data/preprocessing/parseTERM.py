import numpy as np
import os
import json
import pickle
import argparse
from terminator.utils.common import seq_to_ints
from scipy.linalg import block_diag
from parseEtab import parseEtab

HEAD_LEN = len('* TERM ')

'''
Function that parses all relavent data from TERM data dumps

Returns the sequence numerically encoded, the selection,
full sequence ppoe, and all TERMs found
'''
def parseTERMdata(filename):
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

# assign an index based on how close you are to the central element used to create the TERM
# we set 0 to the central element, increment as you go N->C, decrement as you go C->N
# central element is middle residue for a first order TERM, central contact for second order TERM
def contact_idx(focus):
    l = len(focus)
    # if all residues are consecutive, first order TERM
    if focus[-1] - focus[0] + 1 == l:
        if l % 2 == 1: # if it's odd we can easily make this
            return [i - l//2 for i in range(l)]
        else: # if it's even we assign both center elements 0
            tail_list = [i for i in range(l//2)]
            head_list = [-i for i in reversed(tail_list)]
            return head_list + tail_list
    else: # otherwise, second order TERM
        breakpoint = 0
        for i in range(1, l):
            if focus[i] - focus[i-1] != 1:
                breakpoint = i
                break
        first_chain = focus[:breakpoint]
        second_chain = focus[breakpoint:]
        return contact_idx(first_chain) + contact_idx(second_chain)
        

def makeDataPickle(in_path, out_path = None):
    data = parseTERMdata(in_path)

    # get to proper directory
    name = in_path.strip().split('/')[-1]
    if out_path:
        os.chdir(out_path)

    with open(name[:-4] + '.pickle', 'wb') as fp:
        pickle.dump(data, fp)

def dumpTrainingData(in_path, out_path = None, cutoff = 1000):
    data = parseTERMdata(in_path)

    term_features = []
    for term_data in data['terms']:
        training = {}
        # cutoff MSAs at top N
        training['msa'] = term_data['labels'][:cutoff]
        training['focus'] = term_data['focus']

        ppoe = term_data['ppoe']
        term_len = ppoe.shape[0]
        num_alignments = ppoe.shape[2]
        rmsd = np.expand_dims(term_data['rmsds'], 1)
        rmsd_arr = np.concatenate([rmsd for _ in range(num_alignments)], axis=1)
        rmsd_arr = np.expand_dims(rmsd_arr, 1)
        term_len_arr = np.zeros((term_len, num_alignments))
        term_len_arr = np.expand_dims(term_len_arr, 1)
        term_len_arr += term_len
        num_alignments_arr = np.zeros((term_len, num_alignments))
        num_alignments_arr = np.expand_dims(num_alignments_arr, 1)
        num_alignments_arr += num_alignments
        features = np.concatenate([ppoe, rmsd_arr, term_len_arr, num_alignments_arr], axis=1)

        # pytorch does row vector computation, but i formatted in column vectors
        # swap rows and columns
        features = features.transpose(1,2)
        # cutoff features at top N
        training['features'] = features[:cutoff]

        term_features.append(training)

    output = {
        'term_features': term_features,
        'sequence': data['sequence']
    }

    name = in_path.strip().split('/')[-1]
    if out_path:
        os.chdir(out_path)

    with open(name[:-4] + '.features', 'wb') as fp:
        pickle.dump(output, fp)

    return output

def dumpTrainingTensors(in_path, out_path = None, cutoff = 1000, save=True):
    data = parseTERMdata(in_path + '.dat')
    etab, self_etab = parseEtab(in_path + '.etab', save=False)

    term_msas = []
    term_features = []
    term_focuses = []
    term_lens = []
    for term_data in data['terms']:
        # cutoff MSAs at top N
        term_msas.append(term_data['labels'][:cutoff])
        # add focus
        term_focuses += term_data['focus']
        # append term len, the len of the focus
        term_lens.append(len(term_data['focus']))

        # cutoff ppoe at top N
        ppoe = term_data['ppoe'][:cutoff]
        term_len = ppoe.shape[0]
        num_alignments = ppoe.shape[2]
        # cutoff rmsd at top N
        rmsd = np.expand_dims(term_data['rmsds'][:cutoff], 1)
        rmsd_arr = np.concatenate([rmsd for _ in range(num_alignments)], axis=1)
        rmsd_arr = np.expand_dims(rmsd_arr, 1)
        term_len_arr = np.zeros((term_len, num_alignments))
        term_len_arr = np.expand_dims(term_len_arr, 1)
        term_len_arr += term_len
        num_alignments_arr = np.zeros((term_len, num_alignments))
        num_alignments_arr = np.expand_dims(num_alignments_arr, 1)
        num_alignments_arr += num_alignments
        features = np.concatenate([ppoe, rmsd_arr, term_len_arr, num_alignments_arr], axis=1)

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

    # create attn mask for transformer
    blocks = [np.ones((i,i)) for i in term_lens]
    # create a block diagonal matrix mask
    src_mask = block_diag(*blocks)
    # note: masks need to be inverted upon use.
    # they're stored this way so padding is easier later

    output = {
        'features': features_tensor,
        'msas': msa_tensor,
        'focuses': term_focuses,
        'mask': src_mask,
        'term_lens': len_tensor,
        'sequence': data['sequence'],
        'seq_len': len(data['selection']),
        'etab': etab,
        'selfE': self_etab,
        'chain_lens': data['chain_lens']
    }

    """
    print(features_tensor.shape)
    print(msa_tensor.shape)
    print(focus_tensor.shape)
    print(sum(len_tensor))

    import torch
    splits = torch.split(torch.from_numpy(features_tensor), term_lens, dim=2)
    print(len(splits))
    print(len(term_features))
    print(splits[0].shape)
    print(term_features[0].shape)

    exit()
    """

    if save:
        if out_path:
            os.chdir(out_path)

        name = in_path.strip().split('/')[-1]
        with open(name[:-4] + '.features', 'wb') as fp:
            pickle.dump(output, fp)

    return output

# ive decided not to use this approach but im leaving this here for now anyway
def makeDataFolder(in_path, out_path = None):
    data = parseTERMdata(in_path)

    upper_path, name = in_path.strip().split('/')[-2:]
    basename = name[:-4]

    # get to proper directory
    if out_path:
        os.chdir(out_path)

    if not os.path.exists(upper_path):
        os.makedirs(upper_path)
    os.chdir(upper_path)
    os.makedirs(basename)
    os.chdir(basename)

    # save sequence-level features
    overview = {'sequence': data['sequence'], 'selection': data['selection']}
    with open('overview.json', 'w') as overview_fp:
        json.dump(overview, overview_fp)
    np.save('ppoe.npy', data['ppoe'])

    # save each TERM's data in its own folder
    for t in data['terms']:
        idx = str(t['idx'])
        os.makedirs(idx)
        os.chdir(idx)

        np.save('labels', t.pop('labels'))
        np.save('rmsds', t.pop('rmsds'))
        np.save('ppoe', t.pop('ppoe'))
        with open('overview.json', 'w') as overview_fp:
            json.dump(t, overview_fp)

        os.chdir('..')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert a dTERMen output .dat file and .etab into a pickle file with data in a python-friendly format')
    parser.add_argument('dat', metavar='f', help = 'input .etab/.dat file basename')
    parser.add_argument('--out', dest='out', help = 'output path (no file extension)')
    parser.add_argument('--cutoff', dest='cutoff', help = 'max number of MSAs per TERM', default=1000)
    args = parser.parse_args()
    dumpTrainingTensors(args.dat, out_path = args.out, cutoff = args.cutoff)
