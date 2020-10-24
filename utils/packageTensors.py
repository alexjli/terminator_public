import numpy as np
import pickle
from scipy.linalg import block_diag
import glob

from .parseTERM import parseTERMdata
from .parseEtab import parseEtab
from .parseCoords import parseCoords

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

        selected_features = [sin_ppo, cos_ppo, env, rmsd_arr, term_len_arr]

        features = np.concatenate(selected_features, axis=1)

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
    else:
        chains = sorted(coords.keys())
        coords_tensor = np.vstack([coords[c] for c in chains])

    assert coords_tensor.shape[0] == len(data['sequence']), "num aa coords != seq length"

    output = {
        'pdb': pdb,
        'coords': coords_tensor,
        'features': features_tensor,
        'msas': msa_tensor,
        'focuses': term_focuses,
        'term_lens': len_tensor,
        'sequence': np.array(data['sequence']),
        'seq_len': len(data['selection']),
        'chain_lens': data['chain_lens']
        #'etab': etab,
        #'selfE': self_etab
    }

    if save:
        if not out_path:
            out_path = ''

        with open(out_path + '.features', 'wb') as fp:
            pickle.dump(output, fp)
        with open(out_path + '.length', 'w') as fp:
            fp.write(str(len(term_focuses)))

    print('Done with', pdb)

    return output


