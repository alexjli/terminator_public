import numpy as np
import pickle
from scipy.linalg import block_diag
from scipy.special import softmax
import glob

from .parseTERM import parseTERMdata
from .parseEtab import parseEtab
from .parseCoords import parseCoords

NUM_AA = 21 # including X

def dumpTrainingTensors(in_path, out_path = None, cutoff = 1000, save=True, stats = False, weight_fn = "neg"):
    coords = parseCoords(in_path + '.red.pdb', save=False)
    data = parseTERMdata(in_path + '.dat')
    etab, self_etab = parseEtab(in_path + '.etab', save=False)

    selection = data['selection']

    term_msas = []
    term_features = []
    term_sing_stats = []
    term_pair_stats = []
    term_focuses = []
    term_contact_idxs = []
    term_lens = []
    # compute TERM features
    for term_data in data['terms']:
        focus = term_data['focus']
        # only take data for residues that are in the selection
        take = [i for i in range(len(focus)) if focus[i] in selection]

        msa = term_data['labels']
        # apply take
        msa = np.take(msa, take, axis=-1)
        # cutoff MSAs at top N
        term_msas.append(msa[:cutoff])

        # add focus
        focus_take = [item for item in focus if item in selection]
        term_focuses += focus_take
        # add contact idx
        contact_idx = term_data['contact_idx']
        contact_idx_take = [contact_idx[i] for i in range(len(contact_idx)) if focus[i] in selection]
        term_contact_idxs += contact_idx_take
        # append term len, the len of the focus
        term_lens.append(len(focus_take))

        # process ppoe
        ppoe = term_data['ppoe']
        term_len = ppoe.shape[2]
        num_alignments = ppoe.shape[0]
        # project to sin, cos 
        ppo_rads = ppoe[:, :3]/180*np.pi
        sin_ppo = np.sin(ppo_rads)
        cos_ppo = np.cos(ppo_rads)
        env = ppoe[:, 3:]

        # apply take
        ppoe = np.take(ppoe, take, axis=-1)

        # place rmsds into np array
        rmsd = np.expand_dims(term_data['rmsds'], 1) #term_data['rmsds'][:cutoff]
        rmsd_arr = np.concatenate([rmsd for _ in range(term_len)], axis=1)
        rmsd_arr = np.expand_dims(rmsd_arr, 1)
        term_len_arr = np.zeros((cutoff, 1, term_len))
        term_len_arr += term_len
        num_alignments_arr = np.zeros((cutoff, 1, term_len))
        num_alignments_arr += num_alignments

        # select features, cutoff at top N
        selected_features = [sin_ppo[:cutoff], cos_ppo[:cutoff], env[:cutoff], rmsd_arr[:cutoff], term_len_arr]

        features = np.concatenate(selected_features, axis=1)

        # pytorch does row vector computation
        # swap rows and columns
        features = features.transpose(0, 2, 1)
        term_features.append(features)

        if not stats:
            continue

        # compute statistics for features over all matches (vs top N cutoff)
        # one-hot encode msa
        ident = np.eye(NUM_AA)
        one_hot_msa = ident[msa]

        rmsd = np.expand_dims(term_data['rmsds'], (1,2))
        if weight_fn == "neg":
            weights = softmax(-np.sqrt(rmsd))
        elif weight_fn == "inv":
            eps = 1e-8
            weights = softmax(1/(rmsd + eps)) # eps included for safety but in theory isn't necessary
        else:
            raise Exception("weight fn must be either neg or inv")
        weighted_aa_freq = np.sum(one_hot_msa * weights, axis = 0)
        weighted_sin_ppo = np.sum(sin_ppo * weights, axis = 0)
        weighted_cos_ppo = np.sum(cos_ppo * weights, axis = 0)
        weighted_env = np.sum(env * weights, axis = 0)
        var_sin_ppo = np.sum( np.square(sin_ppo - weighted_sin_ppo), axis=0)
        var_cos_ppo = np.sum( np.square(cos_ppo - weighted_cos_ppo), axis=0)
        var_env = np.sum( np.square(env - weighted_env), axis=0)

        sing_stats = np.concatenate([weighted_aa_freq.transpose(),
                                     weighted_sin_ppo,
                                     weighted_cos_ppo,
                                     weighted_env,
                                     var_sin_ppo,
                                     var_cos_ppo,
                                     var_env], axis=0)

        term_sing_stats.append(sing_stats.transpose())

        #print("weighted_aa_freq", weighted_aa_freq)
        #print("raw aa freq", np.mean(one_hot_msa, axis=0))
        #print("top cutoff aa freq", np.mean(one_hot_msa[:cutoff], axis=0))

        pre_cov_features = np.concatenate([one_hot_msa.transpose(0,2,1), sin_ppo, cos_ppo, env], axis=1)
        pre_cov_features = pre_cov_features.transpose(0,2,1)
        mu_x = np.sum(pre_cov_features * weights, axis = 0)
        X = np.expand_dims(pre_cov_features - mu_x, -1).transpose(1,3,2,0)
        X_T = X.transpose(1,0,3,2)
        weighted_cov = X @ X_T
        term_pair_stats.append(np.expand_dims(weighted_cov,0))

    msa_tensor = np.concatenate(term_msas, axis = -1)
    features_tensor = np.concatenate(term_features, axis = 1)
    len_tensor = np.array(term_lens)
    term_focuses = np.array(term_focuses)
    term_contact_idxs = np.array(term_contact_idxs)

    # package cov matrices into one tensor
    max_term_len = max(term_lens)
    num_terms = len(term_lens)
    num_cov_features = term_pair_stats[0].shape[-1]

    if stats:
        sing_stats_tensor = np.concatenate(term_sing_stats, axis = 0)
        pair_stats_tensor = np.zeros([num_terms, 
                                      max_term_len, 
                                      max_term_len, 
                                      num_cov_features, 
                                      num_cov_features])
                                      
        for idx, cov_mat in enumerate(term_pair_stats):
            term_len = cov_mat.shape[1]
            pair_stats_tensor[idx, :term_len, :term_len] = cov_mat
    else:
        sing_stats_tensor = None
        pair_stats_tensor = None

    # check that sum of term lens is as long as the feature tensor
    assert sum(len_tensor) == features_tensor.shape[1]

    # manipulate coords to right shape
    pdb = in_path.split('/')[-1]

    coords_tensor = None
    if len(coords) == 1:
        chain = next(iter(coords.keys()))
        coords_tensor = coords[chain]
    else:
        chains = sorted(coords.keys())
        coords_tensor = np.vstack([coords[c] for c in chains])

    assert coords_tensor.shape[0] == len(data['sequence']), "num aa coords != seq length"

    # embed target ppoe
    ppoe = data['ppoe']
    ppo = ppoe[:, :3]
    env = ppoe[:, 3:]
    embedded_ppoe = np.concatenate([np.sin(ppo), np.cos(ppo), env], axis = 1)

    output = {
        'pdb': pdb,
        'coords': coords_tensor,
        'ppoe': embedded_ppoe,
        'features': features_tensor,
        'sing_stats': sing_stats_tensor,
        'pair_stats': pair_stats_tensor,
        'msas': msa_tensor,
        'focuses': term_focuses,
        'contact_idxs': term_contact_idxs,
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
            fp.write(str(len(term_focuses)) + '\n')
            fp.write(str(len(data['selection'])))

    print('Done with', pdb)

    return output


