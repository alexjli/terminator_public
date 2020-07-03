import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import pickle
from scipy.linalg import block_diag
import glob

def convert(tensor):
    return torch.from_numpy(tensor)

class Dataset():
    def __init__(self, in_folder, pdb_ids = None):
        self.dataset = []
        if pdb_ids:
            for id in pdb_ids:
                folder = id[1:3]
                with open('{}/{}/{}.features'.format(in_folder, folder, id), 'rb') as fp:
                    data = pickle.load(fp)
                    self.dataset.append(data)
        else:
            for filename in glob.glob('{}/*/*.features'.format(in_folder)):
                with open(filename, 'rb') as fp:
                    data = pickle.load(fp)
                    self.dataset.append(data)

    def dumpDataset(self, name='dataset'):
        with open(name + '.features', 'wb') as fp:
            pickle.dump(self.dataset, fp)
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class TERMDataLoader():
    def __init__(self, dataset, batch_size=4, shuffle=True,
                 collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['focuses']) for i in range(self.size)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        sorted_idx = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for count, idx in enumerate(sorted_idx):
            if count != 0 and count % self.batch_size == 0:
                clusters.append(batch)
                batch = [idx]
            else:
                batch.append(idx)
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters
        print(self.clusters)

        self.data_clusters = []

        # wrap up all the tensors with proper padding and masks
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            focus_lens = [self.lengths[i] for i in b_idx]
            features, msas, focuses, seq_lens, coords = [], [], [], [], []
            selfEs = []
            etabs = []
            term_lens = []

            for data in batch:
                # have to transpose these two because then we can use pad_sequence for padding
                features.append(convert(data['features']).transpose(0,1))
                msas.append(convert(data['msas']).transpose(0,1))

                selfEs.append(convert(data['selfE']))
                focuses.append(convert(data['focuses']))
                seq_lens.append(data['seq_len'])
                term_lens.append(data['term_lens'].tolist())
                coords.append(data['chain_dict'])
                etabs.append(data['etab'])

            # transpose back after padding
            features = pad_sequence(features, batch_first=True).transpose(1,2)
            msas = pad_sequence(msas, batch_first=True).transpose(1,2).long()
            selfEs = pad_sequence(selfEs, batch_first=True)

            focuses = pad_sequence(focuses, batch_first=True)
            src_key_mask = pad_sequence([torch.zeros(l) for l in focus_lens], batch_first=True, padding_value=1).bool()

            # we do some padding so that tensor reshaping during batchifyTERM works
            max_aa = focuses.size(-1)
            for lens in term_lens:
                max_term_len = max(lens)
                diff = max_aa - sum(lens)
                lens += [max_term_len] * (diff // max_term_len)
                lens.append(diff % max_term_len)

            X, x_mask, _ = self._featurize(coords, 'cpu')

            # pack all the etabs into one big sparse tensor
            idxs = []
            vals = []
            for batch, e in enumerate(etabs):
                for ix, nrgs in e.items():
                    idxs.append(torch.tensor([batch] + list(ix)))
                    vals.append(convert(nrgs))

            idx_t = torch.stack(idxs).transpose(0,1).long()
            val_t = torch.stack(vals)
            etab = torch.sparse.FloatTensor(idx_t, val_t)

            self.data_clusters.append([msas, features, seq_lens, focuses,
                                       src_key_mask, selfEs, term_lens, X, x_mask, etab])

    def __len__(self):
        return len(self.data_clusters)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.data_clusters)
        for batch in self.data_clusters:
            yield batch

    def _featurize(self, batch, device, shuffle_fraction=0.):
        """ Pack and pad batch into torch tensors """
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        B = len(batch)
        for b in batch:
            del_arr = []
            l = len(b['seq'])
            for ii in range(l):
                all_coords_arr = [b['coords'][c][ii] for c in ['N', 'CA', 'C', 'O']]
                all_coords_arr = np.stack(all_coords_arr)
                # get rid of atom if all coords are nan
                if np.isnan(all_coords_arr).all():
                    #print(ii)
                    del_arr.append(ii)
            for c in ['N', 'CA', 'C', 'O']:
                #print(b['coords'][c].shape)
                b['coords'][c] = np.delete(b['coords'][c], del_arr, 0)
                #print(b['coords'][c].shape, len(del_arr))
            #print()

        lengths = np.array([len(b['coords']['CA']) for b in batch], dtype=np.int32)
        #print(lengths)
        L_max = max(lengths)
        X = np.zeros([B, L_max, 4, 3])

        def shuffle_subset(n, p):
            n_shuffle = np.random.binomial(n, p)
            ix = np.arange(n)
            ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
            ix_subset_shuffled = np.copy(ix_subset)
            np.random.shuffle(ix_subset_shuffled)
            ix[ix_subset] = ix_subset_shuffled
            return ix

        # Build the batch
        for i, b in enumerate(batch):
            x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)

            l = x.shape[0]
            x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
            X[i,:,:,:] = x_pad

        # Mask
        isnan = np.isnan(X)
        mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
        X[isnan] = 0.

        # Conversion
        X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
        mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
        return X, mask, lengths
