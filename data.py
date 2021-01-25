import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import pickle
from scipy.linalg import block_diag
import glob
import random
import os

def convert(tensor):
    return torch.from_numpy(tensor)

class TERMDataset():
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

        self.shuffle_idx = np.arange(len(self.dataset))

    def shuffle(self):
        np.random.shuffle(self.shuffle_idx)
        #random.shuffle(self.dataset)

    def dumpDataset(self, name='dataset'):
        with open(name + '.features', 'wb') as fp:
            pickle.dump(self.dataset, fp)
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_idx = self.shuffle_idx[idx]
        return [self.dataset[i] for i in data_idx]
        #return self.dataset[idx]

class TERMDataLoader():
    def __init__(self, dataset, batch_size=4, shuffle=True,
                 collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['focuses']) for i in range(self.size)]
        self.shuffle = shuffle
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
        # print(self.clusters)

        self.data_clusters = []

        # wrap up all the tensors with proper padding and masks
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            focus_lens = [self.lengths[i] for i in b_idx]
            features, msas, focuses, seq_lens, coords = [], [], [], [], []
            selfEs = []
            etabs = []
            term_lens = []
            seqs = []
            ids = []
            chain_lens = []

            for data in batch:
                # have to transpose these two because then we can use pad_sequence for padding
                features.append(convert(data['features']).transpose(0,1))
                msas.append(convert(data['msas']).transpose(0,1))

                #selfEs.append(convert(data['selfE']))
                focuses.append(convert(data['focuses']))
                seq_lens.append(data['seq_len'])
                term_lens.append(data['term_lens'].tolist())
                coords.append(data['coords'])
                #etabs.append(data['etab'])
                seqs.append(convert(data['sequence']))
                ids.append(data['pdb'])
                chain_lens.append(data['chain_lens'])

            # transpose back after padding
            features = pad_sequence(features, batch_first=True).transpose(1,2)
            msas = pad_sequence(msas, batch_first=True).transpose(1,2).long()

            #selfEs = pad_sequence(selfEs, batch_first=True)
            focuses = pad_sequence(focuses, batch_first=True)

            src_key_mask = pad_sequence([torch.zeros(l) for l in focus_lens], batch_first=True, padding_value=1).bool()
            seqs = pad_sequence(seqs, batch_first = True)

            # we do some padding so that tensor reshaping during batchifyTERM works
            max_aa = focuses.size(-1)
            for lens in term_lens:
                max_term_len = max(lens)
                diff = max_aa - sum(lens)
                lens += [max_term_len] * (diff // max_term_len)
                lens.append(diff % max_term_len)

            X, x_mask, _ = self._featurize(coords, 'cpu')

            """
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
            """

            self.data_clusters.append({'msas':msas, 
                                       'features':features, 
                                       'seq_lens':seq_lens, 
                                       'focuses':focuses,
                                       'src_key_mask':src_key_mask, 
                                       #'selfEs':selfEs, 
                                       'term_lens':term_lens, 
                                       'X':X, 
                                       'x_mask':x_mask, 
                                       'seqs':seqs, 
                                       'ids':ids,
                                       'chain_lens':chain_lens})

    def __len__(self):
        return len(self.data_clusters)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.data_clusters)
        for batch in self.data_clusters:
            yield batch

    def _featurize(self, batch, device, shuffle_fraction=0.):
        """ Pack and pad batch into torch tensors """
        B = len(batch)
        lengths = np.array([b.shape[0] for b in batch], dtype=np.int32)
        L_max = max(lengths)
        X = np.zeros([B, L_max, 4, 3])


        # Build the batch
        for i, x in enumerate(batch):

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

class LazyDataset(Dataset):
    def __init__(self, in_folder, pdb_ids = None, min_protein_len = 30):
        self.dataset = []
        if pdb_ids:
            for id in pdb_ids:
                filename = '{}/{}/{}.features'.format(in_folder, id, id)
                with open('{}/{}/{}.length'.format(in_folder, id, id)) as fp:
                    total_term_length = int(fp.readline().strip())
                    seq_len = int(fp.readline().strip())
                    if seq_len < min_protein_len:
                        continue
                self.dataset.append((os.path.abspath(filename), total_term_length))
        else:
            for filename in glob.glob('{}/*/*.features'.format(in_folder)):
                prefix = os.path.splitext(filename)[0]
                with open(prefix + '.length') as fp:
                    total_term_length = int(fp.readline().strip())
                    seq_len = int(fp.readline().strip())
                    if seq_len < min_protein_len:
                        continue
                self.dataset.append((os.path.abspath(filename), total_term_length))

        self.shuffle_idx = np.arange(len(self.dataset))

    def shuffle(self):
        np.random.shuffle(self.shuffle_idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_idx = self.shuffle_idx[idx]
        if isinstance(data_idx, list):
            return [self.dataset[i] for i in data_idx]
        else:
            return self.dataset[data_idx]

class TERMLazyDataLoader(Sampler):
    def __init__(self, dataset, batch_size=4, sort_data = False, shuffle = True, batch_shuffle = True, drop_last = False, max_term_res = 55000):
        self.dataset = dataset
        self.size = len(dataset)
        self.filepaths, self.lengths = zip(*dataset)
        self.shuffle = shuffle
        self.sort_data = sort_data
        self.batch_shuffle = batch_shuffle
        self.batch_size = batch_size
        self.drop_last = self.drop_last

        # initialize clusters
        self._cluster()

    def _cluster(self):
        """ Shuffle data and make new clusters """

        # if we sort data, use sorted indexes instead
        if self.sort_data:
            shuffle_idx = np.argsort(self.lengths)
        else:
            idx_list = list(range(len(self.dataset)))
            shuffle_idx = np.random.shuffle(idx_list)

        # Cluster into batches of similar sizes
        clusters, batch = [], []

        # if batch_size is None, fit as many proteins we can into a batch
        # without overloading the GPU
        if self.max_term_res > 0 and self.batch_size is None:
            current_batch_lens = []
            total_data_len = 0
            for count, idx in enumerate(shuffle_idx):
                current_batch_lens.append(self.lengths[idx])
                total_data_len = max(current_batch_lens) * len(current_batch_lens)
                if count != 0 and total_data_len > max_total_data_len:
                    clusters.append(batch)
                    batch = [idx]
                    current_batch_lens = [self.lengths[idx]]
                else:
                    batch.append(idx)

        else: # used fixed batch size
            for count, idx in enumerate(shuffle_idx):
                if count != 0 and count % self.batch_size == 0:
                    clusters.append(batch)
                    batch = [idx]
                else:
                    batch.append(idx)

        if len(batch) > 0 and not drop_last:
            clusters.append(batch)
        self.clusters = clusters
 

    def _package(self, b_idx):
        # wrap up all the tensors with proper padding and masks
        if self.batch_shuffle:
            b_idx_copy = b_idx[:]
            random.shuffle(b_idx_copy)
            b_idx = b_idx_copy

        batch = []
        for data in b_idx:
            filepath = data[0]
            with open(filepath, 'rb') as fp:
                batch.append(pickle.load(fp))

        focus_lens = [data[1] for data in b_idx]
        features, msas, focuses, seq_lens, coords = [], [], [], [], []
        selfEs = []
        etabs = []
        term_lens = []
        seqs = []
        ids = []
        chain_lens = []

        for data in batch:
            # have to transpose these two because then we can use pad_sequence for padding
            features.append(convert(data['features']).transpose(0,1))
            msas.append(convert(data['msas']).transpose(0,1))

            #selfEs.append(convert(data['selfE']))
            focuses.append(convert(data['focuses']))
            seq_lens.append(data['seq_len'])
            term_lens.append(data['term_lens'].tolist())
            coords.append(data['coords'])
            #etabs.append(data['etab'])
            seqs.append(convert(data['sequence']))
            ids.append(data['pdb'])
            chain_lens.append(data['chain_lens'])

        # transpose back after padding
        features = pad_sequence(features, batch_first=True).transpose(1,2)
        msas = pad_sequence(msas, batch_first=True).transpose(1,2).long()

        #selfEs = pad_sequence(selfEs, batch_first=True)
        focuses = pad_sequence(focuses, batch_first=True)

        src_key_mask = pad_sequence([torch.zeros(l) for l in focus_lens], batch_first=True, padding_value=1).bool()
        seqs = pad_sequence(seqs, batch_first = True)

        # we do some padding so that tensor reshaping during batchifyTERM works
        max_aa = focuses.size(-1)
        for lens in term_lens:
            max_term_len = max(lens)
            diff = max_aa - sum(lens)
            lens += [max_term_len] * (diff // max_term_len)
            lens.append(diff % max_term_len)

        X, x_mask, _ = self._featurize(coords, 'cpu')

        seq_lens = torch.tensor(seq_lens)
        max_all_term_lens = max([len(term) for term in term_lens])
        for i in range(len(term_lens)):
            term_lens[i] += [-1] * (max_all_term_lens - len(term_lens[i]))
        term_lens = torch.tensor(term_lens)

        return {'msas':msas, 
                'features':features, 
                'seq_lens':seq_lens, 
                'focuses':focuses,
                'src_key_mask':src_key_mask, 
                'term_lens':term_lens, 
                'X':X, 
                'x_mask':x_mask, 
                'seqs':seqs, 
                'ids':ids,
                'chain_lens':chain_lens}

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        if self.shuffle:
            self._cluster()
            np.random.shuffle(self.clusters)
        for batch in self.clusters:
            yield batch

    def _featurize(self, batch, device, shuffle_fraction=0.):
        """ Pack and pad batch into torch tensors """
        B = len(batch)
        lengths = np.array([b.shape[0] for b in batch], dtype=np.int32)
        L_max = max(lengths)
        X = np.zeros([B, L_max, 4, 3])

        # Build the batch
        for i, x in enumerate(batch):
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

class TERMLazyDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas, rank, shuffle = False, batch_shuffle = False, seed = 0, drop_last = False, batch_size = 4, max_total_data_lens = 55000):
        super.init(dataset, num_replicas, rank, shuffle = shuffle, seed = seed, drop_last = drop_last)
        self.sampler = TERMLazyDataLoader(dataset, batch_size = batch_size, shuffle = shuffle, batch_shuffle = shuffle, drop_last = drop_last, max_total_data_lens = max_total_data_lens)
        self.size = len(dataset)
        self.filepaths, self.lengths = zip(*dataset)
        self.batch_size = batch_size
        sorted_idx = np.argsort(self.lengths)

        indices = list(range(len(self.filepaths)))
        if not self.drop_last:
            indices += indices[:(self.total_size - len(indices))]
        else:
            indices = indices[:self.total_size]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.filepaths = self.filepaths[indices]
        self.lengths = self.lengths[indices]

        # Cluster into batches of similar sizes
        clusters, batch = [], []

        # if batch_size is None, fit as many proteins we can into a batch
        # without overloading the GPU
        if max_total_data_len > 0 and batch_size is None:
            current_batch_lens = []
            total_data_len = 0
            for count, idx in enumerate(sorted_idx):
                current_batch_lens.append(self.lengths[idx])
                total_data_len = max(current_batch_lens) * len(current_batch_lens)
                if count != 0 and total_data_len > max_total_data_len:
                    clusters.append(batch)
                    batch = [idx]
                    current_batch_lens = [self.lengths[idx]]
                else:
                    batch.append(idx)
                    #current_batch_lens.append(self.lengths[idx])

        else: # used fixed batch size
            for count, idx in enumerate(sorted_idx):
                if count != 0 and count % self.batch_size == 0:
                    clusters.append(batch)
                    batch = [idx]
                else:
                    batch.append(idx)

        if len(batch) > 0 and not drop_last:
            clusters.append(batch)
        self.clusters = clusters

    def _package(self, b_idx):
        self.sample._package(b_idx)

    def __iter__(self):
        if self.shuffle:
            rng = np.random.RandomState(seed = self.seed + self.epoch)
            rng.shuffle(self.clusters)
        for batch in self.clusters:
            yield batch

