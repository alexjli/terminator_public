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
from tqdm import tqdm 
import multiprocessing as mp

def convert(tensor):
    return torch.from_numpy(tensor)

def load_file(in_folder, id, min_protein_len = 30):
    path = f"{in_folder}/{id}/{id}.features"
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
        seq_len = data['seq_len']
        total_term_length = data['term_lens'].sum()
        if seq_len < min_protein_len:
            return None
    return data, total_term_length, seq_len


class TERMDataset():
    def __init__(self, in_folder, pdb_ids = None, min_protein_len = 30, num_processes = 32):
        self.dataset = []

        pool = mp.Pool(num_processes)

        if pdb_ids:
            print("Loading feature files")
            progress = tqdm(total = len(pdb_ids))
            def update_progress(res):
                progress.update(1)

            res_list = [pool.apply_async(load_file, 
                                        (in_folder, id),
                                        kwds = {"min_protein_len": min_protein_len},
                                        callback=update_progress) 
                        for id in pdb_ids]
            pool.close()
            pool.join()
            progress.close()
            for res in res_list:
                data = res.get()
                if data is not None:
                    features, total_term_length, seq_len = data
                    self.dataset.append((features, total_term_length, seq_len))
        else:
            print("Loading feature file paths")

            filelist = list(glob.glob('{}/*/*.features'.format(in_folder)))
            progress = tqdm(total = len(filelist))
            def update_progress(res):
                progress.update(1)
            
            # get pdb_ids
            pdb_ids = [os.path.basename(path).split(".")[0] for path in filelist]

            res_list = [pool.apply_async(load_file, 
                                        (in_folder, id),
                                        kwds = {"min_protein_len": min_protein_len},
                                        callback=update_progress) 
                        for id in pdb_ids]
            pool.close()
            pool.join()
            progress.close()
            for res in res_list:
                data = res.get()
                if data is not None:
                    features, total_term_length, seq_len = data
                    self.dataset.append((features, total_term_length, seq_len))

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


class TERMDataLoader(Sampler):
    def __init__(self, dataset, batch_size=4, sort_data = False, shuffle = True, semi_shuffle = False, semi_shuffle_cluster_size = 500, batch_shuffle = True, drop_last = False, max_term_res = 55000, max_seq_tokens = 0):
        #self.dataset = dataset
        self.size = len(dataset)
        self.dataset, self.total_term_lengths, self.seq_lengths = zip(*dataset)
        if max_term_res > 0 and max_seq_tokens == 0:
            self.lengths = self.total_term_lengths
        elif max_term_res == 0 and max_seq_tokens > 0:
            self.lengths = self.seq_lengths
        else:
            raise Exception("One and only one of max_term_res and max_seq_tokens must be 0")
        self.shuffle = shuffle
        self.sort_data = sort_data
        self.batch_shuffle = batch_shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.max_term_res = max_term_res
        self.max_seq_tokens = max_seq_tokens
        self.semi_shuffle = semi_shuffle
        self.semi_shuffle_cluster_size = semi_shuffle_cluster_size

        assert not (shuffle and semi_shuffle), "Lazy Dataloader shuffle and semi shuffle cannot both be set"
        #assert not (batch_size is None and (max_term_res <= 0)), "max_term_res>0 required when using variable size batches"
        # an assert for myself but i'm not sure if this is provably true
        #assert semi_shuffle_cluster_size % batch_size != 0, "having cluster size a multiple of batch size will lead to data shuffles that are worse for training"

        # initialize clusters
        self._cluster()

    def _cluster(self):
        """ Shuffle data and make new clusters """

        # if we sort data, use sorted indexes instead
        if self.sort_data:
            idx_list = np.argsort(self.lengths)
        elif self.semi_shuffle:
            # trying to speed up training
            # by shuffling points with similar term res together
            idx_list = np.argsort(self.lengths)
            shuffle_borders = []
            
            # break up datapoints into large clusters
            border = 0
            while border < len(self.lengths):
                shuffle_borders.append(border)
                border += self.semi_shuffle_cluster_size

            # shuffle datapoints within clusters
            last_cluster_idx = len(shuffle_borders)-1
            for cluster_idx in range(last_cluster_idx + 1):
                start = shuffle_borders[cluster_idx]
                if cluster_idx < last_cluster_idx:
                    end = shuffle_borders[cluster_idx + 1]
                    np.random.shuffle(idx_list[start:end])
                else:
                    np.random.shuffle(idx_list[start:])

        else:
            idx_list = list(range(len(self.dataset)))
            np.random.shuffle(idx_list)

        # Cluster into batches of similar sizes
        clusters, batch = [], []

        # if batch_size is None, fit as many proteins we can into a batch
        # without overloading the GPU
        if self.batch_size is None:
            if self.max_term_res > 0 and self.max_seq_tokens == 0:
                cap_len = self.max_term_res
            elif self.max_term_res == 0 and self.max_seq_tokens > 0:
                cap_len = self.max_seq_tokens
               
            current_batch_lens = []
            total_data_len = 0
            for count, idx in enumerate(idx_list):
                current_batch_lens.append(self.lengths[idx])
                total_data_len = max(current_batch_lens) * len(current_batch_lens)
                if count != 0 and total_data_len > cap_len:
                    clusters.append(batch)
                    batch = [idx]
                    current_batch_lens = [self.lengths[idx]]
                else:
                    batch.append(idx)

        else: # used fixed batch size
            for count, idx in enumerate(idx_list):
                if count != 0 and count % self.batch_size == 0:
                    clusters.append(batch)
                    batch = [idx]
                else:
                    batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
            clusters.append(batch)
        self.clusters = clusters
 

    def _package(self, b_idx):
        # wrap up all the tensors with proper padding and masks

        batch = [data[0] for data in b_idx]
        
        focus_lens = [data[1] for data in b_idx]
        features, msas, focuses, seq_lens, coords = [], [], [], [], []
        term_lens = []
        seqs = []
        ids = []
        chain_lens = []
        ppoe = []
        contact_idxs = []
        #sing_stats = [None]
        #pair_stats = [None]

        for idx, data in enumerate(batch):
            # have to transpose these two because then we can use pad_sequence for padding
            features.append(convert(data['features']).transpose(0,1))
            msas.append(convert(data['msas']).transpose(0,1))

            ppoe.append(convert(data['ppoe']))
            focuses.append(convert(data['focuses']))
            contact_idxs.append(convert(data['contact_idxs']))
            seq_lens.append(data['seq_len'])
            term_lens.append(data['term_lens'].tolist())
            coords.append(data['coords'])
            seqs.append(convert(data['sequence']))
            ids.append(data['pdb'])
            chain_lens.append(data['chain_lens'])
            #sing_stats.append(convert(data['sing_stats']))
            #pair_stats.append(convert(data['pair_stats']))

        # transpose back after padding
        features = pad_sequence(features, batch_first=True).transpose(1,2)
        msas = pad_sequence(msas, batch_first=True).transpose(1,2).long()

        # we can pad these using standard pad_sequence
        ppoe = pad_sequence(ppoe, batch_first=True)
        focuses = pad_sequence(focuses, batch_first=True)
        contact_idxs = pad_sequence(contact_idxs, batch_first=True)
        src_key_mask = pad_sequence([torch.zeros(l) for l in focus_lens], batch_first=True, padding_value=1).bool()
        seqs = pad_sequence(seqs, batch_first = True)

        # we do some padding so that tensor reshaping during batchifyTERM works
        max_aa = focuses.size(-1)
        for lens in term_lens:
            max_term_len = max(lens)
            diff = max_aa - sum(lens)
            lens += [max_term_len] * (diff // max_term_len)
            lens.append(diff % max_term_len)

        # featurize coordinates same way as ingraham et al
        X, x_mask, _ = self._featurize(coords, 'cpu')

        # pad with -1 so we can store term_lens in a tensor
        seq_lens = torch.tensor(seq_lens)
        max_all_term_lens = max([len(term) for term in term_lens])
        for i in range(len(term_lens)):
            term_lens[i] += [-1] * (max_all_term_lens - len(term_lens[i]))
        term_lens = torch.tensor(term_lens)

        # generate chain_idx from chain_lens
        chain_idx = []
        for c_lens in chain_lens:
            arrs = []
            for i in range(len(c_lens)):
                l = c_lens[i]
                arrs.append(torch.ones(l)*i)
            chain_idx.append(torch.cat(arrs, dim = -1))
        chain_idx = pad_sequence(chain_idx, batch_first = True)

        return {'msas':msas, 
                'features':features.float(),
                #'sing_stats':sing_stats,
                #'pair_stats':pair_stats,
                'ppoe': ppoe.float(),
                'seq_lens':seq_lens, 
                'focuses':focuses,
                'contact_idxs':contact_idxs,
                'src_key_mask':src_key_mask, 
                'term_lens':term_lens, 
                'X':X, 
                'x_mask':x_mask, 
                'seqs':seqs, 
                'ids':ids,
                'chain_idx':chain_idx}

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        if self.shuffle or self.semi_shuffle:
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


# needs to be outside of object for pickling reasons (?)
def read_lens(in_folder, id, min_protein_len = 30):
    path = f"{in_folder}/{id}/{id}.length"
    with open(path) as fp:
        total_term_length = int(fp.readline().strip())
        seq_len = int(fp.readline().strip())
        if seq_len < min_protein_len:
            return None
    return id, total_term_length, seq_len


class LazyDataset(Dataset):
    def __init__(self, in_folder, pdb_ids = None, min_protein_len = 30, num_processes = 32):
        self.dataset = []

        pool = mp.Pool(num_processes)

        if pdb_ids:
            print("Loading feature file paths")
            progress = tqdm(total = len(pdb_ids))
            def update_progress(res):
                progress.update(1)

            res_list = [pool.apply_async(read_lens, 
                                        (in_folder, id),
                                        kwds = {"min_protein_len": min_protein_len},
                                        callback=update_progress) 
                        for id in pdb_ids]
            pool.close()
            pool.join()
            progress.close()
            for res in res_list:
                data = res.get()
                if data is not None:
                    id, total_term_length, seq_len = data
                    filename = f"{in_folder}/{id}/{id}.features"
                    self.dataset.append((os.path.abspath(filename), total_term_length, seq_len))
        else:
            print("Loading feature file paths")

            filelist = list(glob.glob('{}/*/*.features'.format(in_folder)))
            progress = tqdm(total = len(filelist))
            def update_progress(res):
                progress.update(1)
            
            # get pdb_ids
            pdb_ids = [os.path.basename(path).split(".")[0] for path in filelist]

            res_list = [pool.apply_async(read_lens, 
                                        (in_folder, id),
                                        kwds = {"min_protein_len": min_protein_len},
                                        callback=update_progress) 
                        for id in pdb_ids]
            pool.close()
            pool.join()
            progress.close()
            for res in res_list:
                data = res.get()
                if data is not None:
                    id, total_term_length, seq_len = data
                    filename = f"{in_folder}/{id}/{id}.features"
                    self.dataset.append((os.path.abspath(filename), total_term_length, seq_len))

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
    def __init__(self, 
                 dataset, 
                 batch_size=4, 
                 sort_data = False, 
                 shuffle = True, 
                 semi_shuffle = False, 
                 semi_shuffle_cluster_size = 500, 
                 batch_shuffle = True, 
                 drop_last = False, 
                 max_term_res = 55000, 
                 max_seq_tokens = 0,
                 term_matches_cutoff = None):

        self.dataset = dataset
        self.size = len(dataset)
        self.filepaths, self.total_term_lengths, self.seq_lengths = zip(*dataset)
        if max_term_res > 0 and max_seq_tokens == 0:
            self.lengths = self.total_term_lengths
        elif max_term_res == 0 and max_seq_tokens > 0:
            self.lengths = self.seq_lengths
        else:
            raise Exception("One and only one of max_term_res and max_seq_tokens must be 0")
        self.shuffle = shuffle
        self.sort_data = sort_data
        self.batch_shuffle = batch_shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.max_term_res = max_term_res
        self.max_seq_tokens = max_seq_tokens
        self.semi_shuffle = semi_shuffle
        self.semi_shuffle_cluster_size = semi_shuffle_cluster_size
        self.term_matches_cutoff = term_matches_cutoff

        assert not (shuffle and semi_shuffle), "Lazy Dataloader shuffle and semi shuffle cannot both be set"
        #assert not (batch_size is None and (max_term_res <= 0)), "max_term_res>0 required when using variable size batches"
        # an assert for myself but i'm not sure if this is provably true
        #assert semi_shuffle_cluster_size % batch_size != 0, "having cluster size a multiple of batch size will lead to data shuffles that are worse for training"

        # initialize clusters
        self._cluster()

    def _cluster(self):
        """ Shuffle data and make new clusters """

        # if we sort data, use sorted indexes instead
        if self.sort_data:
            idx_list = np.argsort(self.lengths)
        elif self.semi_shuffle:
            # trying to speed up training
            # by shuffling points with similar term res together
            idx_list = np.argsort(self.lengths)
            shuffle_borders = []
            
            # break up datapoints into large clusters
            border = 0
            while border < len(self.lengths):
                shuffle_borders.append(border)
                border += self.semi_shuffle_cluster_size

            # shuffle datapoints within clusters
            last_cluster_idx = len(shuffle_borders)-1
            for cluster_idx in range(last_cluster_idx + 1):
                start = shuffle_borders[cluster_idx]
                if cluster_idx < last_cluster_idx:
                    end = shuffle_borders[cluster_idx + 1]
                    np.random.shuffle(idx_list[start:end])
                else:
                    np.random.shuffle(idx_list[start:])

        else:
            idx_list = list(range(len(self.dataset)))
            np.random.shuffle(idx_list)

        # Cluster into batches of similar sizes
        clusters, batch = [], []

        # if batch_size is None, fit as many proteins we can into a batch
        # without overloading the GPU
        if self.batch_size is None:
            if self.max_term_res > 0 and self.max_seq_tokens == 0:
                cap_len = self.max_term_res
            elif self.max_term_res == 0 and self.max_seq_tokens > 0:
                cap_len = self.max_seq_tokens
               
            current_batch_lens = []
            total_data_len = 0
            for count, idx in enumerate(idx_list):
                current_batch_lens.append(self.lengths[idx])
                total_data_len = max(current_batch_lens) * len(current_batch_lens)
                if count != 0 and total_data_len > cap_len:
                    clusters.append(batch)
                    batch = [idx]
                    current_batch_lens = [self.lengths[idx]]
                else:
                    batch.append(idx)

        else: # used fixed batch size
            for count, idx in enumerate(idx_list):
                if count != 0 and count % self.batch_size == 0:
                    clusters.append(batch)
                    batch = [idx]
                else:
                    batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
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
                if 'ppoe' not in batch[-1].keys():
                    print(filepath)

        focus_lens = [data[1] for data in b_idx]
        features, msas, focuses, seq_lens, coords = [], [], [], [], []
        term_lens = []
        seqs = []
        ids = []
        chain_lens = []
        ppoe = []
        contact_idxs = []
        #sing_stats = [None]
        #pair_stats = [None]

        for idx, data in enumerate(batch):
            # have to transpose these two because then we can use pad_sequence for padding
            features.append(convert(data['features']).transpose(0,1))
            msas.append(convert(data['msas']).transpose(0,1))

            ppoe.append(convert(data['ppoe']))
            focuses.append(convert(data['focuses']))
            contact_idxs.append(convert(data['contact_idxs']))
            seq_lens.append(data['seq_len'])
            term_lens.append(data['term_lens'].tolist())
            coords.append(data['coords'])
            seqs.append(convert(data['sequence']))
            ids.append(data['pdb'])
            chain_lens.append(data['chain_lens'])
            #sing_stats.append(convert(data['sing_stats']))
            #pair_stats.append(convert(data['pair_stats']))

        """
        # detect if we have sing and pair stats
        if sing_stats[0] == None:
            sing_stats = None
        if pair_stats[0] == None:
            pair_stats == None
        """

        # transpose back after padding
        features = pad_sequence(features, batch_first=True).transpose(1,2)
        msas = pad_sequence(msas, batch_first=True).transpose(1,2).long()
        if self.term_matches_cutoff:
            features = features[:, :self.term_matches_cutoff]
            msas = msas[:, :self.term_matches_cutoff]

        # we can pad these using standard pad_sequence
        ppoe = pad_sequence(ppoe, batch_first=True)
        focuses = pad_sequence(focuses, batch_first=True)
        contact_idxs = pad_sequence(contact_idxs, batch_first=True)
        src_key_mask = pad_sequence([torch.zeros(l) for l in focus_lens], batch_first=True, padding_value=1).bool()
        seqs = pad_sequence(seqs, batch_first = True)

        """
        if sing_stats:
            sing_stats = pad_sequence(sing_stats, batch_first = True)
        """

        # we do some padding so that tensor reshaping during batchifyTERM works
        max_aa = focuses.size(-1)
        for lens in term_lens:
            max_term_len = max(lens)
            diff = max_aa - sum(lens)
            lens += [max_term_len] * (diff // max_term_len)
            lens.append(diff % max_term_len)

        # featurize coordinates same way as ingraham et al
        X, x_mask, _ = self._featurize(coords, 'cpu')

        # pad with -1 so we can store term_lens in a tensor
        seq_lens = torch.tensor(seq_lens)
        max_all_term_lens = max([len(term) for term in term_lens])
        for i in range(len(term_lens)):
            term_lens[i] += [-1] * (max_all_term_lens - len(term_lens[i]))
        term_lens = torch.tensor(term_lens)

        # generate chain_idx from chain_lens
        chain_idx = []
        for c_lens in chain_lens:
            arrs = []
            for i in range(len(c_lens)):
                l = c_lens[i]
                arrs.append(torch.ones(l)*i)
            chain_idx.append(torch.cat(arrs, dim = -1))
        chain_idx = pad_sequence(chain_idx, batch_first = True)

        """
        # process pair stats if present
        if pair_stats:
            # need to be a little fancier for pair_stats
            max_term_len = max(term_lens[0]) # technically not the best way but will still work
            num_features = pair_stats[0].shape[-1]
            num_batch = len(pair_stats)
            pair_stats_padded = torch.zeros([num_batch,
                                             max_all_term_lens,
                                             max_term_len,
                                             max_term_len,
                                             num_features,
                                             num_features])
            for idx, cov_mat in enumerate(pair_stats):
                num_terms = cov_mat.shape[0]
                pair_stats_padded[idx, 
                                 :num_terms, 
                                 :max_term_len, 
                                 :max_term_len,
                                 :num_features,
                                 :num_features] = cov_mat
            pair_stats = pair_stats_padded
        """

        return {'msas':msas, 
                'features':features.float(),
                #'sing_stats':sing_stats,
                #'pair_stats':pair_stats,
                'ppoe': ppoe.float(),
                'seq_lens':seq_lens, 
                'focuses':focuses,
                'contact_idxs':contact_idxs,
                'src_key_mask':src_key_mask, 
                'term_lens':term_lens, 
                'X':X, 
                'x_mask':x_mask, 
                'seqs':seqs, 
                'ids':ids,
                'chain_idx':chain_idx}

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        if self.shuffle or self.semi_shuffle:
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

