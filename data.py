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
            features, msas, focuses, seq_lens, src_masks = [], [], [], [], []
            selfEs = []
            for data in batch:
                # have to transpose these two because then we can use pad_sequence for padding
                features.append(convert(data['features']).transpose(0,1))
                msas.append(convert(data['msas']).transpose(0,1))
                selfEs.append(convert(data['selfE']).transpose(0,1))

                focuses.append(convert(data['focuses']))
                seq_lens.append(data['seq_len'])
                src_masks.append(data['mask'])


            # transpose back after padding
            features = pad_sequence(features, batch_first=True).transpose(1,2).double()
            msas = pad_sequence(msas, batch_first=True).transpose(1,2).long()
            selfEs = pad_sequence(selfEs, batch_first=True).transpose(1,2)

            focuses = pad_sequence(focuses, batch_first=True)
            src_key_mask = pad_sequence([torch.zeros(l) for l in focus_lens], batch_first=True, padding_value=1).byte()

            max_focus_len = max(focus_lens)
            padded_src_masks = []
            for mask in src_masks:
                current_dim = mask.shape[0]
                diff = max_focus_len - current_dim
                padded_mask = block_diag(mask, np.zeros((diff, diff)))
                padded_src_masks.append(convert(padded_mask))

            # invert at this stage, since masks are stored inverted
            padded_src_masks = ~(torch.stack(padded_src_masks).byte())

            self.data_clusters.append([msas, features, seq_lens, focuses, padded_src_masks, src_key_mask, selfEs])

    def __len__(self):
        return len(self.data_clusters)

    def __iter__(self):
        #np.random.shuffle(self.data_clusters)
        for batch in self.data_clusters:
            yield batch
