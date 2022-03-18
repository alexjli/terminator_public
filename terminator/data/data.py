"""Datasets and dataloaders for loading TERMs.

This file contains dataset and dataloader classes
to be used when interacting with TERMs.
"""
import glob
import math
import multiprocessing as mp
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

# pylint: disable=no-member, not-callable

# Ingraham featurization functions


def _ingraham_featurize(batch, device="cpu"):
    """ Pack and pad coords in batch into torch tensors
    as done in https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    batch : list of dict
        list of protein backbone coordinate dictionaries,
        in the format of that outputted by :code:`parseCoords.py`
    device : str
        device to place torch tensors

    Returns
    -------
    X : torch.Tensor
        Batched coordinates tensor
    mask : torch.Tensor
        Mask for X
    lengths : np.ndarray
        Array of lengths of batched proteins
    """
    B = len(batch)
    lengths = np.array([b.shape[0] for b in batch], dtype=np.int32)
    l_max = max(lengths)
    X = np.zeros([B, l_max, 4, 3])

    # Build the batch
    for i, x in enumerate(batch):
        l = x.shape[0]
        x_pad = np.pad(x, [[0, l_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan, ))
        X[i, :, :, :] = x_pad

    # Mask
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    return X, mask, lengths



# Jing featurization functions



def _normalize(tensor, dim=-1):
    '''Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.'''
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.

    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].

    From https://github.com/jingraham/neurips19-graph-protein-design
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    rbf = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return rbf


def _dihedrals(X, eps=1e-7):
    """ Compute dihedral angles between residues given atomic backbone coordinates

    Args
    ----
    X : torch.FloatTensor
        Tensor specifying atomic backbone coordinates
        Shape: num_res x 4 x 3

    Returns
    -------
    D_features : torch.FloatTensor
        Dihedral angles, lifted to the 3-torus
        Shape: num_res x 7
    """
    # From https://github.com/jingraham/neurips19-graph-protein-design

    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _positional_embeddings(edge_index, num_embeddings=16, dev='cpu'):
    """ Sinusoidally encode sequence distances for edges.

    Args
    ----
    edge_index : torch.LongTensor
        Edge indices for sparse representation of protein graph
        Shape: 2 x num_edges
    num_embeddings : int or None, default=128
        Dimensionality of sinusoidal embedding.

    Returns
    -------
    E : torch.FloatTensor
        Sinusoidal encoding of sequence distances
        Shape: num_edges x num_embeddings

    """
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=dev) * -(np.log(10000.0) / num_embeddings))
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def _orientations(X_ca):
    """ Compute forward and backward vectors per residue.

    Args
    ----
    X_ca : torch.FloatTensor
        Tensor specifying atomic backbone coordinates for CA atoms.
        Shape: num_res x 3

    Returns
    -------
    torch.FloatTensor
        Pairs of forward, backward vectors per residue.
        Shape: num_res x 2 x 3
    """
    # From https://github.com/drorlab/gvp-pytorch
    forward = _normalize(X_ca[1:] - X_ca[:-1])
    backward = _normalize(X_ca[:-1] - X_ca[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    """ Compute vectors pointing in the approximate direction of the sidechain.

    Args
    ----
    X : torch.FloatTensor
        Tensor specifying atomic backbone coordinates.
        Shape: num_res x 4 x 3

    Returns
    -------
    vec : torch.FloatTensor
        Sidechain vectors.
        Shape: num_res x 3
    """
    # From https://github.com/drorlab/gvp-pytorch
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def _jing_featurize(protein, dev='cpu'):
    """ Featurize individual proteins for use in torch_geometric Data objects,
    as done in https://github.com/drorlab/gvp-pytorch

    Args
    ----
    protein : dict
        Dictionary of protein features

        - :code:`name` - PDB ID of the protein
        - :code:`coords` - list of dicts specifying backbone atom coordinates
        in the format of that outputted by :code:`parseCoords.py`
        - :code:`seq` - protein sequence
        - :code:`chain_idx` - an integer per residue such that each unique integer represents a unique chain

    Returns
    -------
    torch_geometric.data.Data
        Data object containing
        - :code:`x` - CA atomic coordinates
        - :code:`seq` - sequence of protein
        - :code:`name` - PDB ID of protein
        - :code:`node_s` - Node scalar features
        - :code:`node_v` - Node vector features
        - :code:`edge_s` - Edge scalar features
        - :code:`edge_v` - Edge vector features
        - :code:`edge_index` - Sparse representation of edge
        - :code:`mask` - Residue mask specifying residues with incomplete coordinate sets
    """
    name = protein['name']
    with torch.no_grad():
        coords = torch.as_tensor(protein['coords'], device=dev, dtype=torch.float32)
        seq = torch.as_tensor(protein['seq'], device=dev, dtype=torch.long)

        mask = torch.isfinite(coords.sum(dim=(1, 2)))
        coords[~mask] = np.inf

        X_ca = coords[:, 1]
        edge_index = torch_cluster.knn_graph(X_ca, k=30, loop=True)  # TODO: make param

        pos_embeddings = _positional_embeddings(edge_index)
        # generate mask for interchain interactions
        pos_chain = (protein['chain_idx'][edge_index.view(-1)]).view(2, -1)
        pos_mask = (pos_chain[0] != pos_chain[1])
        # zero out all interchain positional embeddings
        pos_embeddings = pos_mask.unsqueeze(-1) * pos_embeddings

        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=16, device=dev)  # TODO: make param

        dihedrals = _dihedrals(coords)
        orientations = _orientations(X_ca)
        sidechains = _sidechains(coords)

        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))

    data = torch_geometric.data.Data(x=X_ca,
                                     seq=seq,
                                     name=name,
                                     node_s=node_s,
                                     node_v=node_v,
                                     edge_s=edge_s,
                                     edge_v=edge_v,
                                     edge_index=edge_index,
                                     mask=mask)
    return data



# Batching functions


def convert(tensor):
    """Converts given tensor from numpy to pytorch."""
    return torch.from_numpy(tensor)


def _package(b_idx):
    """Package the given datapoints into tensors based on provided indices.

    Tensors are extracted from the data and padded. Coordinates are featurized
    and the length of TERMs and chain IDs are added to the data.

    Args
    ----
    b_idx : list of tuples (dicts, int)
        The feature dictionaries, as well as an int for the sum of the lengths of all TERMs,
        for each datapoint to package.

    Returns
    -------
    dict
        Collection of batched features required for running TERMinator. This contains:

        - :code:`msas` - the multiple sequence alignment

        - :code:`features` - the TERM features

        - :code:`ppoe` - ???

        - :code:`seq_lens` - the lengths of the sequences

        - :code:`focuses` - ???

        - :code:`contact_idxs` - contact indices

        - :code:`src_key_mask` - ???

        - :code:`X` - coordinates

        - :code:`x_mask` - ???

        - :code:`seqs` - the sequences

        - :code:`ids` - the PDB id??

        - :code:`chain_idx` - the chain IDs
    """
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
    gvp_data = []

    for _, data in enumerate(batch):
        # have to transpose these two because then we can use pad_sequence for padding
        features.append(convert(data['features']).transpose(0, 1))
        msas.append(convert(data['msas']).transpose(0, 1))

        ppoe.append(convert(data['ppoe']))
        focuses.append(convert(data['focuses']))
        contact_idxs.append(convert(data['contact_idxs']))
        seq_lens.append(data['seq_len'])
        term_lens.append(data['term_lens'].tolist())
        coords.append(data['coords'])
        seqs.append(convert(data['sequence']))
        ids.append(data['pdb'])
        chain_lens.append(data['chain_lens'])

        chain_idx = []
        for i, c_len in enumerate(data['chain_lens']):
            chain_idx.append(torch.ones(c_len) * i)
        chain_idx = torch.cat(chain_idx, dim=0)
        gvp_data.append(
            _jing_featurize({
                'name': data['pdb'],
                'coords': data['coords'],
                'seq': data['sequence'],
                'chain_idx': chain_idx
            }))

    # transpose back after padding
    features = pad_sequence(features, batch_first=True).transpose(1, 2)
    msas = pad_sequence(msas, batch_first=True).transpose(1, 2).long()

    # we can pad these using standard pad_sequence
    ppoe = pad_sequence(ppoe, batch_first=True)
    focuses = pad_sequence(focuses, batch_first=True)
    contact_idxs = pad_sequence(contact_idxs, batch_first=True)
    src_key_mask = pad_sequence([torch.zeros(l) for l in focus_lens], batch_first=True, padding_value=1).bool()
    seqs = pad_sequence(seqs, batch_first=True)

    # we do some padding so that tensor reshaping during batchifyTERM works
    # TODO(alex): explain this since I have no idea what's going on
    max_aa = focuses.size(-1)
    for lens in term_lens:
        max_term_len = max(lens)
        diff = max_aa - sum(lens)
        lens += [max_term_len] * (diff // max_term_len)
        lens.append(diff % max_term_len)

    # featurize coordinates same way as ingraham et al
    X, x_mask, _ = _ingraham_featurize(coords)

    # pad with -1 so we can store term_lens in a tensor
    seq_lens = torch.tensor(seq_lens)
    max_all_term_lens = max([len(term) for term in term_lens])
    for i, _ in enumerate(term_lens):
        term_lens[i] += [-1] * (max_all_term_lens - len(term_lens[i]))
    term_lens = torch.tensor(term_lens)

    # generate chain_idx from chain_lens
    chain_idx = []
    for c_lens in chain_lens:
        arrs = []
        for i, chain_len in enumerate(c_lens):
            arrs.append(torch.ones(chain_len) * i)
        chain_idx.append(torch.cat(arrs, dim=-1))
    chain_idx = pad_sequence(chain_idx, batch_first=True)

    return {
        'msas': msas,
        'features': features.float(),
        'ppoe': ppoe.float(),
        'seq_lens': seq_lens,
        'focuses': focuses,
        'contact_idxs': contact_idxs,
        'src_key_mask': src_key_mask,
        'term_lens': term_lens,
        'X': X,
        'x_mask': x_mask,
        'seqs': seqs,
        'ids': ids,
        'chain_idx': chain_idx,
        'gvp_data': gvp_data
    }


# Non-lazy data loading functions



def load_file(in_folder, pdb_id, min_protein_len=30):
    """Load the data specified in the proper .features file and return them.
    If the read sequence length is less than :code:`min_protein_len`, instead return None.

    Args
    ----
    in_folder : str
        folder to find TERM file.
    pdb_id : str
        PDB ID to load.
    min_protein_len : int
        minimum cutoff for loading TERM file.

    Returns
    -------
    data : dict
        Data from TERM file (as dict)
    total_term_len : int
        Sum of lengths of all TERMs
    seq_len : int
        Length of protein sequence
    """
    path = f"{in_folder}/{pdb_id}/{pdb_id}.features"
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
        seq_len = data['seq_len']
        total_term_length = data['term_lens'].sum()
        if seq_len < min_protein_len:
            return None
    return data, total_term_length, seq_len


class TERMDataset(Dataset):
    """TERM Dataset that loads all feature files into a Pytorch Dataset-like structure.

    Attributes
    ----
    dataset : list
        list of tuples containing features, TERM length, and sequence length
    shuffle_idx : list
        array of indices for the dataset, for shuffling
    """
    def __init__(self, in_folder, pdb_ids=None, min_protein_len=30, num_processes=32):
        """
        Initializes current TERM dataset by reading in feature files.

        Reads in all feature files from the given directory, using multiprocessing
        with the provided number of processes. Stores the features, the TERM length,
        and the sequence length as a tuple representing the data. Can read from PDB ids or
        file paths directly. Uses the given protein length as a cutoff.

        Args
        ----
        in_folder : str
            path to directory containing feature files generated by :code:`scripts/data/preprocessing/generateDataset.py`
        pdb_ids: list, optional
            list of pdbs from `in_folder` to include in the dataset
        min_protein_len: int, default=30
            minimum length of a protein in the dataset
        num_processes: int, default=32
            number of processes to use during dataloading
        """
        self.dataset = []

        with mp.Pool(num_processes) as pool:

            if pdb_ids:
                print("Loading feature files")
                progress = tqdm(total=len(pdb_ids))

                def update_progress(res):
                    del res
                    progress.update(1)

                res_list = [
                    pool.apply_async(load_file, (in_folder, id),
                                     kwds={"min_protein_len": min_protein_len},
                                     callback=update_progress) for id in pdb_ids
                ]
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

                filelist = list(glob.glob(f'{in_folder}/*/*.features'))
                progress = tqdm(total=len(filelist))

                def update_progress(res):
                    del res
                    progress.update(1)

                # get pdb_ids
                pdb_ids = [os.path.basename(path).split(".")[0] for path in filelist]

                res_list = [
                    pool.apply_async(load_file, (in_folder, id),
                                     kwds={"min_protein_len": min_protein_len},
                                     callback=update_progress) for id in pdb_ids
                ]
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
        """Shuffle the current dataset."""
        np.random.shuffle(self.shuffle_idx)

    def __len__(self):
        """Returns length of the given dataset.

        Returns
        -------
        int
            length of dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Extract a given item with provided index.

        Args
        ----
        idx : int
            Index of item to return.
        Returns
        ----
        data : dict
            Data from TERM file (as dict)
        total_term_len : int
            Sum of lengths of all TERMs
        seq_len : int
            Length of protein sequence
        """
        data_idx = self.shuffle_idx[idx]
        if isinstance(data_idx, list):
            return [self.dataset[i] for i in data_idx]
        return self.dataset[data_idx]


class TERMBatchSampler(Sampler):
    """BatchSampler/Dataloader helper class for TERM data using TERMDataset.

    Attributes
    ----
    size: int
        Length of the dataset
    dataset: List
        List of features from TERM dataset
    total_term_lengths: List
        List of TERM lengths from the given dataset
    seq_lengths: List
        List of sequence lengths from the given dataset
    lengths: List
        TERM lengths or sequence lengths, depending on
        whether :code:`max_term_res` or :code:`max_seq_tokens` is set.
    batch_size : int or None, default=4
        Size of batches created. If variable sized batches are desired, set to None.
    sort_data : bool, default=False
        Create deterministic batches by sorting the data according to the
        specified length metric and creating batches from the sorted data.
        Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
    shuffle : bool, default=True
        Shuffle the data completely before creating batches.
        Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
    semi_shuffle : bool, default=False
        Sort the data according to the specified length metric,
        then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
        Within each partition perform a complete shuffle. The upside is that
        batching with similar lengths reduces padding making for more efficient computation,
        but the downside is that it does a less complete shuffle.
    semi_shuffle_cluster_size : int, default=500
        Size of partition to use when :code:`semi_shuffle=True`.
    batch_shuffle : bool, default=True
        If set to :code:`True`, shuffle samples within a batch.
    drop_last : bool, default=False
        If set to :code:`True`, drop the last samples if they don't form a complete batch.
    max_term_res : int or None, default=55000
        When :code:`batch_size=None, max_term_res>0, max_seq_tokens=None`,
        batch by fitting as many datapoints as possible with the total number of
        TERM residues included below `max_term_res`.
        Calibrated using :code:`nn.DataParallel` on two V100 GPUs.
    max_seq_tokens : int or None, default=None
        When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
        batch by fitting as many datapoints as possible with the total number of
        sequence residues included below `max_seq_tokens`.
    """
    def __init__(self,
                 dataset,
                 batch_size=4,
                 sort_data=False,
                 shuffle=True,
                 semi_shuffle=False,
                 semi_shuffle_cluster_size=500,
                 batch_shuffle=True,
                 drop_last=False,
                 max_term_res=55000,
                 max_seq_tokens=None):
        """
        Reads in and processes a given dataset.

        Given the provided dataset, load all the data. Then cluster the data using
        the provided method, either shuffled or sorted and then shuffled.

        Args
        ----
        dataset : TERMDataset
            Dataset to batch.
        batch_size : int or None, default=4
            Size of batches created. If variable sized batches are desired, set to None.
        sort_data : bool, default=False
            Create deterministic batches by sorting the data according to the
            specified length metric and creating batches from the sorted data.
            Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
        shuffle : bool, default=True
            Shuffle the data completely before creating batches.
            Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
        semi_shuffle : bool, default=False
            Sort the data according to the specified length metric,
            then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
            Within each partition perform a complete shuffle. The upside is that
            batching with similar lengths reduces padding making for more efficient computation,
            but the downside is that it does a less complete shuffle.
        semi_shuffle_cluster_size : int, default=500
            Size of partition to use when :code:`semi_shuffle=True`.
        batch_shuffle : bool, default=True
            If set to :code:`True`, shuffle samples within a batch.
        drop_last : bool, default=False
            If set to :code:`True`, drop the last samples if they don't form a complete batch.
        max_term_res : int or None, default=55000
            When :code:`batch_size=None, max_term_res>0, max_seq_tokens=None`,
            batch by fitting as many datapoints as possible with the total number of
            TERM residues included below `max_term_res`.
            Calibrated using :code:`nn.DataParallel` on two V100 GPUs.
        max_seq_tokens : int or None, default=None
            When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
            batch by fitting as many datapoints as possible with the total number of
            sequence residues included below `max_seq_tokens`. Exactly one of :code:`max_term_res`
            and :code:`max_seq_tokens` must be None.
        """
        super().__init__(dataset)
        self.size = len(dataset)
        self.dataset, self.total_term_lengths, self.seq_lengths = zip(*dataset)
        assert not (max_term_res is None
                    and max_seq_tokens is None), "Exactly one of max_term_res and max_seq_tokens must be None"
        if max_term_res is None and max_seq_tokens > 0:
            self.lengths = self.seq_lengths
        elif max_term_res > 0 and max_seq_tokens is None:
            self.lengths = self.total_term_lengths
        else:
            raise ValueError("Exactly one of max_term_res and max_seq_tokens must be None")
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

        # initialize clusters
        self._cluster()

    def _cluster(self):
        """ Shuffle data and make clusters of indices corresponding to batches of data.

        This method speeds up training by sorting data points with similar TERM lengths
        together, if :code:`sort_data` or :code:`semi_shuffle` are on. Under `sort_data`,
        the data is sorted by length. Under `semi_shuffle`, the data is broken up
        into clusters based on length and shuffled within the clusters. Otherwise,
        it is randomly shuffled. Data is then loaded into batches based on the number
        of proteins that will fit into the GPU without overloading it, based on
        :code:`max_term_res` or :code:`max_seq_tokens`.
        """

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
            last_cluster_idx = len(shuffle_borders) - 1
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
            if self.max_term_res is None and self.max_seq_tokens > 0:
                cap_len = self.max_seq_tokens
            elif self.max_term_res > 0 and self.max_seq_tokens is None:
                cap_len = self.max_term_res

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

        else:  # used fixed batch size
            for count, idx in enumerate(idx_list):
                if count != 0 and count % self.batch_size == 0:
                    clusters.append(batch)
                    batch = [idx]
                else:
                    batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
            clusters.append(batch)
        self.clusters = clusters

    def package(self, b_idx):
        """Package the given datapoints into tensors based on provided indices.

        Tensors are extracted from the data and padded. Coordinates are featurized
        and the length of TERMs and chain IDs are added to the data.

        Args
        ----
        b_idx : list of tuples (dicts, int, int)
            The feature dictionaries, the sum of the lengths of all TERMs, and the sum of all sequence lengths
            for each datapoint to package.

        Returns
        -------
        dict
            Collection of batched features required for running TERMinator. This contains:

            - :code:`msas` - the multiple sequence alignment

            - :code:`features` - the TERM features

            - :code:`ppoe` - ???

            - :code:`seq_lens` - the lengths of the sequences

            - :code:`focuses` - ???

            - :code:`contact_idxs` - contact indices

            - :code:`src_key_mask` - ???

            - :code:`X` - coordinates

            - :code:`x_mask` - ???

            - :code:`seqs` - the sequences

            - :code:`ids` - the PDB id??

            - :code:`chain_idx` - the chain IDs
        """
        return _package([b[0:2] for b in b_idx])

    def __len__(self):
        """Returns length of dataset, i.e. number of batches.

        Returns
        -------
        int
            length of dataset.
        """
        return len(self.clusters)

    def __iter__(self):
        """Allows iteration over dataset."""
        if self.shuffle or self.semi_shuffle:
            self._cluster()
            np.random.shuffle(self.clusters)
        for batch in self.clusters:
            yield batch


# needs to be outside of object for pickling reasons (?)
def read_lens(in_folder, pdb_id, min_protein_len=30):
    """ Reads the lengths specified in the proper .length file and return them.

    If the read sequence length is less than :code:`min_protein_len`, instead return None.

    Args
    ----
    in_folder : str
        folder to find TERM file.
    pdb_id : str
        PDB ID to load.
    min_protein_len : int
        minimum cutoff for loading TERM file.
    Returns
    -------
    pdb_id : str
        PDB ID that was loaded
    total_term_length : int
        number of TERMS in file
    seq_len : int
        sequence length of file, or None if sequence length is less than :code:`min_protein_len`
    """
    path = f"{in_folder}/{pdb_id}/{pdb_id}.length"
    # pylint: disable=unspecified-encoding
    with open(path, 'rt') as fp:
        total_term_length = int(fp.readline().strip())
        seq_len = int(fp.readline().strip())
        if seq_len < min_protein_len:
            return None
    return pdb_id, total_term_length, seq_len


class TERMLazyDataset(Dataset):
    """TERM Dataset that loads all feature files into a Pytorch Dataset-like structure.

    Unlike TERMDataset, this just loads feature filenames, not actual features.

    Attributes
    ----
    dataset : list
        list of tuples containing feature filenames, TERM length, and sequence length
    shuffle_idx : list
        array of indices for the dataset, for shuffling
    """
    def __init__(self, in_folder, pdb_ids=None, min_protein_len=30, num_processes=32):
        """
        Initializes current TERM dataset by reading in feature files.

        Reads in all feature files from the given directory, using multiprocessing
        with the provided number of processes. Stores the feature filenames, the TERM length,
        and the sequence length as a tuple representing the data. Can read from PDB ids or
        file paths directly. Uses the given protein length as a cutoff.

        Args
        ----
        in_folder : str
            path to directory containing feature files generated by :code:`scripts/data/preprocessing/generateDataset.py`
        pdb_ids: list, optional
            list of pdbs from `in_folder` to include in the dataset
        min_protein_len: int, default=30
            minimum length of a protein in the dataset
        num_processes: int, default=32
            number of processes to use during dataloading
        """
        self.dataset = []

        with mp.Pool(num_processes) as pool:

            if pdb_ids:
                print("Loading feature file paths")
                progress = tqdm(total=len(pdb_ids))

                def update_progress(res):
                    del res
                    progress.update(1)

                res_list = [
                    pool.apply_async(read_lens, (in_folder, pdb_id),
                                     kwds={"min_protein_len": min_protein_len},
                                     callback=update_progress) for pdb_id in pdb_ids
                ]
                pool.close()
                pool.join()
                progress.close()
                for res in res_list:
                    data = res.get()
                    if data is not None:
                        pdb_id, total_term_length, seq_len = data
                        filename = f"{in_folder}/{pdb_id}/{pdb_id}.features"
                        self.dataset.append((os.path.abspath(filename), total_term_length, seq_len))
            else:
                print("Loading feature file paths")

                filelist = list(glob.glob(f'{in_folder}/*/*.features'))
                progress = tqdm(total=len(filelist))

                def update_progress(res):
                    del res
                    progress.update(1)

                # get pdb_ids
                pdb_ids = [os.path.basename(path).split(".")[0] for path in filelist]

                res_list = [
                    pool.apply_async(read_lens, (in_folder, pdb_id),
                                     kwds={"min_protein_len": min_protein_len},
                                     callback=update_progress) for pdb_id in pdb_ids
                ]
                pool.close()
                pool.join()
                progress.close()
                for res in res_list:
                    data = res.get()
                    if data is not None:
                        pdb_id, total_term_length, seq_len = data
                        filename = f"{in_folder}/{pdb_id}/{pdb_id}.features"
                        self.dataset.append((os.path.abspath(filename), total_term_length, seq_len))

        self.shuffle_idx = np.arange(len(self.dataset))

    def shuffle(self):
        """Shuffle the dataset"""
        np.random.shuffle(self.shuffle_idx)

    def __len__(self):
        """Returns length of the given dataset.

        Returns
        -------
        int
            length of dataset
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """Extract a given item with provided index.

        Args
        ----
        idx : int
            Index of item to return.
        Returns
        ----
        data : dict
            Data from TERM file (as dict)
        total_term_len : int
            Sum of lengths of all TERMs
        seq_len : int
            Length of protein sequence
        """
        data_idx = self.shuffle_idx[idx]
        if isinstance(data_idx, list):
            return [self.dataset[i] for i in data_idx]
        return self.dataset[data_idx]


class TERMLazyBatchSampler(Sampler):
    """BatchSampler/Dataloader helper class for TERM data using TERMLazyDataset.

    Attributes
    ----------
    dataset : TERMLazyDataset
        Dataset to batch.
    size : int
        Length of dataset
    batch_size : int or None, default=4
        Size of batches created. If variable sized batches are desired, set to None.
    sort_data : bool, default=False
        Create deterministic batches by sorting the data according to the
        specified length metric and creating batches from the sorted data.
        Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
    shuffle : bool, default=True
        Shuffle the data completely before creating batches.
        Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
    semi_shuffle : bool, default=False
        Sort the data according to the specified length metric,
        then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
        Within each partition perform a complete shuffle. The upside is that
        batching with similar lengths reduces padding making for more efficient computation,
        but the downside is that it does a less complete shuffle.
    semi_shuffle_cluster_size : int, default=500
        Size of partition to use when :code:`semi_shuffle=True`.
    batch_shuffle : bool, default=True
        If set to :code:`True`, shuffle samples within a batch.
    drop_last : bool, default=False
        If set to :code:`True`, drop the last samples if they don't form a complete batch.
    max_term_res : int or None, default=55000
        When :code:`batch_size=None, max_term_res>0, max_seq_tokens=None`,
        batch by fitting as many datapoints as possible with the total number of
        TERM residues included below `max_term_res`.
        Calibrated using :code:`nn.DataParallel` on two V100 GPUs.
    max_seq_tokens : int or None, default=None
        When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
        batch by fitting as many datapoints as possible with the total number of
        sequence residues included below `max_seq_tokens`.
    term_matches_cutoff : int or None, default=None
        Use the top :code:`term_matches_cutoff` TERM matches for featurization.
        If :code:`None`, apply no cutoff.
    term_dropout : str or None, default=None
        Let `t` be the number of TERM matches in the given datapoint.
        Select a random int `n` from 1 to `t`, and take a random subset `n`
        of the given TERM matches to keep. If :code:`term_dropout='keep_first'`,
        keep the first match and choose `n-1` from the rest.
        If :code:`term_dropout='all'`, choose `n` matches from all matches.
    """
    def __init__(self,
                 dataset,
                 batch_size=4,
                 sort_data=False,
                 shuffle=True,
                 semi_shuffle=False,
                 semi_shuffle_cluster_size=500,
                 batch_shuffle=True,
                 drop_last=False,
                 max_term_res=55000,
                 max_seq_tokens=None,
                 term_matches_cutoff=None,
                 term_dropout=None):
        """
        Reads in and processes a given dataset.

        Given the provided dataset, load all the data. Then cluster the data using
        the provided method, either shuffled or sorted and then shuffled.

        Args
        ----
        dataset : TERMLazyDataset
            Dataset to batch.
        batch_size : int or None, default=4
            Size of batches created. If variable sized batches are desired, set to None.
        sort_data : bool, default=False
            Create deterministic batches by sorting the data according to the
            specified length metric and creating batches from the sorted data.
            Incompatible with :code:`shuffle=True` and :code:`semi_shuffle=True`.
        shuffle : bool, default=True
            Shuffle the data completely before creating batches.
            Incompatible with :code:`sort_data=True` and :code:`semi_shuffle=True`.
        semi_shuffle : bool, default=False
            Sort the data according to the specified length metric,
            then partition the data into :code:`semi_shuffle_cluster_size`-sized partitions.
            Within each partition perform a complete shuffle. The upside is that
            batching with similar lengths reduces padding making for more efficient computation,
            but the downside is that it does a less complete shuffle.
        semi_shuffle_cluster_size : int, default=500
            Size of partition to use when :code:`semi_shuffle=True`.
        batch_shuffle : bool, default=True
            If set to :code:`True`, shuffle samples within a batch.
        drop_last : bool, default=False
            If set to :code:`True`, drop the last samples if they don't form a complete batch.
        max_term_res : int or None, default=55000
            When :code:`batch_size=None, max_term_res>0, max_seq_tokens=None`,
            batch by fitting as many datapoints as possible with the total number of
            TERM residues included below `max_term_res`.
            Calibrated using :code:`nn.DataParallel` on two V100 GPUs.
        max_seq_tokens : int or None, default=None
            When :code:`batch_size=None, max_term_res=None, max_seq_tokens>0`,
            batch by fitting as many datapoints as possible with the total number of
            sequence residues included below `max_seq_tokens`.
        term_matches_cutoff : int or None, default=None
            Use the top :code:`term_matches_cutoff` TERM matches for featurization.
            If :code:`None`, apply no cutoff.
        term_dropout : str or None, default=None
            Let `t` be the number of TERM matches in the given datapoint.
            Select a random int `n` from 1 to `t`, and take a random subset `n`
            of the given TERM matches to keep. If :code:`term_dropout='keep_first'`,
            keep the first match and choose `n-1` from the rest.
            If :code:`term_dropout='all'`, choose `n` matches from all matches.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.size = len(dataset)
        self.filepaths, self.total_term_lengths, self.seq_lengths = zip(*dataset)
        assert not (max_term_res is None
                    and max_seq_tokens is None), "Exactly one of max_term_res and max_seq_tokens must be None"
        if max_term_res is None and max_seq_tokens > 0:
            self.lengths = self.seq_lengths
        elif max_term_res > 0 and max_seq_tokens is None:
            self.lengths = self.total_term_lengths
        else:
            raise Exception("Exactly one of max_term_res and max_seq_tokens must be None")
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
        assert term_dropout in ["keep_first", "all", None], f"term_dropout={term_dropout} is not a valid argument"
        self.term_dropout = term_dropout

        assert not (shuffle and semi_shuffle), "Lazy Dataloader shuffle and semi shuffle cannot both be set"

        # initialize clusters
        self._cluster()

    def _cluster(self):
        """ Shuffle data and make clusters of indices corresponding to batches of data.

        This method speeds up training by sorting data points with similar TERM lengths
        together, if :code:`sort_data` or :code:`semi_shuffle` are on. Under `sort_data`,
        the data is sorted by length. Under `semi_shuffle`, the data is broken up
        into clusters based on length and shuffled within the clusters. Otherwise,
        it is randomly shuffled. Data is then loaded into batches based on the number
        of proteins that will fit into the GPU without overloading it, based on
        :code:`max_term_res` or :code:`max_seq_tokens`.
        """

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
            last_cluster_idx = len(shuffle_borders) - 1
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
            if self.max_term_res is None and self.max_seq_tokens > 0:
                cap_len = self.max_seq_tokens
            elif self.max_term_res > 0 and self.max_seq_tokens is None:
                cap_len = self.max_term_res

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

        else:  # used fixed batch size
            for count, idx in enumerate(idx_list):
                if count != 0 and count % self.batch_size == 0:
                    clusters.append(batch)
                    batch = [idx]
                else:
                    batch.append(idx)

        if len(batch) > 0 and not self.drop_last:
            clusters.append(batch)
        self.clusters = clusters

    def package(self, b_idx):
        """Package the given datapoints into tensors based on provided indices.

        Tensors are extracted from the data and padded. Coordinates are featurized
        and the length of TERMs and chain IDs are added to the data.

        Args
        ----
        b_idx : list of (str, int, int)
            The path to the feature files, the sum of the lengths of all TERMs, and the sum of all sequence lengths
            for each datapoint to package.

        Returns
        -------
        dict
            Collection of batched features required for running TERMinator. This contains:

            - :code:`msas` - the multiple sequence alignment

            - :code:`features` - the TERM features

            - :code:`ppoe` - ???

            - :code:`seq_lens` - the lengths of the sequences

            - :code:`focuses` - ???

            - :code:`contact_idxs` - contact indices

            - :code:`src_key_mask` - ???

            - :code:`X` - coordinates

            - :code:`x_mask` - ???

            - :code:`seqs` - the sequences

            - :code:`ids` - the PDB id??

            - :code:`chain_idx` - the chain IDs
        """
        if self.batch_shuffle:
            b_idx_copy = b_idx[:]
            random.shuffle(b_idx_copy)
            b_idx = b_idx_copy

        # load the files specified by filepaths
        batch = []
        for data in b_idx:
            filepath = data[0]
            with open(filepath, 'rb') as fp:
                batch.append((pickle.load(fp), data[1]))
                if 'ppoe' not in batch[-1][0].keys():
                    print(filepath)

        # package batch
        packaged_batch = _package(batch)

        features = packaged_batch["features"]
        msas = packaged_batch["msas"]
        # apply TERM matches cutoff
        if self.term_matches_cutoff:
            features = features[:, :self.term_matches_cutoff]
            msas = msas[:, :self.term_matches_cutoff]
        # apply TERM matches dropout
        if self.term_dropout:
            # sample a random number of alignments to keep
            n_batch, n_align, n_terms, n_features = features.shape
            if self.term_dropout == 'keep_first':
                n_keep = torch.randint(0, n_align, [1]).item()
            elif self.term_dropout == 'all':
                n_keep = torch.randint(1, n_align, [1]).item()
            # sample from a multinomial distribution
            weights = torch.ones([1, 1]).expand([n_batch * n_terms, n_keep])
            if n_keep == 0:
                sample_idx = torch.ones(1)
            else:
                sample_idx = torch.multinomial(weights, n_keep)
                sample_idx = sample_idx.view([n_batch, n_terms, n_keep]).transpose(-1, -2)
                sample_idx_features = sample_idx.unsqueeze(-1).expand([n_batch, n_keep, n_terms, n_features])
                sample_idx_msas = sample_idx

            if self.term_dropout == 'keep_first':
                if n_keep == 0:
                    features = features[:, 0:1]
                    msas = msas[:, 0:1]
                else:
                    sample_features = torch.gather(features[:, 1:], 1, sample_idx_features)
                    sample_msas = torch.gather(msas[:, 1:], 1, sample_idx_msas)
                    features = torch.cat([features[:, 0:1], sample_features], dim=1)
                    msas = torch.cat([msas[:, 0:1], sample_msas], dim=1)
            elif self.term_dropout == 'all':
                features = torch.gather(features, 1, sample_idx_features)
                msas = torch.gather(msas, 1, sample_idx_msas)

        packaged_batch["features"] = features
        packaged_batch["msas"] = msas

        return packaged_batch

    def __len__(self):
        """Returns length of dataset, i.e. number of batches.

        Returns
        -------
        int
            length of dataset.
        """
        return len(self.clusters)

    def __iter__(self):
        """Allows iteration over dataset."""
        if self.shuffle or self.semi_shuffle:
            self._cluster()
            np.random.shuffle(self.clusters)
        for batch in self.clusters:
            yield batch
