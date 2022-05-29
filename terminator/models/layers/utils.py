""" Util functions useful in TERMinator modules """
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
import torch_geometric.utils

# pylint: disable=no-member

# batchify functions



def pad_sequence_12(sequences, padding_value=0):
    """Given a sequence of tensors, batch them together by pads both dims 1 and 2 to max length.

    Args
    ----
    sequences : list of torch.Tensor
        Sequence of tensors with number of axes `N >= 2`
    padding value : int, default=0
        What value to pad the tensors with

    Returns
    -------
    out_tensor : torch.Tensor
        Batched tensor with shape (n_batch, max_dim1, max_dim2, ...)
    """
    n_batches = len(sequences)
    out_dims = list(sequences[0].size())
    dim1, dim2 = 0, 1
    max_dim1 = max([s.size(dim1) for s in sequences])
    max_dim2 = max([s.size(dim2) for s in sequences])
    out_dims[dim1] = max_dim1
    out_dims[dim2] = max_dim2
    out_dims = [n_batches] + out_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        len1 = tensor.size(0)
        len2 = tensor.size(1)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :len1, :len2, ...] = tensor

    return out_tensor


def batchify(batched_flat_terms, term_lens):
    """ Take a flat representation of TERM information and batch them into a stacked representation.

    In the TERM information condensor, TERM information is initially stored by concatenating all
    TERM tensors side by side in one dimension. However, for message passing, it's convenient to batch
    these TERMs by splitting them and stacking them in a new dimension.

    Args
    ----
    batched_flat_terms : torch.Tensor
        Tensor with shape :code:`(n_batch, sum_term_len, ...)`
    term_lens : list of (list of int)
        Length of each TERM per protein

    Returns
    -------
    batchify_terms : torch.Tensor
        Tensor with shape :code:`(n_batch, max_num_terms, max_term_len, ...)`
    """
    n_batches = batched_flat_terms.shape[0]
    flat_terms = torch.unbind(batched_flat_terms)
    list_terms = [torch.split(flat_terms[i], term_lens[i]) for i in range(n_batches)]
    padded_terms = [pad_sequence(terms) for terms in list_terms]
    padded_terms = [term.transpose(0, 1) for term in padded_terms]
    batchify_terms = pad_sequence_12(padded_terms)
    return batchify_terms


# gather and cat functions
# struct level


def gather_edges(edges, neighbor_idx):
    """ Gather the edge features of the nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    edges : torch.Tensor
        The edge features in dense form
        Shape: n_batch x n_res x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    edge_features : torch.Tensor
        The gathered edge features
        Shape : n_batch x n_res x k x n_hidden
    """
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    """ Gather node features of nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    neighbor_features : torch.Tensor
        The gathered neighbor node features
        Shape : n_batch x n_res x k x n_hidden
    """
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """ Concatenate node features onto the ends of gathered edge features given kNN sparse edge indices

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_res x n_hidden
    h_neighbors : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_res x k x n_hidden
    """
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def cat_edge_endpoints(h_edges, h_nodes, E_idx):
    """ Concatenate both node features onto the ends of gathered edge features given kNN sparse edge indices

    Args
    ----
    h_edges : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_res x k x n_hidden
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_res x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_res x k x n_hidden
    """
    # Neighbor indices E_idx [B,N,K]
    # Edge features h_edges [B,N,N,C]
    # Node features h_nodes [B,N,C]
    k = E_idx.shape[-1]

    h_i_idx = E_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, k).contiguous()
    h_j_idx = E_idx

    h_i = gather_nodes(h_nodes, h_i_idx)
    h_j = gather_nodes(h_nodes, h_j_idx)

    # output features [B, N, K, 3C]
    h_nn = torch.cat([h_i, h_j, h_edges], -1)
    return h_nn


def gather_pairEs(pairEs, neighbor_idx):
    """ Gather the pair energies features of the nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    pairEs : torch.Tensor
        The pair energies in dense form
        Shape: n_batch x n_res x n_res x n_aa x n_aa
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    pairE_features : torch.Tensor
        The gathered pair energies
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    n_aa = pairEs.size(-1)
    neighbors = neighbor_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n_aa, n_aa)
    pairE_features = torch.gather(pairEs, 2, neighbors)
    return pairE_features


# term level


def gather_term_nodes(nodes, neighbor_idx):
    """ Gather TERM node features of nearest neighbors.

    Adatped from https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_terms x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    neighbor_features : torch.Tensor
        The gathered neighbor node features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    # Features [B,T,N,C] at Neighbor indices [B,T,N,K] => [B,T,N,K,C]
    # Flatten and expand indices per batch [B,T,N,K] => [B,T,NK] => [B,T,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], neighbor_idx.shape[1], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, -1, nodes.size(3))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 2, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:4] + [-1])
    return neighbor_features


def gather_term_edges(edges, neighbor_idx):
    """ Gather the TERM edge features of the nearest neighbors.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    edges : torch.Tensor
        The edge features in dense form
        Shape: n_batch x n_terms x n_res x n_res x n_hidden
    neighbor_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    edge_features : torch.Tensor
        The gathered edge features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    # Features [B,T,N,N,C] at Neighbor indices [B,T,N,K] => Neighbor features [B,T,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 3, neighbors)
    return edge_features


def cat_term_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """ Concatenate node features onto the ends of gathered edge features given kNN sparse edge indices

    From https://github.com/jingraham/neurips19-graph-protein-design

    Args
    ----
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_terms x n_res x n_hidden
    h_neighbors : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_terms x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    h_nodes = gather_term_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def cat_term_edge_endpoints(h_edges, h_nodes, E_idx):
    """ Concatenate both node features onto the ends of gathered edge features given kNN sparse edge indices

    Args
    ----
    h_edges : torch.Tensor
        The gathered edge features
        Shape: n_batch x n_terms x n_res x k x n_hidden
    h_nodes : torch.Tensor
        The node features for all nodes
        Shape: n_batch x n_terms x n_res x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    h_nn : torch.Tensor
        The gathered concatenated node and edge features
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    # Neighbor indices E_idx [B,T,N,K]
    # Edge features h_edges [B,T,N,N,C]
    # Node features h_nodes [B,T,N,C]
    k = E_idx.shape[-1]

    h_i_idx = E_idx[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, k).contiguous()
    h_j_idx = E_idx

    h_i = gather_term_nodes(h_nodes, h_i_idx)
    h_j = gather_term_nodes(h_nodes, h_j_idx)

    # e_ij = gather_edges(h_edges, E_idx)
    e_ij = h_edges

    # output features [B, T, N, K, 3C]
    h_nn = torch.cat([h_i, h_j, e_ij], -1)
    return h_nn



# merge edge fns


def merge_duplicate_edges(h_E_update, E_idx):
    """ Average embeddings across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings.

    Args
    ----
    h_E_update : torch.Tensor
        Update tensor for edges embeddings in kNN sparse form
        Shape : n_batch x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    merged_E_updates : torch.Tensor
        Edge update with merged updates for bidirectional edges
        Shape : n_batch x n_res x k x n_hidden
    """
    dev = h_E_update.device
    n_batch, n_nodes, _, hidden_dim = h_E_update.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_nodes, n_nodes, hidden_dim)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, hidden_dim).to(dev)
    collection.scatter_(2, neighbor_idx, h_E_update)
    # transpose to get same edge in reverse direction
    collection = collection.transpose(1, 2)
    # gather reverse edges
    reverse_E_update = gather_edges(collection, E_idx)
    # average h_E_update and reverse_E_update at non-zero positions
    merged_E_updates = torch.where(reverse_E_update != 0, (h_E_update + reverse_E_update) / 2, h_E_update)
    return merged_E_updates


def merge_duplicate_edges_geometric(h_E_update, edge_index):
    """ Average embeddings across bidirectional edges for Torch Geometric graphs

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings.

    This function assumes edge_index is sorted by columns, and will fail if
    this is not the case.

    Args
    ----
    h_E_update : torch.Tensor
        Update tensor for edges embeddings in Torch Geometric sparse form
        Shape : n_edge x n_hidden
    edge_index : torch.LongTensor
        Torch Geometric sparse edge indices
        Shape : 2 x n_edge

    Returns
    -------
    merged_E_updates : torch.Tensor
        Edge update with merged updates for bidirectional edges
        Shape : n_edge x n_hidden
    """

    original_edge = torch.ones_like(edge_index[0])
    dummy_edge = torch.zeros_like(edge_index[0])
    edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

    u_edge_index, h_E_update = torch_geometric.utils.coalesce(
                                    edge_index=torch.cat([edge_index, edge_index_t], dim=-1),
                                    edge_attr=torch.cat([h_E_update, h_E_update], dim=0),
                                    reduce="mean",
                                    sort_by_row=False)
    u_edge_index_c, u_count = torch_geometric.utils.coalesce(
                                    edge_index=torch.cat([edge_index, edge_index_t], dim=-1),
                                    edge_attr=torch.cat([original_edge, dummy_edge], dim=0),
                                    reduce="max",
                                    sort_by_row=False)
    assert (u_edge_index == u_edge_index_c).all()
    select = (u_count == 1)
    assert (u_edge_index[:, select] == edge_index).all()

    return h_E_update[select]


def merge_duplicate_term_edges(h_E_update, E_idx):
    """ Average embeddings across bidirectional TERM edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings.

    Args
    ----
    h_E_update : torch.Tensor
        Update tensor for edges embeddings in kNN sparse form
        Shape : n_batch x n_terms x n_res x k x n_hidden
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_terms x n_res x k

    Returns
    -------
    merged_E_updates : torch.Tensor
        Edge update with merged updates for bidirectional edges
        Shape : n_batch x n_terms x n_res x k x n_hidden
    """
    dev = h_E_update.device
    n_batch, n_terms, n_aa, _, hidden_dim = h_E_update.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_terms, n_aa, n_aa, hidden_dim)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, -1, hidden_dim).to(dev)
    collection.scatter_(3, neighbor_idx, h_E_update)
    # transpose to get same edge in reverse direction
    collection = collection.transpose(2, 3)
    # gather reverse edges
    reverse_E_update = gather_term_edges(collection, E_idx)
    # average h_E_update and reverse_E_update at non-zero positions
    merged_E_updates = torch.where(reverse_E_update != 0, (h_E_update + reverse_E_update) / 2, h_E_update)
    return merged_E_updates


def merge_duplicate_pairE(h_E, E_idx):
    """ Average pair energy tables across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    try:
        return merge_duplicate_pairE_dense(h_E, E_idx)
    except RuntimeError as err:
        print(err, file=sys.stderr)
        print("We're handling this error as if it's an out-of-memory error", file=sys.stderr)
        torch.cuda.empty_cache()  # this is probably unnecessary but just in case
        return merge_duplicate_pairE_sparse(h_E, E_idx)


def merge_duplicate_pairE_dense(h_E, E_idx):
    """ Dense method to average pair energy tables across bidirectional edges.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    dev = h_E.device
    n_batch, n_nodes, _, n_aa, _ = h_E.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_nodes, n_nodes, n_aa, n_aa)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n_aa, n_aa).to(dev)
    collection.scatter_(2, neighbor_idx, h_E)
    # transpose to get same edge in reverse direction
    collection = collection.transpose(1, 2)
    # transpose each pair energy table as well
    collection = collection.transpose(-2, -1)
    # gather reverse edges
    reverse_E = gather_pairEs(collection, E_idx)
    # average h_E and reverse_E at non-zero positions
    merged_E = torch.where(reverse_E != 0, (h_E + reverse_E) / 2, h_E)
    return merged_E


# TODO: rigorous test that this is equiv to the dense version
def merge_duplicate_pairE_sparse(h_E, E_idx):
    """ Sparse method to average pair energy tables across bidirectional edges.

    Note: This method involves a significant slowdown so it's only worth using if memory is an issue.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    Args
    ----
    h_E : torch.Tensor
        Pair energies in kNN sparse form
        Shape : n_batch x n_res x k x n_aa x n_aa
    E_idx : torch.LongTensor
        kNN sparse edge indices
        Shape : n_batch x n_res x k

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_batch x n_res x k x n_aa x n_aa
    """
    dev = h_E.device
    n_batch, n_nodes, k, n_aa, _ = h_E.shape
    # convert etab into a sparse etab
    # self idx of the edge
    ref_idx = E_idx[:, :, 0:1].expand(-1, -1, k)
    # sparse idx
    g_idx = torch.cat([E_idx.unsqueeze(1), ref_idx.unsqueeze(1)], dim=1)
    sparse_idx = g_idx.view([n_batch, 2, -1])
    # generate a 1D idx for the forward and backward direction
    scaler = torch.ones_like(sparse_idx).to(dev)
    scaler = scaler * n_nodes
    scaler_f = scaler
    scaler_f[:, 0] = 1
    scaler_r = torch.flip(scaler_f, [1])
    batch_offset = torch.arange(n_batch).unsqueeze(-1).expand([-1, n_nodes * k]) * n_nodes * k
    batch_offset = batch_offset.to(dev)
    sparse_idx_f = torch.sum(scaler_f * sparse_idx, 1) + batch_offset
    flat_idx_f = sparse_idx_f.view([-1])
    sparse_idx_r = torch.sum(scaler_r * sparse_idx, 1) + batch_offset
    flat_idx_r = sparse_idx_r.view([-1])
    # generate sparse tensors
    flat_h_E_f = h_E.view([n_batch * n_nodes * k, n_aa**2])
    reverse_h_E = h_E.transpose(-2, -1).contiguous()
    flat_h_E_r = reverse_h_E.view([n_batch * n_nodes * k, n_aa**2])
    sparse_etab_f = torch.sparse_coo_tensor(flat_idx_f.unsqueeze(0), flat_h_E_f,
                                            (n_batch * n_nodes * n_nodes, n_aa**2))
    count_f = torch.sparse_coo_tensor(flat_idx_f.unsqueeze(0), torch.ones_like(flat_idx_f),
                                      (n_batch * n_nodes * n_nodes, ))
    sparse_etab_r = torch.sparse_coo_tensor(flat_idx_r.unsqueeze(0), flat_h_E_r,
                                            (n_batch * n_nodes * n_nodes, n_aa**2))
    count_r = torch.sparse_coo_tensor(flat_idx_r.unsqueeze(0), torch.ones_like(flat_idx_r),
                                      (n_batch * n_nodes * n_nodes, ))
    # merge
    sparse_etab = sparse_etab_f + sparse_etab_r
    sparse_etab = sparse_etab.coalesce()
    count = count_f + count_r
    count = count.coalesce()

    # this step is very slow, but implementing something faster is probably a lot of work
    # requires pytorch 1.10 to be fast enough to be usable
    collect = sparse_etab.index_select(0, flat_idx_f).to_dense()
    weight = count.index_select(0, flat_idx_f).to_dense()

    flat_merged_etab = collect / weight.unsqueeze(-1)
    merged_etab = flat_merged_etab.view(h_E.shape)
    return merged_etab


def merge_duplicate_pairE_geometric(h_E, edge_index):
    """ Sparse method to average pair energy tables across bidirectional edges with Torch Geometric.

    TERMinator edges are represented as two bidirectional edges, and to allow for
    communication between these edges we average the embeddings. In the case for
    pair energies, we transpose the tables to ensure that the pair energy table
    is symmetric upon inverse (e.g. the pair energy between i and j should be
    the same as the pair energy between j and i)

    This function assumes edge_index is sorted by columns, and will fail if
    this is not the case.

    Args
    ----
    h_E : torch.Tensor
        Pair energies in Torch Geometric sparse form
        Shape : n_edge x 400
    E_idx : torch.LongTensor
        Torch Geometric sparse edge indices
        Shape : 2 x n_edge

    Returns
    -------
    torch.Tensor
        Pair energies with merged energies for bidirectional edges
        Shape : n_edge x 400
    """
    original_edge = torch.ones_like(edge_index[0])
    dummy_edge = torch.zeros_like(edge_index[0])
    edge_index_t = torch.stack([edge_index[1], edge_index[0]])
    h_E_transpose = h_E.view([-1, 20, 20]).transpose(-1, -2).reshape([-1, 400])

    unfused_h_E = torch.cat([h_E, h_E_transpose], dim=0)
    unfused_edge_index = torch.cat([edge_index, edge_index_t], dim=-1)
    unfused_count = torch.cat([original_edge, dummy_edge], dim=0)

    f_edge_index, h_E = torch_geometric.utils.coalesce(
                            edge_index=unfused_edge_index,
                            edge_attr=unfused_h_E,
                            reduce="mean",
                            sort_by_row=False)
    f_edge_index_c, f_count = torch_geometric.utils.coalesce(
                                    edge_index=unfused_edge_index,
                                    edge_attr=unfused_count,
                                    reduce="max",
                                    sort_by_row=False)
    assert (f_edge_index == f_edge_index_c).all()
    select = (f_count == 1)
    assert (f_edge_index[:, select] == edge_index).all()

    return h_E[select]


# edge aggregation fns


def aggregate_edges(edge_embeddings, E_idx, max_seq_len):
    """ Aggregate TERM edge embeddings into a sequence-level dense edge features tensor

    Args
    ----
    edge_embeddings : torch.Tensor
        TERM edge features tensor
        Shape : n_batch x n_terms x n_aa x n_neighbors x n_hidden
    E_idx : torch.LongTensor
        TERM edge indices
        Shape : n_batch x n_terms x n_aa x n_neighbors
    max_seq_len : int
        Max length of a sequence in the batch

    Returns
    -------
    torch.Tensor
        Dense sequence-level edge features
        Shape : n_batch x max_seq_len x max_seq_len x n_hidden
    """
    dev = edge_embeddings.device
    n_batch, _, _, n_neighbors, hidden_dim = edge_embeddings.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, max_seq_len, max_seq_len, hidden_dim)).to(dev)
    # edge the edge indecies
    self_idx = E_idx[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, n_neighbors)
    neighbor_idx = E_idx
    # tensor needed for accumulation
    layer = torch.arange(n_batch).view([n_batch, 1, 1, 1]).expand(neighbor_idx.shape).to(dev)
    # thicc index_put_
    collection.index_put_((layer, self_idx, neighbor_idx), edge_embeddings, accumulate=True)

    # we also need counts for averaging
    count = torch.zeros((n_batch, max_seq_len, max_seq_len)).to(dev)
    count_idx = torch.ones_like(neighbor_idx).float().to(dev)
    count.index_put_((layer, self_idx, neighbor_idx), count_idx, accumulate=True)

    # we need to set all 0s to 1s so we dont get nans
    count[count == 0] = 1

    return collection / count.unsqueeze(-1)
