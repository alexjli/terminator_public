import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from layers.gvp import vs_concat

"""
    batchify functions
"""

# pads both dims 1 and 2 to max length
def pad_sequence_12(sequences, padding_value=0):
    n_batches = len(sequences)
    out_dims = list(sequences[0].size())
    dim1, dim2 = 0,1
    max_dim1 = max([s.size(dim1) for s in sequences])
    max_dim2 = max([s.size(dim2) for s in sequences])
    out_dims[dim1] = max_dim1
    out_dims[dim2] = max_dim2
    out_dims = [n_batches] + out_dims
    #print(out_dims)

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        len1 = tensor.size(0)
        len2 = tensor.size(1)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :len1, :len2, ...] = tensor

    return out_tensor

def batchify(batched_flat_terms, term_lens):
    n_batches = batched_flat_terms.shape[0]
    flat_terms = torch.unbind(batched_flat_terms)
    list_terms = [torch.split(flat_terms[i], term_lens[i]) for i in range(n_batches)]
    padded_terms = [pad_sequence(terms) for terms in list_terms]
    padded_terms = [term.transpose(0,1) for term in padded_terms]
    batchify = pad_sequence_12(padded_terms)
    return batchify


class BatchifyTERM(nn.Module):
    def __init__(self):
        super(BatchifyTERM, self).__init__()

    def forward(self, batched_flat_terms, term_lens):
        return batchify(batched_flat_terms, term_lens)

"""
    gather and cat functions
"""

""" struct level """
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def cat_edge_endpoints(h_edges, h_nodes, E_idx):
    # Neighbor indices E_idx [B,N,K]
    # Edge features h_edges [B,N,N,C]
    # Node features h_nodes [B,N,C]
    n_batches, n_nodes, k = E_idx.shape

    h_i_idx = E_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, k).contiguous()
    h_j_idx = E_idx

    h_i = gather_nodes(h_nodes, h_i_idx)
    h_j = gather_nodes(h_nodes, h_j_idx)

    #e_ij = gather_edges(h_edges, E_idx)
    e_ij = h_edges

    # output features [B, N, K, 3C]
    h_nn = torch.cat([h_i, h_j, e_ij], -1)
    return h_nn

""" term level """
def gather_term_nodes(nodes, neighbor_idx):
    # Features [B,T,N,C] at Neighbor indices [B,T,N,K] => [B,T,N,K,C]
    # Flatten and expand indices per batch [B,T,N,K] => [B,T,NK] => [B,T,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], neighbor_idx.shape[1], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, -1, nodes.size(3))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 2, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:4] + [-1])
    return neighbor_features

def gather_term_edges(edges, neighbor_idx):
    # Features [B,T,N,N,C] at Neighbor indices [B,T,N,K] => Neighbor features [B,T,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 3, neighbors)
    return edge_features

def cat_term_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_term_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

def cat_term_edge_endpoints(h_edges, h_nodes, E_idx):
    # Neighbor indices E_idx [B,T,N,K]
    # Edge features h_edges [B,T,N,N,C]
    # Node features h_nodes [B,T,N,C]
    n_batches, n_terms, n_nodes, k = E_idx.shape

    h_i_idx = E_idx[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, k).contiguous()
    h_j_idx = E_idx

    h_i = gather_term_nodes(h_nodes, h_i_idx)
    h_j = gather_term_nodes(h_nodes, h_j_idx)

    #e_ij = gather_edges(h_edges, E_idx)
    e_ij = h_edges

    # output features [B, T, N, K, 3C]
    h_nn = torch.cat([h_i, h_j, e_ij], -1)
    return h_nn

""" gvp cat functions """
def cat_gvp_neighbors_nodes(h_nodes, h_neighbors, E_idx, nv_nodes, nv_neighbors):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return vs_concat(h_neighbors, h_nodes, nv_neighbors, nv_nodes)

def cat_gvp_term_edge_endpoints(h_edges, h_nodes, E_idx, n_node, n_edge):
    # Neighbor indices E_idx [B,T,N,K]
    # Edge features h_edges [B,T,N,N,C]
    # Node features h_nodes [B,T,N,C]
    n_batches, n_terms, n_nodes, k = E_idx.shape

    h_i_idx = E_idx[:, :, :, 0].unsqueeze(-1).expand(-1, -1, -1, k).contiguous()
    h_j_idx = E_idx

    h_i = gather_term_nodes(h_nodes, h_i_idx)
    h_j = gather_term_nodes(h_nodes, h_j_idx)

    #e_ij = gather_edges(h_edges, E_idx)
    e_ij = h_edges

    # output features [B, T, N, K, 3C]
    h_nn = vs_concat(vs_concat(h_i, h_j, n_node, n_node), e_ij, n_node * 2, n_edge)
    return h_nn

def cat_gvp_edge_endpoints(h_edges, h_nodes, E_idx, n_node, n_edge):
    # Neighbor indices E_idx [B,N,K]
    # Edge features h_edges [B,N,N,C]
    # Node features h_nodes [B,N,C]
    n_batches, n_nodes, k = E_idx.shape

    h_i_idx = E_idx[ :, :, 0].unsqueeze(-1).expand(-1, -1, k).contiguous()
    h_j_idx = E_idx

    h_i = gather_nodes(h_nodes, h_i_idx)
    h_j = gather_nodes(h_nodes, h_j_idx)

    #e_ij = gather_edges(h_edges, E_idx)
    e_ij = h_edges

    # output features [B, N, K, 3C]
    h_nn = vs_concat(vs_concat(h_i, h_j, n_node, n_node), e_ij, n_node * 2, n_edge)
    return h_nn


"""
    merge edge fns
"""

def merge_duplicate_edges(h_E_update, E_idx):
    dev = h_E_update.device
    n_batch, n_nodes, k, hidden_dim = h_E_update.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_nodes, n_nodes, hidden_dim)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, hidden_dim).to(dev)
    collection.scatter_(2, neighbor_idx, h_E_update)
    # transpose to get same edge in reverse direction
    collection = collection.transpose(1,2)
    # gather reverse edges
    reverse_E_update = gather_edges(collection, E_idx)
    # average h_E_update and reverse_E_update at non-zero positions
    merged_E_updates = torch.where(reverse_E_update != 0, (h_E_update + reverse_E_update)/2, h_E_update)
    return merged_E_updates

def merge_duplicate_term_edges(h_E_update, E_idx):
    dev = h_E_update.device
    n_batch, n_terms, n_aa, n_neighbors, hidden_dim = h_E_update.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, n_terms, n_aa, n_aa, hidden_dim)).to(dev)
    neighbor_idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, -1, hidden_dim).to(dev)
    collection.scatter_(3, neighbor_idx, h_E_update)
    # transpose to get same edge in reverse direction
    collection = collection.transpose(2,3)
    # gather reverse edges
    reverse_E_update = gather_term_edges(collection, E_idx)
    # average h_E_update and reverse_E_update at non-zero positions
    merged_E_updates = torch.where(reverse_E_update != 0, (h_E_update + reverse_E_update)/2, h_E_update)
    return merged_E_updates

"""
    edge aggregation fns
"""
def aggregate_edges(edge_embeddings, E_idx, max_seq_len):
    dev = edge_embeddings.device
    n_batch, n_terms, n_aa, n_neighbors, hidden_dim = edge_embeddings.shape
    # collect edges into NxN tensor shape
    collection = torch.zeros((n_batch, max_seq_len, max_seq_len, hidden_dim)).to(dev)
    # edge the edge indecies
    self_idx = E_idx[:,:,:,0].unsqueeze(-1).expand(-1, -1, -1, n_neighbors)
    neighbor_idx = E_idx
    # tensor needed for accumulation
    layer = torch.arange(n_batch).view([n_batch, 1, 1, 1]).expand(neighbor_idx.shape).to(dev)
    # thicc index_put_
    collection.index_put_((layer, self_idx, neighbor_idx), edge_embeddings, accumulate = True)

    # we also need counts for averaging
    count = torch.zeros((n_batch, max_seq_len, max_seq_len)).to(dev)
    count_idx = torch.ones_like(neighbor_idx).float().to(dev)
    count.index_put_((layer, self_idx, neighbor_idx), count_idx, accumulate = True)

    # we need to set all 0s to 1s so we dont get nans
    count[count == 0] = 1

    return collection / count.unsqueeze(-1)

"""
    some random debugging fns
"""

ERROR_FILE = '/nobackup/users/alexjli/TERMinator/run.error'

def process_nan(t, prev_t, msg):
    if torch.isnan(t).any():
        with open(ERROR_FILE, 'w') as fp:
            fp.write(repr(prev_t) + '\n')
            fp.write(repr(t) + '\n')
            fp.write(str(msg))
        raise KeyboardInterrupt

def stat_cuda(msg):
    dev1, dev2, dev3 = 0, 1, 2
    print('--', msg)
    print('allocated 1: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated(dev1) / 1024 / 1024,
        torch.cuda.max_memory_allocated(dev1) / 1024 / 1024,
        torch.cuda.memory_cached(dev1) / 1024 / 1024,
        torch.cuda.max_memory_cached(dev1) / 1024 / 1024
    ))
    print('allocated 2: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated(dev2) / 1024 / 1024,
        torch.cuda.max_memory_allocated(dev2) / 1024 / 1024,
        torch.cuda.memory_cached(dev2) / 1024 / 1024,
        torch.cuda.max_memory_cached(dev2) / 1024 / 1024
    ))
    print('allocated 3: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated(dev3) / 1024 / 1024,
        torch.cuda.max_memory_allocated(dev3) / 1024 / 1024,
        torch.cuda.memory_cached(dev3) / 1024 / 1024,
        torch.cuda.max_memory_cached(dev3) / 1024 / 1024
    ))

def inf_nan_hook_fn(self, input, output):
    """
    try:
        print('mean', self.running_mean)
        print('var', self.running_var)
    except:
        pass
    """
    if has_large(input[0]):
        print("large mag input")
        with open(ERROR_FILE, 'w') as fp:
            abs_input = torch.abs(input[0])
            fp.write('Input to ' + repr(self) + ' forward\n')
            fp.write('Inputs from b_idx 0 channel 0' + repr(input[0][0][0]) + '\n')
            fp.write("top 5 " + repr(torch.topk(abs_input.view([-1]), 5)) + '\n')
            fp.write('Weights ' + repr(self.weight) + '\n')
            fp.write('Biases ' + repr(self.bias) + '\n')
            try:
                fp.write('Running mean ' + repr(self.running_mean) + '\n')
                fp.write('Running var ' + repr(self.running_var) + '\n')
            except:
                fp.write('Not a batchnorm layer')
        exit()

    elif (output == float('inf')).any() or (output == float('-inf')).any() or torch.isnan(output).any():
        print('we got an inf/nan rip')
        with open(ERROR_FILE, 'w') as fp:
            fp.write('Inf/nan is from ' + repr(self) + ' forward\n')
            fp.write('Inputs from b_idx 0 channel 0' + repr(input[0][0][0]) + '\n')
            fp.write('Outputs from b_idx 0 channel 0' + repr(output[0][0]) + '\n')
            fp.write('Weights ' + repr(self.weight) + '\n')
            fp.write('Biases ' + repr(self.bias) + '\n')
            try:
                fp.write('Running mean ' + repr(self.running_mean) + '\n')
                fp.write('Running var ' + repr(self.running_var) + '\n')
            except:
                fp.write('Not a batchnorm layer')
        exit()
    else:
        try:
            if is_nan_inf(self.running_mean) or is_nan_inf(self.running_var):
                with open(ERROR_FILE, 'w') as fp:
                    fp.write('Inf/nan is from ' + repr(self) + ' forward\n')
                    fp.write('Inputs from b_idx 0 channel 0' + repr(input[0][0][0]) + '\n')
                    fp.write('Outputs from b_idx 0 channel 0' + repr(output[0][0]) + '\n')
                    fp.write('Weights ' + repr(self.weight) + '\n')
                    fp.write('Biases ' + repr(self.bias) + '\n')
                    fp.write('Running mean ' + repr(self.running_mean) + '\n')
                    fp.write('Running var ' + repr(self.running_var) + '\n')
                exit()
        except:
            pass


def is_nan_inf(output):
    return (output == float('inf')).any() or (output == float('-inf')).any() or torch.isnan(output).any()

def has_large(input):
    return torch.max(torch.abs(input)) > 1000

