""" Ingraham Geometric Layers

Ingraham Structure MPNN, but implemented in Torch Geometric
"""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from terminator.models.layers.s2s_modules import Normalize
from ..utils import merge_duplicate_edges_geometric, merge_duplicate_pairE_geometric
# pylint: disable=no-member


# pylint: disable=abstract-method
class NodeConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self,
                 in_dims,
                 out_dims,
                 edge_dims,
                 n_layers=3,
                 module_list=None,
                 aggr="mean"):
        super().__init__(aggr=aggr)
        self.si = in_dims
        self.so = out_dims
        self.se = edge_dims

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    nn.Linear(2 * self.si + self.se, self.so))
            else:
                module_list.append(nn.Linear(2 * self.si + self.se, self.so))
                module_list.append(nn.ReLU())
                for _ in range(n_layers - 2):
                    module_list.append(nn.Linear(self.so, self.so))
                    module_list.append(nn.ReLU())
                module_list.append(nn.Linear(self.so, self.se))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, h_V, edge_index, edge_attr):
        '''
        :param h_V: `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: `torch.Tensor`
        '''
        return self.propagate(edge_index, h_V=h_V, edge_attr=edge_attr)

    # pylint: disable=arguments-differ
    def message(self, h_V_i, h_V_j, edge_attr):
        message = torch.cat([h_V_i, edge_attr, h_V_j], dim=-1)
        return self.message_func(message)


class NodeMPNNLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self,
                 node_dims,
                 edge_dims,
                 n_message=3,
                 n_feedforward=2,
                 drop_rate=.1):
        super().__init__()
        self.conv = NodeConv(node_dims,
                             node_dims,
                             edge_dims,
                             n_message,
                             aggr="mean")
        self.norm = nn.ModuleList([Normalize(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([nn.Dropout(p=drop_rate) for _ in range(2)])

        # FFN
        n_feedforward = 2
        ff_func = []
        if n_feedforward == 1:
            ff_func.append(nn.Linear(node_dims, node_dims))
        else:
            hid_dims = 2 * node_dims
            ff_func.append(nn.Linear(node_dims, hid_dims))
            ff_func.append(nn.ReLU())
            for _ in range(n_feedforward - 2):
                ff_func.append(nn.Linear(hid_dims, hid_dims))
                ff_func.append(nn.ReLU())
            ff_func.append(nn.Linear(hid_dims, node_dims))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`.
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node
                embeddings `x` will still be the base of the update and the
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = x[node_mask], dh[node_mask]

        x = self.norm[0](x + self.dropout[0](dh))

        dh = self.ff_func(x)
        x = self.norm[1](x + self.dropout[1](dh))

        if node_mask is not None:
            x_[node_mask] = x
            x = x_
        return x


class EdgeMPNNLayer(nn.Module):
    """ Torch Geometric version of Edge MPNN Layer (see :code:``)"""
    def __init__(self,
                 node_dims,
                 edge_dims,
                 drop_rate=0.1,
                 n_layers=3,
                 module_list=None):
        """ TODO """
        super().__init__()

        # Edge Messages
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    nn.Linear(2 * node_dims + edge_dims, edge_dims))
            else:
                module_list.append(nn.Linear(2 * node_dims + edge_dims, edge_dims))
                for _ in range(n_layers - 2):
                    module_list.append(nn.Linear(edge_dims, edge_dims))
                    module_list.append(nn.ReLU())
                module_list.append(nn.Linear(edge_dims, edge_dims))
        self.message_func = nn.Sequential(*module_list)

        # norm and dropout
        self.norm = nn.ModuleList([Normalize(edge_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([nn.Dropout(p=drop_rate) for _ in range(2)])

        # FFN
        n_feedforward = 2
        ff_func = []
        if n_feedforward == 1:
            ff_func.append(nn.Linear(edge_dims, edge_dims))
        else:
            hid_dims = 2 * edge_dims
            ff_func.append(nn.Linear(edge_dims, hid_dims))
            for _ in range(n_feedforward - 2):
                ff_func.append(nn.Linear(hid_dims, hid_dims))
                ff_func.append(nn.ReLU())
            ff_func.append(nn.Linear(hid_dims, edge_dims))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, h_V, edge_index, h_E, node_mask=None):
        """ TODO """
        h_dim = h_V.shape[-1]
        edge_index_flat = edge_index.flatten()
        edge_index_flat = edge_index_flat.unsqueeze(-1).expand([-1, h_dim])
        h_V_gather = torch.gather(h_V, -2, edge_index_flat)
        h_V_ij = h_V_gather.view(list(edge_index.shape) + [h_dim])

        h_V_i, h_V_j = torch.unbind(h_V_ij)

        h_EV = torch.cat([h_V_i, h_E, h_V_j], dim=-1)
        dh = self.message_func(h_EV)
        # merge duplicate edges
        dh = merge_duplicate_edges_geometric(dh, edge_index)

        x = h_E
        if node_mask is not None:
            x_ = x
            x, dh = x[node_mask], dh[node_mask]

        x = self.norm[0](x + self.dropout[0](dh))

        dh = self.ff_func(x)
        x = self.norm[1](x + self.dropout[1](dh))

        if node_mask is not None:
            x_[node_mask] = x
            x = x_
        return x


class GeometricPairEnergies(nn.Module):
    '''GNN Potts Model Encoder using Torch Geometric

    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (39, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP layers
    :param num_layers: number of GVP layers in each of the encoder
                       and decoder modules
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        node_in_dim = hparams['energies_input_dim'] + 6
        node_h_dim = 128
        edge_in_dim = hparams['energies_input_dim'] + 39
        edge_h_dim = 128
        num_layers = hparams['energies_encoder_layers']
        drop_rate = hparams['energies_dropout']
        output_dim = hparams['energies_output_dim']

        self.W_v = nn.Sequential(nn.Linear(node_in_dim, node_h_dim), nn.LayerNorm(node_h_dim))
        self.W_e = nn.Sequential(nn.Linear(edge_in_dim, edge_h_dim), nn.LayerNorm(edge_h_dim))

        self.node_encoder_layers = nn.ModuleList(
            NodeMPNNLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))
        self.edge_encoder_layers = nn.ModuleList(
            EdgeMPNNLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        self.W_out = nn.Linear(edge_h_dim, output_dim)

        # if enabled, generate self energies in etab from node embeddings
        if "node_self_sub" in hparams.keys() and hparams["node_self_sub"] is True:
            self.W_proj = nn.Linear(node_h_dim, 20)

    def forward(self, h_V, edge_index, h_E):
        '''Forward pass to be used at train-time, or evaluating likelihood.

        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        '''
        local_dev = self.W_v[0].weight.device  # slightly hacky way of getting current device
        h_V = h_V.to(local_dev)
        h_E = h_E.to(local_dev)
        edge_index = edge_index.to(local_dev)

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        for node_layer, edge_layer in zip(self.node_encoder_layers, self.edge_encoder_layers):
            h_V = node_layer(h_V, edge_index, h_E)
            h_E = edge_layer(h_V, edge_index, h_E)

        etab = self.W_out(h_E)
        # merge pairE tables to ensure each edge has the same energy dist
        etab = merge_duplicate_pairE_geometric(etab, edge_index)
        self_edge_select = (edge_index[0] == edge_index[1])
        # zero out off-diag energies for self edges
        etab[self_edge_select] = etab[self_edge_select] * torch.eye(20).view(1, -1).to(etab.device)

        # if specified, use generate self energies from node embeddings
        if "node_self_sub" in self.hparams.keys() and self.hparams["node_self_sub"] is True:
            h_V = self.W_proj(h_V)
            etab_select = etab[(edge_index[0] == edge_index[1])]
            etab[edge_index[0] == edge_index[1]] = torch.diag_embed(h_V, dim1=-2, dim2=-1).view(etab_select.shape)

        return etab, edge_index
