""" GNN Potts Model Encoder modules

This file contains the GNN Potts Model Encoder, as well as an ablated version of
itself. """
from __future__ import print_function

import torch
from torch import nn

from terminator.models.layers.graph_features import MultiChainProteinFeatures
from terminator.models.layers.s2s_modules import (EdgeMPNNLayer,
                                                  EdgeTransformerLayer,
                                                  NodeMPNNLayer,
                                                  NodeTransformerLayer)
from terminator.models.layers.utils import (cat_edge_endpoints,
                                            cat_neighbors_nodes, gather_edges,
                                            gather_nodes,
                                            merge_duplicate_pairE)

# pylint: disable=no-member, not-callable


class AblatedPairEnergies(nn.Module):
    """Ablated GNN Potts Model Encoder

    Attributes
    ----------
    dev: str
        Device representing where the model is held
    hparams: dict
        Dictionary of parameter settings (see :code:`scripts/models/train/default_hparams.py`)
    features : MultiChainProteinFeatures
        Module that featurizes a protein backbone (including multimeric proteins)
    W : nn.Linear
        Output layer that projects edge embeddings to proper output dimensionality
    """
    def __init__(self, hparams):
        """ Graph labeling network """
        super().__init__()
        hdim = hparams['energies_hidden_dim']
        self.hparams = hparams

        # Featurization layers
        self.features = MultiChainProteinFeatures(node_features=hdim,
                                                  edge_features=hdim,
                                                  top_k=hparams['k_neighbors'],
                                                  features_type=hparams['energies_protein_features'],
                                                  augment_eps=hparams['energies_augment_eps'],
                                                  dropout=hparams['energies_dropout'])

        self.W = nn.Linear(hparams['energies_input_dim'] * 3, hparams['energies_output_dim'])

    def forward(self, V_embed, E_embed, X, x_mask, chain_idx):
        """ Create kNN etab from TERM features, then project to proper output dimensionality.

        Args
        ----
        V_embed : torch.Tensor
            TERM node embeddings
            Shape: n_batch x n_res x n_hidden
        E_embed : torch.Tensor
            TERM edge embeddings
            Shape : n_batch x n_res x n_res x n_hidden
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x n_res x 4 x 3
        x_mask : torch.ByteTensor
            Mask for X.
            Shape: n_batch x n_res
        chain_idx : torch.LongTensor
            Indices such that each chain is assigned a unique integer and each residue in that chain
            is assigned that integer.
            Shape: n_batch x n_res

        Returns
        -------
        etab : torch.Tensor
            Energy table in kNN dense form
            Shape: n_batch x n_res x k x n_hidden
        E_idx : torch.LongTensor
            Edge index for `etab`
            Shape: n_batch x n_res x k
        """
        # compute the kNN etab
        _, _, E_idx = self.features(X, chain_idx, x_mask)  # notably, we throw away the backbone features
        E_embed_neighbors = gather_edges(E_embed, E_idx)
        h_E = cat_edge_endpoints(E_embed_neighbors, V_embed, E_idx)
        etab = self.W(h_E)

        # merge duplicate pairEs
        n_batch, n_res, k, out_dim = h_E.shape
        etab = etab.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
        etab = merge_duplicate_pairE(etab, E_idx)
        etab = etab.view(n_batch, n_res, k, out_dim)

        return etab, E_idx


class PairEnergies(nn.Module):
    """GNN Potts Model Encoder

    Attributes
    ----------
    dev: str
        Device representing where the model is held
    hparams: dict
        Dictionary of parameter settings (see :code:`scripts/models/train/default_hparams.py`)
    features : MultiChainProteinFeatures
        Module that featurizes a protein backbone (including multimeric proteins)
    W_v : nn.Linear
        Embedding layer for incoming TERM node embeddings
    W_e : nn.Linear
        Embedding layer for incoming TERM edge embeddings
    edge_encoder : nn.ModuleList of EdgeTransformerLayer or EdgeMPNNLayer
        Edge graph update layers
    node_encoder : nn.ModuleList of NodeTransformerLayer or NodeMPNNLayer
        Node graph update layers
    W_out : nn.Linear
        Output layer that projects edge embeddings to proper output dimensionality
    W_proj : nn.Linear (optional)
        Output layer that projects node embeddings to proper output dimensionality.
        Enabled when :code:`hparams["node_self_sub"]=True`
    """
    def __init__(self, hparams):
        """ Graph labeling network """
        super().__init__()
        self.hparams = hparams

        hdim = hparams['energies_hidden_dim']

        # Hyperparameters
        self.node_features = hdim
        self.edge_features = hdim
        self.input_dim = hdim
        hidden_dim = hdim
        output_dim = hparams['energies_output_dim']
        dropout = hparams['energies_dropout']
        num_encoder_layers = hparams['energies_encoder_layers']

        # Featurization layers
        self.features = MultiChainProteinFeatures(node_features=hdim,
                                                  edge_features=hdim,
                                                  top_k=hparams['k_neighbors'],
                                                  features_type=hparams['energies_protein_features'],
                                                  augment_eps=hparams['energies_augment_eps'],
                                                  dropout=hparams['energies_dropout'])

        # Embedding layers
        self.W_v = nn.Linear(hdim + hparams['energies_input_dim'], hdim, bias=True)
        self.W_e = nn.Linear(hdim + hparams['energies_input_dim'], hdim, bias=True)
        edge_layer = EdgeTransformerLayer if not hparams['energies_use_mpnn'] else EdgeMPNNLayer
        node_layer = NodeTransformerLayer if not hparams['energies_use_mpnn'] else NodeMPNNLayer

        # Encoder layers
        self.edge_encoder = nn.ModuleList(
            [edge_layer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_encoder_layers)])
        self.node_encoder = nn.ModuleList(
            [node_layer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers)])

        # if enabled, generate self energies in etab from node embeddings
        if "node_self_sub" in hparams.keys() and hparams["node_self_sub"] is True:
            self.W_proj = nn.Linear(hidden_dim, 20)

        # project edges to proper output dimensionality
        self.W_out = nn.Linear(hidden_dim, output_dim, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V_embed, E_embed, X, x_mask, chain_idx):
        """ Create kNN etab from backbone and TERM features, then project to proper output dimensionality.

        Args
        ----
        V_embed : torch.Tensor or None
            TERM node embeddings. None only accepted if :code:`hparams['energies_input_dim']=0`.
            Shape: n_batch x n_res x n_hidden
        E_embed : torch.Tensor or None
            TERM edge embeddings. None only accepted if :code:`hparams['energies_input_dim']=0`.
            Shape : n_batch x n_res x n_res x n_hidden
        X : torch.Tensor
            Backbone coordinates
            Shape: n_batch x n_res x 4 x 3
        x_mask : torch.ByteTensor
            Mask for X.
            Shape: n_batch x n_res
        chain_idx : torch.LongTensor
            Indices such that each chain is assigned a unique integer and each residue in that chain
            is assigned that integer.
            Shape: n_batch x n_res

        Returns
        -------
        etab : torch.Tensor
            Energy table in kNN dense form
            Shape: n_batch x n_res x k x n_hidden
        E_idx : torch.LongTensor
            Edge index for `etab`
            Shape: n_batch x n_res x k
        """
        # Prepare node and edge embeddings
        if self.hparams['energies_input_dim'] != 0:
            V, E, E_idx = self.features(X, chain_idx, x_mask)
            if not self.hparams['use_coords']:  # this is hacky/inefficient but i am lazy
                V = torch.zeros_like(V)
                E = torch.zeros_like(E)
            # fuse backbone and TERM embeddings
            h_V = self.W_v(torch.cat([V, V_embed], dim=-1))
            E_embed_neighbors = gather_edges(E_embed, E_idx)
            h_E = self.W_e(torch.cat([E, E_embed_neighbors], dim=-1))
        else:
            # just use backbone features
            V, E, E_idx = self.features(X, chain_idx, x_mask)
            h_V = self.W_v(V)
            h_E = self.W_e(E)

        # Graph updates
        mask_attend = gather_nodes(x_mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = x_mask.unsqueeze(-1) * mask_attend
        for edge_layer, node_layer in zip(self.edge_encoder, self.node_encoder):
            h_EV_edges = cat_edge_endpoints(h_E, h_V, E_idx)
            h_E = edge_layer(h_E, h_EV_edges, E_idx, mask_E=x_mask, mask_attend=mask_attend)

            h_EV_nodes = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = node_layer(h_V, h_EV_nodes, mask_V=x_mask, mask_attend=mask_attend)

        # project to output and merge duplicate pairEs
        h_E = self.W_out(h_E)
        n_batch, n_res, k, out_dim = h_E.shape
        h_E = h_E.unsqueeze(-1).view(n_batch, n_res, k, 20, 20)
        h_E = merge_duplicate_pairE(h_E, E_idx)

        # if specified, use generate self energies from node embeddings
        if "node_self_sub" in self.hparams.keys() and self.hparams["node_self_sub"] is True:
            h_V = self.W_proj(h_V)
            h_E[..., 0, :, :] = torch.diag_embed(h_V, dim1=-2, dim2=-1)

        # reshape to fit kNN output format
        h_E = h_E.view(n_batch, n_res, k, out_dim)
        return h_E, E_idx
