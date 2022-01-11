"""TERMinator models"""
import torch
import torch.nn as nn

from terminator.utils.loop_utils import nlcpl as _nlcpl

from .layers.condense import CondenseTERM
from .layers.energies.gvp import GVPPairEnergies
from .layers.energies.s2s import (AblatedPairEnergies_g, 
                                  MultiChainPairEnergies_g,
                                  PairEnergiesFullGraph)
# pylint: disable=no-member


class TERMinator(nn.Module):
    """TERMinator model for multichain proteins that utilizes contact indices"""
    def __init__(self, hparams, device='cuda:0'):
        """
        Args
        ----
        hparams : dict
            Dictionary of parameter settings (see :code:`scripts/models/train/default_hparams.py`)
        device : str
            Device to place model on
        """
        super().__init__()
        self.dev = device
        self.hparams = hparams
        self.bot = CondenseTERM(hparams, device=self.dev)

        if self.hparams["use_terms"]:
            self.hparams['energies_input_dim'] = self.hparams['term_hidden_dim']
        else:
            self.hparams['energies_input_dim'] = 0

        if hparams['struct2seq_linear']:
            self.top = AblatedPairEnergies_g(hparams).to(self.dev)
        elif hparams['energies_gvp']:
            self.top = GVPPairEnergies(hparams).to(self.dev)
        elif hparams['energies_full_graph']:
            self.top = PairEnergiesFullGraph(hparams).to(self.dev)
        else:
            self.top = MultiChainPairEnergies_g(hparams).to(self.dev)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data, max_seq_len):
        """Compute the Potts model parameters for the structure

        Args
        ----
        data : dict 
            Contains the following keys:
            
            msas : torch.LongTensor
                Integer encoding of sequence matches. Shape: n_batch x n_term_res x n_matches
            features : torch.FloatTensor
                Featurization of match structural data. Shape: n_batch x n_term_res x n_matches x n_features(=9 by default)
            seq_lens : int np.ndarray
                1D Array of batched sequence lengths. Shape: n_batch
            focuses : torch.LongTensor
                Indices for TERM residues matches. Shape: n_batch x n_term_res
            term_lens : int np.ndarray
                2D Array of batched TERM lengths. Shape: n_batch x n_terms
            src_key_mask : torch.ByteTensor
                Mask for TERM residue positions padding. Shape: n_batch x n_term_res
            X : torch.FloatTensor
                Raw coordinates of protein backbones. Shape: n_batch x n_res
            x_mask : torch.ByteTensor
                Mask for X. Shape: n_batch x n_res
            sequence : torch.LongTensor
                Integer encoding of ground truth native sequences. Shape: n_batch x n_res
            max_seq_len : int
                Max length of protein in the batch.
            ppoe : torch.FloatTensor
                Featurization of target protein structural data. Shape: n_batch x n_res x n_features(=9 by default)
            chain_idx : torch.LongTensor
                Integers indices that designate ever residue to a chain.
                Shape: n_batch x n_res
            contact_idx : torch.LongTensor
                Integers representing contact indices across all TERM residues.
                Shape: n_batch x n_term_res

        Returns
        -------
        etab : torch.FloatTensor
            Dense kNN representation of the energy table, with :code:`E_idx`
            denotating which energies correspond to which edge.
            Shape: n_batch x n_res x k(=30 by default) x :code:`hparams['energies_output_dim']` (=400 by default)
        E_idx : torch.LongTensor
            Indices representing edges in the kNN graph.
            Given node `res_idx`, the set of edges centered around that node are
            given by :code:`E_idx[b_idx][res_idx]`, with the `i`-th closest node given by
            :code:`E_idx[b_idx][res_idx][i]`.
            Shape: n_batch x n_res x k(=30 by default)
        """
        if self.hparams['use_terms']:
            node_embeddings, edge_embeddings = self.bot(data, max_seq_len)
        else:
            node_embeddings, edge_embeddings = 0, 0
        etab, E_idx = self.top(node_embeddings, 
                               edge_embeddings, 
                               data['X'], 
                               data['x_mask'], 
                               data['chain_idx'])

        return etab, E_idx
