"""TERMinator models"""
import torch
from torch import nn

from terminator.utils.loop_utils import nlcpl as _nlcpl

from .layers.condense import CondenseMSA, MultiChainCondenseMSA_g
from .layers.energies.gvp import GVPPairEnergies
from .layers.energies.s2s import (AblatedPairEnergies, AblatedPairEnergies_g, MultiChainPairEnergies_g, PairEnergies,
                                  PairEnergiesFullGraph)
# pylint: disable=no-member, not-callable, arguments-differ


class TERMinator(nn.Module):
    """Barebone TERMinator model for single-chain proteins.

    Attributes
    ----------
    dev: str
        Device representing where the model is held
    hparams: dict
        Dictionary of parameter settings (see :code:`scripts/models/train/default_hparams.py`)
    bot: CondenseMSA
        TERM information condenser network
    top: PairEnergies (or appropriate variant thereof)
        GNN Potts Model Encoder network"""
    def __init__(self, hparams, device='cuda:0'):
        """
        Initializes TERMinator according to given parameters.

        Also prints out the hidden dimensionality of the TERM information condenser
        and that of the GNN Potts Model Encoder.

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
        self.bot = CondenseMSA(hparams=self.hparams, device=self.dev)

        if self.hparams["use_terms"]:
            self.hparams['energies_input_dim'] = self.hparams['term_hidden_dim']
        else:
            self.hparams['energies_input_dim'] = 0

        if self.hparams['struct2seq_linear']:
            self.top = AblatedPairEnergies(hparams=self.hparams).to(self.dev)
        else:
            self.top = PairEnergies(hparams=self.hparams).to(self.dev)

        print(
            f'TERM information condenser hidden dimensionality is {self.bot.hparams["term_hidden_dim"]} and GNN Potts Model Encoder hidden dimensionality is {self.top.hparams["energies_hidden_dim"]}'
        )

        self.prior = torch.zeros(20).view(1, 1, 20).to(self.dev)

    def forward(self, msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, sequence, max_seq_len,
                ppoe):
        """ Compute the negative composite psuedolikelihood of sequences given featurized structures.

        See :code:`terminator.utils.loop_utils` for a description of the negative composite
        psuedolikelihood computation.

        Args
        ----
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

        Returns
        -------
        nlcpl : torch.FloatTensor
            Negative composite psuedolikelihood of sequences given the produced Potts model.
            Shape: 1
        avg_prob : torch.FloatTensor
            Mean probability of pairs of native residues across every edge in the kNN graph
            given the Potts model. Averaged across all edges across all proteins in the batch.
            Shape: 1
        counter : torch.FloatTensor
            Total number of edges in the batch.
            Shape: 1
        """
        etab, E_idx = self.potts(msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len,
                                 ppoe)
        nlcpl, avg_prob, counter = _nlcpl(etab, E_idx, sequence, x_mask)
        return nlcpl, avg_prob, counter

    def potts(self, msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe):
        """Compute the Potts model parameters for the structure.

        Runs the full TERMinator network for prediction.

        Args
        ----
        msas : torch.LongTensor
            Integer encoding of sequence matches.
            Shape: n_batch x n_term_res x n_matches
        features : torch.FloatTensor
            Featurization of match structural data.
            Shape: n_batch x n_term_res x n_matches x n_features(=9 by default)
        seq_lens : int np.ndarray
            1D Array of batched sequence lengths.
            Shape: n_batch
        focuses : torch.LongTensor
            Indices for TERM residues matches.
            Shape: n_batch x n_term_res
        term_lens : int np.ndarray
            2D Array of batched TERM lengths.
            Shape: n_batch x n_terms
        src_key_mask : torch.ByteTensor
            Mask for TERM residue positions padding.
            Shape: n_batch x n_term_res
        X : torch.FloatTensor
            Raw coordinates of protein backbones.
            Shape: n_batch x n_res
        x_mask : torch.ByteTensor
            Mask for X.
            Shape: n_batch x n_res
        max_seq_len : int
            Max length of protein in the batch
        ppoe : torch.FloatTensor
            Featurization of target protein structural data.
            Shape: n_batch x n_res x n_features(=9 by default)

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
            condense = self.bot(msas, features, seq_lens, focuses, term_lens, src_key_mask, max_seq_len, ppoe)
            etab, E_idx = self.top(X, x_mask, V_embed=condense)
        else:
            etab, E_idx = self.top(X, x_mask)
        return etab, E_idx


class MultiChainTERMinator_gcnkt(TERMinator):
    """TERMinator model for multichain proteins that utilizes contact indices

    Attributes
    ----------
    dev: str
        Device representing where the model is held
    hparams: dict
        Dictionary of parameter settings (see :code:`scripts/models/train/default_hparams.py`)
    bot: MultiChainCondenseMSA_g
        TERM information condenser network
    top: PairEnergies (or appropriate variant thereof)
        GNN Potts Model Encoder network
    """
    def __init__(self, hparams, device='cuda:0'):
        """
        Initializes TERMinator according to given parameters.

        Args
        ----
        hparams : dict
            Dictionary of parameter settings (see :code:`scripts/models/train/default_hparams.py`)
        device : str
            Device to place model on
        """
        super().__init__(hparams, device)
        self.dev = device
        self.hparams = hparams
        self.bot = MultiChainCondenseMSA_g(hparams, device=self.dev)

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

    def forward(self, msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, sequence, max_seq_len,
                ppoe, chain_idx, contact_idx):
        """ Compute the negative composite psuedolikelihood of sequences given featurized structures

        See :code:`terminator.utils.loop_utils` for a description of the negative composite
        psuedolikelihood computation.

        Args
        ----
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
        nlcpl : torch.FloatTensor
            Negative composite psuedolikelihood of sequences given the produced Potts model.
            Shape: 1
        avg_prob : torch.FloatTensor
            Mean probability of pairs of native residues across every edge in the kNN graph
            given the Potts model. Averaged across all edges across all proteins in the batch.
            Shape: 1
        counter : torch.FloatTensor
            Total number of edges in the batch.
            Shape: 1
        """
        etab, E_idx = self.potts(msas,
                                 features,
                                 seq_lens,
                                 focuses,
                                 term_lens,
                                 src_key_mask,
                                 X,
                                 x_mask,
                                 max_seq_len,
                                 ppoe,
                                 chain_idx,
                                 contact_idx=contact_idx)
        nlcpl, avg_prob, counter = _nlcpl(etab, E_idx, sequence, x_mask)
        return nlcpl, avg_prob, counter

    def potts(self, msas, features, seq_lens, focuses, term_lens, src_key_mask, X, x_mask, max_seq_len, ppoe,
              chain_idx, contact_idx):
        """Compute the Potts model parameters for the structure

        Runs the full TERMinator network for prediction.

        Args
        ----
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
            node_embeddings, edge_embeddings = self.bot(msas,
                                                        features,
                                                        seq_lens,
                                                        focuses,
                                                        term_lens,
                                                        src_key_mask,
                                                        max_seq_len,
                                                        chain_idx,
                                                        X,
                                                        ppoe,
                                                        contact_idx=contact_idx)
        else:
            node_embeddings, edge_embeddings = 0, 0
        etab, E_idx = self.top(node_embeddings, edge_embeddings, X, x_mask, chain_idx)

        return etab, E_idx
