"""ProteinMPNN model wrapper: load, sample, compute log-probabilities."""

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Import from the bundled ProteinMPNN code
from .protein_mpnn_utils import (
    parse_PDB,
    StructureDatasetPDB,
    _S_to_seq,
    tied_featurize,
    ProteinMPNN,
)


@dataclass
class SampleResult:
    """Result from MPNN sequence sampling."""
    sequences: list[str]
    X: torch.Tensor
    S: torch.Tensor
    S_sample: torch.Tensor
    mask: torch.Tensor
    chain_M: torch.Tensor
    chain_M_pos: torch.Tensor
    residue_idx: torch.Tensor
    chain_encoding_all: torch.Tensor
    randn: torch.Tensor
    decoding_order: torch.Tensor


class MPNNModel:
    """Wrapper around ProteinMPNN for sampling and log-probability computation."""

    def __init__(self, model: ProteinMPNN, device: str, num_edges: int):
        self.model = model
        self.device = device
        self.num_edges = num_edges

    @classmethod
    def load(cls, weights_path: str, device: str = "cuda") -> "MPNNModel":
        """Load a ProteinMPNN model from checkpoint."""
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        num_edges = checkpoint["num_edges"]

        model = ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0,
            k_neighbors=num_edges,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.train()
        logger.info(f"Loaded ProteinMPNN from {weights_path} (k={num_edges})")
        return cls(model, device, num_edges)

    def copy_frozen(self) -> "MPNNModel":
        """Create a frozen copy for use as reference model."""
        ref = copy.deepcopy(self.model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        return MPNNModel(ref, self.device, self.num_edges)

    def sample(
        self,
        pdb_path: str,
        chain: str,
        n: int,
        temperature: float = 0.3,
        design_positions: list[int] = None,
    ) -> SampleResult:
        """Sample n sequence variants for the specified chain.
        If design_positions is provided (1-based indices), only those positions will be designed.
        """
        pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
        chain_id_dict = {pdb_dict_list[0]["name"]: ([chain], [])}
        dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=200000)
        batch_clones = [copy.deepcopy(dataset[0]) for _ in range(n)]

        fixed_position_dict = None
        if design_positions:
            seq_len = len(pdb_dict_list[0].get(f"seq_chain_{chain}", ""))
            fixed_pos = [i for i in range(1, seq_len + 1) if i not in design_positions]
            fixed_position_dict = {pdb_dict_list[0]["name"]: {chain: fixed_pos}}
            logger.info(f"Designing {len(design_positions)} residues out of {seq_len} in chain {chain}")

        feats = tied_featurize(
            batch_clones, self.device, chain_id_dict,
            fixed_position_dict=fixed_position_dict,
        )
        (X, S, mask, lengths, chain_M, chain_encoding_all,
         chain_list_list, visible_list_list, masked_list_list,
         masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
         residue_idx, dihedral_mask, tied_pos_list_of_lists_list,
         pssm_coef, pssm_bias, pssm_log_odds_all,
         bias_by_res_all, tied_beta) = feats

        with torch.no_grad():
            randn = torch.randn(chain_M.shape, device=X.device)
            sample_dict = self.model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask, temperature=temperature, chain_M_pos=chain_M_pos,
                omit_AAs_np=np.zeros(21), bias_AAs_np=np.zeros(21),
                omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
                pssm_bias=pssm_bias, bias_by_res=bias_by_res_all,
            )

        S_sample = sample_dict["S"]
        sequences = [_S_to_seq(S_sample[i], chain_M[i]) for i in range(n)]

        return SampleResult(
            sequences=sequences,
            X=X, S=S, S_sample=S_sample, mask=mask,
            chain_M=chain_M, chain_M_pos=chain_M_pos,
            residue_idx=residue_idx, chain_encoding_all=chain_encoding_all,
            randn=randn, decoding_order=sample_dict["decoding_order"],
        )

    def compute_log_probs(self, sr: SampleResult) -> torch.Tensor:
        """Compute per-token log-probabilities for sampled sequences.

        Returns:
            Tensor of shape [batch, seq_len] with log-probs of sampled tokens.
        """
        logps_dist = self.model(
            sr.X, sr.S_sample, sr.mask,
            sr.chain_M * sr.chain_M_pos,
            sr.residue_idx, sr.chain_encoding_all, sr.randn,
            use_input_decoding_order=True,
            decoding_order=sr.decoding_order,
        )
        return torch.gather(logps_dist, 2, sr.S_sample.unsqueeze(-1)).squeeze(-1)

    def loss_mask(self, sr: SampleResult) -> torch.Tensor:
        """Get the mask for loss computation (designed positions only)."""
        return sr.mask * sr.chain_M * sr.chain_M_pos

    def save(self, path: str, extra: dict = None) -> None:
        """Save model checkpoint."""
        data = {
            "model_state_dict": self.model.state_dict(),
            "num_edges": self.num_edges,
        }
        if extra:
            data.update(extra)
        torch.save(data, path)
        logger.info(f"Model saved to {path}")

    def get_chain_ids(self, pdb_path: str) -> list[str]:
        """Parse PDB and return ordered chain IDs."""
        pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
        return [
            k.replace("seq_chain_", "")
            for k in pdb_dict_list[0]
            if k.startswith("seq_chain_")
        ]
