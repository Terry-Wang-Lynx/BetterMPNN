"""Unified configuration for BetterMPNN."""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
import re


@dataclass
class RewardWeights:
    """Weights for combining structural metrics into a scalar reward."""
    iptm: float = 0.55
    ptm: float = 0.05
    pae: float = 0.4
    clash_penalty: float = 0.1
    open_rmsd: float = 0.0    # Weight for deviating from open state
    closed_rmsd: float = 0.0  # Weight for reaching closed state


@dataclass
class ScaffoldConfig:
    """Scaffold protein configuration."""
    name: str = ""
    sequence: str = ""


@dataclass
class LigandConfig:
    """Target ligand configuration."""
    name: str = ""
    smiles: str = ""


@dataclass
class DesignConfig:
    """Design scope: which residues to redesign.

    Format: ["A10", "A11", ...] where A is chain ID + residue number.
    If empty, the entire masked chain is designed.
    """
    redesign_residues: List[str] = field(default_factory=list)

    def get_redesign_positions(self, chain: str) -> List[int]:
        """Extract 1-based residue numbers for the given chain."""
        positions = []
        for entry in self.redesign_residues:
            m = re.match(r'^([A-Za-z])(\d+)$', entry)
            if m and m.group(1) == chain:
                positions.append(int(m.group(2)))
        return sorted(positions)

    def get_fixed_positions(self, chain: str, chain_length: int) -> List[int]:
        """Return 1-based residue numbers to FIX (inverse of redesign).

        ProteinMPNN's fixed_position_dict expects positions to be FIXED.
        We invert: all positions minus redesign positions = fixed positions.
        """
        if not self.redesign_residues:
            return []  # Empty = design everything
        redesign = set(self.get_redesign_positions(chain))
        return [i for i in range(1, chain_length + 1) if i not in redesign]


@dataclass
class FilterConfig:
    """Filtering criteria for large-scale sampling."""
    iptm_min: float = 0.9            # Hard threshold for binding
    open_rmsd_min: float = 2.0       # Min conformational change RMSD vs open ref (Å)
    closed_rmsd_max: float = 2.5     # Max RMSD vs closed ref (must reach closed state) (Å)
    local_drmsd_max: float = 3.0     # Max local DRMSD for fold integrity (Å)
    ptm_min: float = 0.0             # Optional pTM threshold
    pae_max: float = 31.75           # Optional PAE threshold

    # Decoy / interferent specificity filters
    decoy_iptm_max: float = 0.88     # Decoy iPTM must be below this
    decoy_closed_rmsd_min: float = 3.0 # Decoy must NOT reach closed state (RMSD to closed ref > this)
    decoy_num_seeds: int = 5         # Fewer seeds for decoy evaluation (speed)


@dataclass
class EnvironmentConfig:
    """Configuration for the evaluation environment."""
    type: str = "alphafold3"

    # AlphaFold 3 paths
    af3_sif: str = ""
    af3_run_dir: str = ""
    af3_model_dir: str = ""
    af3_db_dir: str = ""
    # Container runtime: "auto" detects apptainer/singularity on PATH.
    # Set explicitly (e.g. "apptainer" or "singularity") to override.
    apptainer_path: str = "auto"

    # Environment activation script (direct mode, no container)
    af3_env_script: str = ""

    # Template JSON for AF3 input
    template_json: str = ""

    # MSA handling
    run_data_pipeline: bool = False  # False = use pre-computed MSA

    # Index of the designed chain in the template JSON "sequences" list.
    # Default 0 matches the shipped example (binder = chain A = sequences[0])
    # and must point to the same chain as Config.chain.
    design_chain_index: int = 0

    # Reward weights
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


@dataclass
class Config:
    """Top-level training configuration."""

    # Mode: "train" (original GRPO) or "sample" (frozen large-scale sampling)
    mode: str = "train"

    # Input
    pdb: str = ""
    chain: str = "A"  # designed chain; matches environment.design_chain_index=0

    # Scaffold, ligand, and design scope (centralized target config)
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    ligand: LigandConfig = field(default_factory=LigandConfig)
    design: DesignConfig = field(default_factory=DesignConfig)

    # MPNN
    mpnn_weights: str = "weights/vanilla/v_48_020.pt"

    # Training
    steps: int = 10  # conservative default; configs/example.yaml uses 200
    variants: int = 8
    lr: float = 1e-4
    beta: float = 0.01
    temperature: float = 0.3
    grad_clip: float = 1.0
    iterative: bool = False

    # Random seed for reproducibility (None = nondeterministic)
    seed: Optional[int] = None

    # Advantage shaping
    reward_shaping_alpha: Optional[float] = 0.7
    advantage_scale_factor: float = 5.0

    # Checkpointing
    save_every: int = 10  # matches configs/example.yaml
    cleanup_every: int = 0  # 0 = no cleanup

    # Output
    output_dir: str = "output"

    # AF3 seed count (for sampling mode)
    num_seeds: int = 5

    # Reference structures for RMSD calculation (sampling mode)
    closed_ref_pdb: str = ""   # Bound/closed reference PDB (e.g., holo.pdb)
    open_ref_pdb: str = ""     # Apo/open reference PDB (e.g., apo.pdb)
    design_chain_id: str = "A" # Chain ID for Cα extraction in RMSD calc

    # Decoy ligands for specificity testing (SMILES strings)
    decoy_smiles: List[str] = field(default_factory=list)

    # Legacy flat field (kept for backward compat, prefer ligand.smiles)
    target_smiles: str = ""

    # Cleanup: delete AF3 prediction files after extracting results
    cleanup_predictions: bool = True

    # Filter config (sampling mode)
    filter: FilterConfig = field(default_factory=FilterConfig)

    # Environment
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    @property
    def effective_target_smiles(self) -> str:
        """Get target ligand SMILES: prefer ligand.smiles, fallback to target_smiles."""
        return self.ligand.smiles or self.target_smiles

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        env_data = data.pop("environment", {})
        reward_data = env_data.pop("reward_weights", {})
        filter_data = data.pop("filter", {})
        # Map legacy keys
        if "conf_rmsd_min" in filter_data:
            filter_data["open_rmsd_min"] = filter_data.pop("conf_rmsd_min")
        if "decoy_conf_rmsd_max" in filter_data:
            filter_data["decoy_closed_rmsd_min"] = filter_data.pop("decoy_conf_rmsd_max")
        
        scaffold_data = data.pop("scaffold", {})
        ligand_data = data.pop("ligand", {})
        design_data = data.pop("design", {})

        reward_weights = RewardWeights(**reward_data) if reward_data else RewardWeights()
        env_config = EnvironmentConfig(**env_data, reward_weights=reward_weights) if env_data else EnvironmentConfig()
        filter_config = FilterConfig(**filter_data) if filter_data else FilterConfig()
        scaffold_config = ScaffoldConfig(**scaffold_data) if scaffold_data else ScaffoldConfig()
        ligand_config = LigandConfig(**ligand_data) if ligand_data else LigandConfig()
        design_config = DesignConfig(**design_data) if design_data else DesignConfig()

        # Ensure numeric types (YAML may parse 1e-4 as string)
        for key in ("lr", "beta", "temperature", "grad_clip", "reward_shaping_alpha", "advantage_scale_factor"):
            if key in data and data[key] is not None:
                data[key] = float(data[key])
        for key in ("steps", "variants", "save_every", "cleanup_every", "num_seeds", "seed"):
            if key in data and data[key] is not None:
                data[key] = int(data[key])

        return cls(
            **data,
            environment=env_config,
            filter=filter_config,
            scaffold=scaffold_config,
            ligand=ligand_config,
            design=design_config,
        )

    def to_yaml(self, path: str) -> None:
        """Save config to a YAML file."""
        import dataclasses
        data = dataclasses.asdict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def resolve_paths(self, base_dir: str = "") -> None:
        """Resolve relative paths against a base directory."""
        if not base_dir:
            return
        for attr in ("pdb", "mpnn_weights", "output_dir"):
            val = getattr(self, attr)
            if val and not os.path.isabs(val):
                setattr(self, attr, os.path.join(base_dir, val))
        for attr in ("af3_sif", "af3_run_dir", "af3_model_dir", "af3_db_dir", "template_json"):
            val = getattr(self.environment, attr)
            if val and not os.path.isabs(val):
                setattr(self.environment, attr, os.path.join(base_dir, val))
