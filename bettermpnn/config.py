"""Unified configuration for BetterMPNN."""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardWeights:
    """Weights for combining structural metrics into a scalar reward."""
    iptm: float = 0.55
    ptm: float = 0.05
    pae: float = 0.4
    clash_penalty: float = 0.1


@dataclass
class EnvironmentConfig:
    """Configuration for the evaluation environment."""
    type: str = "alphafold3"

    # AlphaFold 3 paths
    af3_sif: str = ""
    af3_run_dir: str = ""
    af3_model_dir: str = ""
    af3_db_dir: str = ""
    apptainer_path: str = "singularity"

    # Environment activation script (direct mode, no container)
    af3_env_script: str = ""

    # Template JSON for AF3 input
    template_json: str = ""

    # MSA handling
    run_data_pipeline: bool = False  # False = use pre-computed MSA

    # Index of the designed chain in the JSON sequences list
    design_chain_index: int = 1

    # Reward weights
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


@dataclass
class Config:
    """Top-level training configuration."""

    # Input
    pdb: str = ""
    chain: str = "B"

    # MPNN
    mpnn_weights: str = "weights/vanilla/v_48_020.pt"

    # Training
    steps: int = 10
    variants: int = 8
    lr: float = 1e-4
    beta: float = 0.01
    temperature: float = 0.3
    grad_clip: float = 1.0
    iterative: bool = False

    # Advantage shaping
    reward_shaping_alpha: Optional[float] = 0.7
    advantage_scale_factor: float = 5.0

    # Checkpointing
    save_every: int = 15
    cleanup_every: int = 0  # 0 = no cleanup

    # Output
    output_dir: str = "output"

    # Environment
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        env_data = data.pop("environment", {})
        reward_data = env_data.pop("reward_weights", {})

        reward_weights = RewardWeights(**reward_data) if reward_data else RewardWeights()
        env_config = EnvironmentConfig(**env_data, reward_weights=reward_weights) if env_data else EnvironmentConfig()

        # Ensure numeric types (YAML may parse 1e-4 as string)
        for key in ("lr", "beta", "temperature", "grad_clip", "reward_shaping_alpha", "advantage_scale_factor"):
            if key in data and data[key] is not None:
                data[key] = float(data[key])
        for key in ("steps", "variants", "save_every", "cleanup_every"):
            if key in data and data[key] is not None:
                data[key] = int(data[key])

        return cls(**data, environment=env_config)

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
