"""Command-line interface for BetterMPNN."""

import argparse
import logging
import os
import random

from .config import Config

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducible runs."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


def main():
    parser = argparse.ArgumentParser(
        description="BetterMPNN: GRPO-based protein sequence optimization",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--pdb", help="Override PDB path")
    parser.add_argument("--chain", help="Override chain ID to design")
    parser.add_argument("--steps", type=int, help="Override training steps")
    parser.add_argument("--variants", type=int, help="Override variants per step")
    parser.add_argument("--iterative", action="store_true", help="Enable iterative backbone update")
    parser.add_argument("--output", help="Override output directory")
    parser.add_argument("--mode", choices=["train", "sample"], help="Override mode (train/sample)")
    parser.add_argument("--step-range", help="Step range for parallel jobs, e.g. '0-4' (0-indexed, inclusive)")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load config
    config = Config.from_yaml(args.config)

    # Apply CLI overrides
    if args.pdb:
        config.pdb = args.pdb
    if args.chain:
        config.chain = args.chain
    if args.steps is not None:
        config.steps = args.steps
    if args.variants is not None:
        config.variants = args.variants
    if args.iterative:
        config.iterative = True
    if args.output:
        config.output_dir = args.output
    if args.mode:
        config.mode = args.mode
    if args.seed is not None:
        config.seed = args.seed

    # Validate
    if not config.pdb:
        parser.error("PDB path is required (via --pdb or config file)")
    if config.steps < 1:
        parser.error(f"steps must be >= 1 (got {config.steps})")
    # GRPO needs at least two variants per step to form a group baseline.
    if config.mode == "train" and config.variants < 2:
        parser.error(f"train mode needs variants >= 2 to compute group-relative advantages (got {config.variants})")

    # Reproducibility
    if config.seed is not None:
        set_seed(config.seed)

    # Build components
    import torch

    from .mpnn import MPNNModel
    from .environment.alphafold3 import AlphaFold3Environment

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    mpnn = MPNNModel.load(config.mpnn_weights, device=device)
    env = AlphaFold3Environment(
        config.environment,
        output_dir=config.output_dir,
        target_smiles=config.effective_target_smiles,
        scaffold_name=config.scaffold.name,
        ligand_name=config.ligand.name,
        design_chain_id=config.design_chain_id,
    )

    if config.mode == "sample":
        # Sampling mode: frozen MPNN + multi-seed AF3 + filtering
        from .rl.sampler import Sampler

        # Parse step range for parallel job array support
        step_range = None
        if args.step_range:
            parts = args.step_range.split("-")
            try:
                start, end = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                parser.error(f"--step-range must be 'START-END' (0-indexed), got {args.step_range!r}")
            if len(parts) != 2 or start < 0 or end < start:
                parser.error(f"--step-range must be 'START-END' with 0 <= START <= END, got {args.step_range!r}")
            step_range = (start, end)
            logger.info(f"Parallel mode: processing steps {step_range[0]}-{step_range[1]}")

        sampler = Sampler(mpnn, env, config)
        sampler.run(step_range=step_range)
    else:
        # Training mode: original GRPO loop
        from .rl.trainer import Trainer
        trainer = Trainer(mpnn, env, config)
        trainer.train()


if __name__ == "__main__":
    main()
