"""Command-line interface for BetterMPNN."""

import argparse
import logging
import sys

from .config import Config


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
    if args.steps:
        config.steps = args.steps
    if args.variants:
        config.variants = args.variants
    if args.iterative:
        config.iterative = True
    if args.output:
        config.output_dir = args.output

    # Validate
    if not config.pdb:
        parser.error("PDB path is required (via --pdb or config file)")

    # Build components
    from .mpnn import MPNNModel
    from .environment.alphafold3 import AlphaFold3Environment
    from .rl.trainer import Trainer

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    logging.getLogger(__name__).info(f"Device: {device}")

    mpnn = MPNNModel.load(config.mpnn_weights, device=device)
    env = AlphaFold3Environment(config.environment, output_dir=config.output_dir)
    trainer = Trainer(mpnn, env, config)
    trainer.train()


if __name__ == "__main__":
    main()
