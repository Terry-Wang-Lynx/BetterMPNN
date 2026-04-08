"""Training loop: orchestrates MPNN sampling, environment evaluation, and GRPO updates."""

import csv
import json
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim

from ..config import Config
from ..environment.base import Environment
from ..mpnn.wrapper import MPNNModel
from ..utils.plotting import plot_training_curves
from .grpo import compute_advantages, compute_grpo_loss

logger = logging.getLogger(__name__)


class Trainer:
    """GRPO trainer for ProteinMPNN fine-tuning."""

    def __init__(self, mpnn: MPNNModel, environment: Environment, config: Config):
        self.mpnn = mpnn
        self.ref_mpnn = mpnn.copy_frozen()
        self.env = environment
        self.config = config
        self.device = mpnn.device

        self.optimizer = optim.Adam(mpnn.model.parameters(), lr=config.lr)
        self.current_pdb = config.pdb

        os.makedirs(config.output_dir, exist_ok=True)
        self._init_csv_log()

    def _init_csv_log(self):
        """Initialize the per-variant CSV log file."""
        self.csv_path = os.path.join(self.config.output_dir, "variant_log.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "variant", "score", "sequence",
                    "iptm", "ptm", "mean_pae", "has_clash", "ranking_score",
                    "timestamp",
                ])

    def _log_variant(self, step: int, variant: int, score: float,
                     sequence: str, metrics: dict):
        """Append one variant record to the CSV log."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step + 1, variant, f"{score:.6f}", sequence,
                metrics.get("iptm", ""),
                metrics.get("ptm", ""),
                metrics.get("mean_pae", ""),
                metrics.get("has_clash", ""),
                metrics.get("ranking_score", ""),
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ])

    def train(self) -> list[dict]:
        """Run the full training loop."""
        cfg = self.config
        results = []
        start_time = time.time()

        logger.info(f"Starting GRPO training: {cfg.steps} steps, {cfg.variants} variants/step")
        logger.info(f"Backbone: {self.current_pdb}, Chain: {cfg.chain}, Iterative: {cfg.iterative}")

        for step in range(cfg.steps):
            step_start = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {step + 1}/{cfg.steps}")
            logger.info(f"{'='*60}")

            step_result = self._train_step(step)
            step_result["time"] = time.time() - step_start
            results.append(step_result)

            logger.info(
                f"  reward={step_result['mean_reward']:.4f}  "
                f"loss={step_result['loss']:.4f}  "
                f"kl={step_result['kl']:.6f}  "
                f"time={step_result['time']:.1f}s"
            )

            # Incremental save
            self._save_results(results)
            plot_training_curves(results, cfg.output_dir)

            # Checkpoint
            if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"mpnn_step_{step + 1}.pt")
                self.mpnn.save(ckpt_path, extra={"step": step + 1, "results": results})

        # Final save
        total_time = time.time() - start_time
        logger.info(f"\nTraining complete. Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info(f"  Initial reward: {results[0]['mean_reward']:.4f}")
        logger.info(f"  Final reward:   {results[-1]['mean_reward']:.4f}")

        final_path = os.path.join(cfg.output_dir, "mpnn_final.pt")
        self.mpnn.save(final_path, extra={"results": results})
        self._save_results(results)

        return results

    def _train_step(self, step: int) -> dict:
        """Execute one training step: sample -> evaluate -> update."""
        cfg = self.config

        # --- Sample sequences ---
        samples = self.mpnn.sample(
            self.current_pdb, cfg.chain, cfg.variants, cfg.temperature,
        )

        # --- Evaluate with environment ---
        results = self.env.evaluate_batch(samples.sequences, step=step)
        metrics_acc = defaultdict(list)

        for i, (seq, res) in enumerate(zip(samples.sequences, results)):
            logger.info(f"  variant {i}: score={res.score:.4f}  {res.metrics}")
            # Log every variant to CSV
            self._log_variant(step, i, res.score, seq, res.metrics)
            for k, v in res.metrics.items():
                if isinstance(v, (int, float)):
                    metrics_acc[k].append(v)

        rewards = torch.tensor([r.score for r in results], device=self.device)

        # --- GRPO update ---
        advantages = compute_advantages(
            rewards,
            shaping_alpha=cfg.reward_shaping_alpha,
            scale_factor=cfg.advantage_scale_factor,
        )

        current_logps = self.mpnn.compute_log_probs(samples)
        with torch.no_grad():
            ref_logps = self.ref_mpnn.compute_log_probs(samples)

        mask = self.mpnn.loss_mask(samples)
        grpo = compute_grpo_loss(current_logps, ref_logps, advantages, mask, cfg.beta)

        self.optimizer.zero_grad()
        grpo.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mpnn.model.parameters(), max_norm=cfg.grad_clip)
        self.optimizer.step()

        # --- Iterative backbone update ---
        if cfg.iterative:
            self._maybe_update_backbone(results, step)

        # --- Cleanup GPU ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Per-step summary with best/worst tracking
        scores = [r.score for r in results]
        return {
            "step": step + 1,
            "mean_reward": rewards.mean().item(),
            "max_reward": max(scores),
            "min_reward": min(scores),
            "loss": grpo.loss.item(),
            "kl": grpo.kl.item(),
            "policy_loss": grpo.policy_loss.item(),
            **{k: float(np.mean(v)) for k, v in metrics_acc.items()},
        }

    def _maybe_update_backbone(self, results, step: int) -> None:
        """Update the backbone PDB to the best predicted structure (iterative mode)."""
        best = max(results, key=lambda r: r.score)
        if best.structure_path and os.path.exists(best.structure_path):
            new_pdb = os.path.join(self.config.output_dir, f"backbone_step_{step}.pdb")
            try:
                from ..utils.structure import cif_to_pdb
                cif_to_pdb(best.structure_path, new_pdb)
                self.current_pdb = new_pdb
                logger.info(f"  Backbone updated -> {new_pdb}")
            except Exception as e:
                logger.warning(f"  Backbone update failed: {e}")

    def _save_results(self, results: list[dict]) -> None:
        path = os.path.join(self.config.output_dir, "training_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
