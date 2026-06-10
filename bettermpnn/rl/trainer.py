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
from ..environment.alphafold3 import SeedResult, DecoyResult
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

        # Load reference structures so conformational-change RMSD reward terms
        # (reward_weights.open_rmsd / closed_rmsd) are actually applied in train
        # mode. Without this the env never computes those metrics.
        self._load_reference_structures()

        os.makedirs(config.output_dir, exist_ok=True)
        self._init_csv_log()

    def _load_reference_structures(self) -> None:
        """Inject open/closed reference Cα coords into the environment, if set."""
        cfg = self.config
        if not (cfg.open_ref_pdb or cfg.closed_ref_pdb):
            return
        if not hasattr(self.env, "open_ref_ca"):
            return  # custom environment without RMSD support
        from ..utils.structure import extract_ca_from_pdb
        for attr, path, name in (
            ("open_ref_ca", cfg.open_ref_pdb, "open"),
            ("closed_ref_ca", cfg.closed_ref_pdb, "closed"),
        ):
            if path and os.path.exists(path):
                try:
                    ca = extract_ca_from_pdb(path, cfg.design_chain_id)
                    setattr(self.env, attr, ca)
                    logger.info(f"Loaded {name} reference ({len(ca)} Cα) from {path}")
                except Exception as e:
                    logger.warning(f"Failed to load {name} reference {path}: {e}")
            elif path:
                logger.warning(f"{name} reference PDB not found: {path}")

    def _init_csv_log(self):
        """Initialize the per-variant CSV log file."""
        self.csv_path = os.path.join(self.config.output_dir, "variant_log.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "variant", "score", "sequence",
                    "iptm", "ptm", "mean_pae", "has_clash", "ranking_score",
                    "open_rmsd", "closed_rmsd",
                    "decoy_tested", "passed_specificity",
                    "timestamp",
                ])

    def _log_variant(self, step: int, variant: int, score: float,
                     sequence: str, metrics: dict, decoy_info: dict = None):
        """Append one variant record to the CSV log."""
        di = decoy_info or {}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step + 1, variant, f"{score:.6f}", sequence,
                metrics.get("iptm", ""),
                metrics.get("ptm", ""),
                metrics.get("mean_pae", ""),
                metrics.get("has_clash", ""),
                metrics.get("ranking_score", ""),
                f"{metrics.get('open_rmsd'):.3f}" if "open_rmsd" in metrics else "",
                f"{metrics.get('closed_rmsd'):.3f}" if "closed_rmsd" in metrics else "",
                "Y" if di.get("tested") else "N",
                di.get("passed_str", ""),
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
        design_positions = cfg.design.get_redesign_positions(cfg.chain)
        samples = self.mpnn.sample(
            self.current_pdb, cfg.chain, cfg.variants, cfg.temperature,
            design_positions=design_positions if design_positions else None,
        )

        # --- Evaluate with environment ---
        results = self.env.evaluate_batch(samples.sequences, step=step)
        metrics_acc = defaultdict(list)

        # 1. Identify best variant index for this step
        best_idx = np.argmax([r.score for r in results])
        
        for i, (seq, res) in enumerate(zip(samples.sequences, results)):
            decoy_info = {}
            # Optional: Run decoy check for the BEST variant of each step to track specificity progress
            if i == best_idx and cfg.decoy_smiles and hasattr(self.env, 'evaluate_decoys'):
                logger.info(f"  variant {i} (BEST): running decoy check...")
                decoy_results = self.env.evaluate_decoys(
                    sequence=seq,
                    decoy_smiles_list=cfg.decoy_smiles,
                    step=step,
                    variant=i,
                    num_seeds=cfg.filter.decoy_num_seeds,
                    # Note: We don't have refs in Trainer by default, but env might have them
                    design_chain_id=cfg.design_chain_id,
                    filter_config=cfg.filter
                )
                if decoy_results:
                    all_passed = all(dr.passed_specificity for dr in decoy_results)
                    decoy_info = {
                        "tested": True,
                        "passed_specificity": all_passed,
                        "passed_str": "PASS" if all_passed else "FAIL",
                        "results": decoy_results
                    }
                    # Cleanup decoy predictions (save space)
                    if cfg.cleanup_predictions:
                        for d_idx in range(len(cfg.decoy_smiles)):
                            decoy_job = f"step_{step}_variant_{i}_decoy{d_idx}"
                            self.env.cleanup_prediction_dir(os.path.join(self.env.prediction_dir, decoy_job))
                            self.env.cleanup_prediction_dir(os.path.join(self.env.input_dir, decoy_job))

            logger.info(f"  variant {i}: score={res.score:.4f}  {res.metrics}")
            # Log every variant to CSV
            self._log_variant(step, i, res.score, seq, res.metrics, decoy_info)
            for k, v in res.metrics.items():
                if isinstance(v, (int, float)):
                    metrics_acc[k].append(v)
            
            # Save the best structure in a dedicated folder
            if i == best_idx:
                self._save_best_variant_set(step, i, res, decoy_info)

        rewards = torch.tensor([r.score for r in results], device=self.device)

        # Exclude variants whose AF3 evaluation failed. Their placeholder
        # score=0.0 would otherwise shift the group baseline and inject a
        # spurious gradient into the GRPO update.
        valid = [i for i, r in enumerate(results) if "error" not in r.metrics]
        n_failed = len(results) - len(valid)
        if n_failed:
            logger.warning(
                f"  {n_failed}/{len(results)} variant(s) failed AF3; excluded from the GRPO update"
            )

        current_logps = self.mpnn.compute_log_probs(samples)
        with torch.no_grad():
            ref_logps = self.ref_mpnn.compute_log_probs(samples)
        mask = self.mpnn.loss_mask(samples)

        # --- GRPO update over the successful variants only ---
        loss_val = kl_val = policy_loss_val = 0.0
        if len(valid) >= 2:
            idx = torch.tensor(valid, device=self.device)
            advantages = compute_advantages(
                rewards[idx],
                shaping_alpha=cfg.reward_shaping_alpha,
                scale_factor=cfg.advantage_scale_factor,
            )
            grpo = compute_grpo_loss(
                current_logps[idx], ref_logps[idx], advantages, mask[idx], cfg.beta
            )
            self.optimizer.zero_grad()
            grpo.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mpnn.model.parameters(), max_norm=cfg.grad_clip)
            self.optimizer.step()
            loss_val, kl_val, policy_loss_val = (
                grpo.loss.item(), grpo.kl.item(), grpo.policy_loss.item(),
            )
        else:
            logger.warning(
                "  Fewer than 2 successful variants this step; skipping the GRPO update"
            )

        # --- Iterative backbone update ---
        if cfg.iterative:
            self._maybe_update_backbone(results, step)

        # --- Cleanup GPU ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Per-step summary with best/worst tracking (over successful variants)
        scores = [results[i].score for i in valid] or [r.score for r in results]
        return {
            "step": step + 1,
            "mean_reward": float(np.mean(scores)),
            "max_reward": max(scores),
            "min_reward": min(scores),
            "n_failed": n_failed,
            "loss": loss_val,
            "kl": kl_val,
            "policy_loss": policy_loss_val,
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

    def _save_best_variant_set(self, step: int, variant_idx: int, res, decoy_info: dict):
        """Save the best variant's structures (target, decoys, open-ref) to a folder."""
        best_dir = os.path.join(self.config.output_dir, f"step{step+1}_best_var{variant_idx}")
        os.makedirs(best_dir, exist_ok=True)
        
        import shutil
        # 1. Target Bound
        if res.structure_path and os.path.exists(res.structure_path):
            shutil.copy2(res.structure_path, os.path.join(best_dir, "bound_target.cif"))
        
        # 2. Open Reference
        if self.config.open_ref_pdb and os.path.exists(self.config.open_ref_pdb):
            ref_ext = os.path.splitext(self.config.open_ref_pdb)[1]
            shutil.copy2(self.config.open_ref_pdb, os.path.join(best_dir, f"open_reference{ref_ext}"))
            
        # 3. Decoys
        for d_idx, dr in enumerate(decoy_info.get("results", [])):
            if dr.structure_path and os.path.exists(dr.structure_path):
                d_name = "".join(c if c.isalnum() else "_" for c in dr.decoy_smiles[:20])
                shutil.copy2(dr.structure_path, os.path.join(best_dir, f"bound_decoy_{d_idx}_{d_name}.cif"))
        
        logger.info(f"  Best structure set saved to: {best_dir}")

    def _save_results(self, results: list[dict]) -> None:
        path = os.path.join(self.config.output_dir, "training_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
