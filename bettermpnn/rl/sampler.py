"""Sampling loop: MPNN sampling + AF3 evaluation + filtering (no RL update)."""

import csv
import json
import logging
import os
import shutil
import time
from typing import Optional

import torch

from ..config import Config
from ..environment.alphafold3 import AlphaFold3Environment, SeedResult, DecoyResult
from ..mpnn.wrapper import MPNNModel
from ..utils.structure import extract_ca_from_pdb

logger = logging.getLogger(__name__)


class Sampler:
    """Large-scale sampling pipeline: frozen MPNN → AF3 → filter.

    Unlike Trainer, this class:
    - Freezes MPNN weights (no optimizer, no backward pass)
    - Evaluates ALL AF3 seeds per variant (not just best-of)
    - Applies iptm + RMSD filtering criteria
    - Outputs comprehensive screening logs
    """

    def __init__(self, mpnn: MPNNModel, environment: AlphaFold3Environment, config: Config):
        self.mpnn = mpnn
        self.env = environment
        self.config = config
        self.device = mpnn.device

        # Freeze MPNN
        self.mpnn.model.eval()
        for p in self.mpnn.model.parameters():
            p.requires_grad = False

        # Load reference Cα coordinates for RMSD
        self.open_ref_ca = self._load_reference_ca(config.open_ref_pdb, config.design_chain_id, "open")
        self.closed_ref_ca = self._load_reference_ca(config.closed_ref_pdb, config.design_chain_id, "closed")

        # Fail fast if a reference was given but extracted no Cα (wrong chain id).
        # No fallback: without an explicit open_ref_pdb the open-RMSD filter stays
        # off (it would otherwise reject plain binders by comparing to the scaffold).
        for path, ca in ((config.closed_ref_pdb, self.closed_ref_ca),
                         (config.open_ref_pdb, self.open_ref_ca)):
            if path and (ca is None or len(ca) == 0):
                raise ValueError(f"{path}: no Cα for chain '{config.design_chain_id}'.")

        if config.decoy_smiles and (self.closed_ref_ca is None or len(self.closed_ref_ca) == 0):
            logger.info("No closed reference: decoy specificity judged by iPTM only.")

        # Output setup
        os.makedirs(config.output_dir, exist_ok=True)
        self.passed_dir = os.path.join(config.output_dir, "passed")
        self.passed_struct_dir = os.path.join(config.output_dir, "passed_structural_only")
        os.makedirs(self.passed_dir, exist_ok=True)
        os.makedirs(self.passed_struct_dir, exist_ok=True)

        # CSV log will be initialized in run() to support per-worker naming
        self.csv_path = None

        # Counters
        self.total_seeds = 0
        self.total_passed = 0
        self.total_passed_with_specificity = 0
        self.total_decoy_tested = 0
        self.passed_variants = set()
        self.specific_variants = set()

    @staticmethod
    def _load_reference_ca(pdb_path: str, chain_id: str, name: str):
        """Load reference Cα coordinates from a PDB file."""
        if not pdb_path:
            return None
        if not os.path.exists(pdb_path):
            logger.warning(f"Reference PDB not found for {name}: {pdb_path}")
            return None
        try:
            ca = extract_ca_from_pdb(pdb_path, chain_id)
            logger.info(f"Loaded {name} reference: {len(ca)} Cα atoms from {pdb_path}")
            return ca
        except Exception as e:
            logger.error(f"Failed to load {name} reference {pdb_path}: {e}")
            return None

    def _init_csv_log(self, suffix: str = ""):
        """Initialize the per-seed screening CSV log file."""
        name = f"screening_log{suffix}.csv"
        self.csv_path = os.path.join(self.config.output_dir, name)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "variant", "seed", "sequence",
                    "iptm", "ptm", "mean_pae", "has_clash", "ranking_score",
                    "open_rmsd", "closed_rmsd", "local_drmsd",
                    "passed_structure",
                    "decoy_tested", "decoy_max_iptm", "decoy_min_closed_rmsd",
                    "passed_specificity", "final_pass",
                    "structure_path", "timestamp",
                ])

    def _log_seed_result(
        self, step: int, variant: int, sequence: str, sr: SeedResult,
        decoy_info: dict = None,
    ):
        """Append one seed record to the screening CSV log."""
        di = decoy_info or {}
        final_pass = sr.passed and di.get("passed_specificity", True) if sr.passed else False

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step + 1, variant, sr.seed, sequence,
                sr.metrics.get("iptm", ""),
                sr.metrics.get("ptm", ""),
                sr.metrics.get("mean_pae", ""),
                sr.metrics.get("has_clash", ""),
                sr.metrics.get("ranking_score", ""),
                f"{sr.open_rmsd:.4f}" if sr.open_rmsd is not None else "",
                f"{sr.closed_rmsd:.4f}" if sr.closed_rmsd is not None else "",
                f"{sr.local_drmsd:.4f}" if sr.local_drmsd is not None else "",
                "PASS" if sr.passed else "FAIL",
                "Y" if di.get("tested", False) else "N",
                f"{di['max_iptm']:.3f}" if "max_iptm" in di else "",
                f"{di['min_closed_rmsd']:.2f}" if "min_closed_rmsd" in di else "",
                di.get("passed_str", ""),
                "PASS" if final_pass else "FAIL",
                sr.structure_path or "",
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ])

    def run(self, step_range: Optional[tuple[int, int]] = None) -> dict:
        """Run the full sampling + filtering pipeline.

        Args:
            step_range: Optional (start, end) inclusive step indices for
                        parallel job array support. If None, runs all steps.
        """
        cfg = self.config
        start_time = time.time()

        # Determine which steps this worker handles
        if step_range is not None:
            step_start, step_end = step_range
            step_end = min(step_end, cfg.steps - 1)
            steps_to_run = list(range(step_start, step_end + 1))
            worker_suffix = f"_step{step_start}-{step_end}"
        else:
            steps_to_run = list(range(cfg.steps))
            worker_suffix = ""

        if not steps_to_run:
            logger.warning(f"No steps to run for range {step_range} (steps={cfg.steps}); nothing to do.")
            return self._save_summary(worker_suffix)

        # Init CSV with worker-specific name to avoid contention
        self._init_csv_log(worker_suffix)

        logger.info(f"{'='*60}")
        logger.info("BetterMPNN Large-Scale Sampling Pipeline")
        logger.info(f"{'='*60}")
        logger.info(f"Steps: {steps_to_run[0]+1}-{steps_to_run[-1]+1} "
                     f"({len(steps_to_run)} of {cfg.steps} total)")
        logger.info(f"Variants/step: {cfg.variants}, Seeds/variant: {cfg.num_seeds}")
        logger.info(f"Backbone: {cfg.pdb}, Chain: {cfg.chain}")
        logger.info(f"Filter: iPTM≥{cfg.filter.iptm_min}, open_rmsd≥{cfg.filter.open_rmsd_min}Å, "
                     f"local_drmsd<{cfg.filter.local_drmsd_max}Å")
        if cfg.decoy_smiles:
            logger.info(f"Decoy ligands: {len(cfg.decoy_smiles)} interferents")
            logger.info(f"  Decoy iPTM max: {cfg.filter.decoy_iptm_max}, "
                         f"Decoy closed_rmsd min: {cfg.filter.decoy_closed_rmsd_min}Å")
        logger.info(f"Cleanup predictions: {cfg.cleanup_predictions}")
        logger.info(f"Open ref: {cfg.open_ref_pdb}")
        logger.info(f"Closed ref: {cfg.closed_ref_pdb}")
        logger.info(f"Output: {cfg.output_dir}")
        logger.info(f"CSV log: {self.csv_path}")
        logger.info(f"{'='*60}")

        for idx, step in enumerate(steps_to_run):
            step_start_t = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {step + 1}/{cfg.steps} (worker progress: {idx+1}/{len(steps_to_run)})")
            logger.info(f"{'='*60}")

            self._sample_step(step)

            elapsed = time.time() - step_start_t
            logger.info(
                f"  Step {step+1} done in {elapsed:.1f}s | "
                f"Total passed: {self.total_passed}/{self.total_seeds} seeds "
                f"({len(self.passed_variants)} unique variants)"
            )

            # Save summary incrementally
            self._save_summary(worker_suffix)

            # Cleanup GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Sampling complete. Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info(f"  Steps processed: {steps_to_run[0]+1}-{steps_to_run[-1]+1}")
        logger.info(f"  Total seeds evaluated: {self.total_seeds}")
        logger.info(f"  Total passed: {self.total_passed}")
        logger.info(f"  Unique variants with ≥1 pass: {len(self.passed_variants)}")
        logger.info(f"  Pass rate: {100*self.total_passed/max(1,self.total_seeds):.2f}%")
        logger.info(f"{'='*60}")

        summary = self._save_summary(worker_suffix)
        return summary

    def _sample_step(self, step: int):
        """Execute one sampling step: sample sequences → evaluate all seeds → decoy check → filter."""
        cfg = self.config

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + step)  # distinct per step; avoids worker collisions

        # --- Sample sequences (frozen, no grad) ---
        design_positions = cfg.design.get_redesign_positions(cfg.chain)
        with torch.no_grad():
            samples = self.mpnn.sample(
                cfg.pdb, cfg.chain, cfg.variants, cfg.temperature,
                design_positions=design_positions if design_positions else None,
            )

        # --- Evaluate each variant across all seeds ---
        for i, seq in enumerate(samples.sequences):
            logger.info(f"  Variant {i}: evaluating {cfg.num_seeds} seeds...")

            seed_results = self.env.evaluate_all_seeds(
                sequence=seq,
                step=step,
                variant=i,
                num_seeds=cfg.num_seeds,
                open_ref_ca=self.open_ref_ca,
                closed_ref_ca=self.closed_ref_ca,
                design_chain_id=cfg.design_chain_id,
                filter_config=cfg.filter,
            )

            # Check if any seed passed structural filter
            variant_has_pass = any(sr.passed for sr in seed_results)

            # --- Decoy specificity check & Apo state prediction (only for variants with at least one PASS) ---
            decoy_info = {}
            predicted_apo_path = None
            if variant_has_pass:
                logger.info(f"  Variant {i}: structural PASS, predicting apo state...")
                if hasattr(self.env, 'evaluate_apo'):
                    predicted_apo_path = self.env.evaluate_apo(seq, step, i, num_seeds=1)
                
                if cfg.decoy_smiles:
                    logger.info(f"  Variant {i}: structural PASS, running decoy specificity test...")
                    decoy_info = self._evaluate_decoy_specificity(
                        seq, step, i, cfg.decoy_smiles
                    )
                    self.total_decoy_tested += 1

                if decoy_info.get("passed_specificity", True):
                    self.total_passed_with_specificity += 1
                    self.specific_variants.add((step, i))
                    logger.info(f"  Variant {i}: ✓ PASSED all filters (structure + specificity)")
                else:
                    logger.info(f"  Variant {i}: ✗ FAILED specificity (responds to interferent)")
                    # NOTE: We do NOT set variant_has_pass = False here because we still
                    # want to log the structural-only success in the CSV and folders.

            # --- Log all seed results and extract structures ---
            if variant_has_pass:
                self.passed_variants.add((step, i))

            for sr in seed_results:
                self.total_seeds += 1
                
                # Link decoy results to the seed for extraction logic
                if sr.passed:
                    sr.predicted_apo_path = predicted_apo_path
                    if "details" in decoy_info:
                        sr.decoy_results = [
                            DecoyResult(
                                decoy_smiles=d["smiles"],
                                best_iptm=d["iptm"],
                                min_closed_rmsd=d["min_closed_rmsd"],
                                causes_closing=d["causes_closing"],
                                passed_specificity=d["passed"],
                                structure_path=d.get("structure_path")
                            ) for d in decoy_info["details"]
                        ]

                self._log_seed_result(step, i, seq, sr, decoy_info if sr.passed else {})

                status = "✓ PASS" if sr.passed else "✗ FAIL"
                rmsd_str = f"{sr.open_rmsd:.2f}" if sr.open_rmsd is not None else "N/A"
                drmsd_str = f"{sr.local_drmsd:.2f}" if sr.local_drmsd is not None else "N/A"
                logger.info(
                    f"    seed={sr.seed:2d}  iptm={sr.metrics.get('iptm',0):.3f}  "
                    f"rmsd={rmsd_str}  drmsd={drmsd_str}  {status}"
                )

                if sr.passed:
                    # Create a dedicated directory for this successful candidate
                    folder_name = f"step{step+1}_var{i}_seed{sr.seed}"
                    
                    # 1. Structural only pass directory (even if specificity failed)
                    struct_only_dir = os.path.join(self.passed_struct_dir, folder_name)
                    os.makedirs(struct_only_dir, exist_ok=True)
                    self._extract_complete_set(sr, struct_only_dir)

                    # 2. Final pass (requires specificity)
                    if decoy_info.get("passed_specificity", True):
                        self.total_passed += 1
                        final_dir = os.path.join(self.passed_dir, folder_name)
                        os.makedirs(final_dir, exist_ok=True)
                        self._extract_complete_set(sr, final_dir)

            if not seed_results:
                logger.warning(f"  Variant {i}: no seed results (AF3 may have failed)")

            # --- Cleanup AF3 prediction files ---
            if cfg.cleanup_predictions:
                # Cleanup target predictions
                target_pred_dir = os.path.join(self.env.prediction_dir, f"step_{step}_variant_{i}")
                self.env.cleanup_prediction_dir(target_pred_dir)
                target_input_dir = os.path.join(self.env.input_dir, f"step_{step}_variant_{i}")
                self.env.cleanup_prediction_dir(target_input_dir)
                
                # Cleanup apo prediction
                apo_job = f"step_{step}_variant_{i}_apo"
                self.env.cleanup_prediction_dir(os.path.join(self.env.prediction_dir, apo_job))
                self.env.cleanup_prediction_dir(os.path.join(self.env.input_dir, apo_job))
                
                # Cleanup decoy predictions (if any)
                if cfg.decoy_smiles:
                    for d_idx in range(len(cfg.decoy_smiles)):
                        decoy_job = f"step_{step}_variant_{i}_decoy{d_idx}"
                        self.env.cleanup_prediction_dir(os.path.join(self.env.prediction_dir, decoy_job))
                        self.env.cleanup_prediction_dir(os.path.join(self.env.input_dir, decoy_job))

    def _extract_complete_set(self, sr, target_dir):
        """Copy target, decoy, and open-ref structures to the given directory."""
        # Target bound structure
        if sr.structure_path and os.path.exists(sr.structure_path):
            shutil.copy2(sr.structure_path, os.path.join(target_dir, "bound_target.cif"))
        
        # Apo/Open reference
        if self.config.open_ref_pdb and os.path.exists(self.config.open_ref_pdb):
            ref_ext = os.path.splitext(self.config.open_ref_pdb)[1]
            shutil.copy2(self.config.open_ref_pdb, os.path.join(target_dir, f"open_reference{ref_ext}"))
            
        # Predicted Apo structure
        if hasattr(sr, 'predicted_apo_path') and sr.predicted_apo_path and os.path.exists(sr.predicted_apo_path):
            shutil.copy2(sr.predicted_apo_path, os.path.join(target_dir, "predicted_apo.cif"))
            
        # Decoy bound structures
        for d_idx, dr in enumerate(sr.decoy_results):
            if dr.structure_path and os.path.exists(dr.structure_path):
                # Sanitize decoy name for filename
                d_name = "".join(c if c.isalnum() else "_" for c in dr.decoy_smiles[:20])
                shutil.copy2(dr.structure_path, os.path.join(target_dir, f"bound_decoy_{d_idx}_{d_name}.cif"))

    def _evaluate_decoy_specificity(
        self, sequence: str, step: int, variant: int,
        decoy_smiles: list[str],
    ) -> dict:
        """Run decoy AF3 evaluation and return aggregated specificity info."""
        cfg = self.config
        fc = cfg.filter

        decoy_results = self.env.evaluate_decoys(
            sequence=sequence,
            decoy_smiles_list=decoy_smiles,
            step=step,
            variant=variant,
            num_seeds=fc.decoy_num_seeds,
            open_ref_ca=self.open_ref_ca,
            closed_ref_ca=self.closed_ref_ca,
            design_chain_id=cfg.design_chain_id,
            filter_config=fc,
        )

        if not decoy_results:
            return {"tested": False}

        all_passed = all(dr.passed_specificity for dr in decoy_results)
        max_iptm = max(dr.best_iptm for dr in decoy_results)
        min_closed_rmsd = min(dr.min_closed_rmsd for dr in decoy_results)

        return {
            "tested": True,
            "passed_specificity": all_passed,
            "max_iptm": max_iptm,
            "min_closed_rmsd": min_closed_rmsd,
            "passed_str": "PASS" if all_passed else "FAIL",
            "details": [
                {
                    "smiles": dr.decoy_smiles,
                    "iptm": dr.best_iptm,
                    "min_closed_rmsd": dr.min_closed_rmsd,
                    "causes_closing": dr.causes_closing,
                    "passed": dr.passed_specificity,
                    "structure_path": dr.structure_path,
                }
                for dr in decoy_results
            ],
        }

    def _save_summary(self, suffix: str = "") -> dict:
        """Save screening summary JSON."""
        cfg = self.config
        summary = {
            "total_steps": cfg.steps,
            "variants_per_step": cfg.variants,
            "seeds_per_variant": cfg.num_seeds,
            "total_variants": cfg.steps * cfg.variants,
            "total_seeds_evaluated": self.total_seeds,
            "passed_seeds": self.total_passed,
            "passed_unique_variants": len(self.passed_variants),
            "decoy_tested": self.total_decoy_tested,
            "passed_with_specificity": self.total_passed_with_specificity,
            "specific_unique_variants": len(self.specific_variants),
            "pass_rate": f"{100*self.total_passed/max(1,self.total_seeds):.2f}%",
            "filter_config": {
                "iptm_min": cfg.filter.iptm_min,
                "open_rmsd_min": cfg.filter.open_rmsd_min,
                "local_drmsd_max": cfg.filter.local_drmsd_max,
                "ptm_min": cfg.filter.ptm_min,
                "pae_max": cfg.filter.pae_max,
                "decoy_iptm_max": cfg.filter.decoy_iptm_max,
                "decoy_closed_rmsd_min": cfg.filter.decoy_closed_rmsd_min,
            },
            "decoy_smiles": cfg.decoy_smiles,
            "references": {
                "scaffold_pdb": cfg.pdb,
                "open_ref_pdb": cfg.open_ref_pdb,
                "closed_ref_pdb": cfg.closed_ref_pdb,
                "design_chain": cfg.design_chain_id,
            },
        }
        path = os.path.join(cfg.output_dir, f"screening_summary{suffix}.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary
