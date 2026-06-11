"""AlphaFold 3 environment: evaluates sequences via AF3 structure prediction."""

import json
import logging
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import numpy as np

from .base import Environment, EvalResult
from ..config import EnvironmentConfig, FilterConfig
from ..utils.structure import extract_ca_from_cif
from ..utils.rmsd import calculate_rmsd, compute_local_drmsd

logger = logging.getLogger(__name__)

PAE_MAX = 31.75  # Maximum PAE value for normalization


@dataclass
class DecoyResult:
    """Evaluation result for a decoy/interferent ligand."""
    decoy_smiles: str
    best_iptm: float = 0.0              # Best (worst-case) iPTM across seeds
    min_closed_rmsd: float = float("inf") # Worst-case (lowest) closed_rmsd across seeds
    causes_closing: bool = False        # True if decoy triggers conformational change
    passed_specificity: bool = True     # True if the variant is specific against this decoy
    structure_path: Optional[str] = None # Path to best (worst-case) decoy structure


@dataclass
class SeedResult:
    """Evaluation result for a single AF3 seed."""
    seed: int
    metrics: dict                       # iptm, ptm, pae, has_clash, ranking_score
    open_rmsd: Optional[float] = None   # RMSD vs open-ref PDB (if provided)
    closed_rmsd: Optional[float] = None # RMSD vs closed-ref PDB (if provided)
    local_drmsd: Optional[float] = None # Fold integrity check
    structure_path: Optional[str] = None
    passed: bool = False
    decoy_results: List[DecoyResult] = field(default_factory=list)
    predicted_apo_path: Optional[str] = None


class AlphaFold3Environment(Environment):
    """Evaluate protein sequences using AlphaFold 3 structure prediction.

    Runs AF3 in a Singularity/Apptainer container, parses confidence metrics
    from the output, and returns a weighted reward score.
    """

    def __init__(self, config: EnvironmentConfig, output_dir: str = "output",
                 target_smiles: str = "", scaffold_name: str = "",
                 ligand_name: str = "", design_chain_id: str = "A"):
        self.config = config
        self.output_dir = output_dir
        self.scaffold_name = scaffold_name
        self.ligand_name = ligand_name
        # Chain ID used for Cα extraction when computing RMSD rewards.
        self.design_chain_id = design_chain_id
        self.input_dir = os.path.join(output_dir, "af3_inputs")
        self.prediction_dir = os.path.join(output_dir, "af3_predictions")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.prediction_dir, exist_ok=True)

        # Optional reference Cα coords for conformational-change reward during
        # training mode. Populated by the Trainer from config.open_ref_pdb /
        # config.closed_ref_pdb; the sampler passes references explicitly as
        # method arguments instead.
        self.open_ref_ca: Optional["np.ndarray"] = None
        self.closed_ref_ca: Optional["np.ndarray"] = None
        self._chain_len_warned = False

        # Load template JSON
        with open(config.template_json) as f:
            self.template = json.load(f)

        # Override ligand SMILES in template if provided via config
        if target_smiles:
            for seq_entry in self.template.get("sequences", []):
                if "ligand" in seq_entry:
                    seq_entry["ligand"]["smiles"] = target_smiles
                    logger.info(f"Template ligand SMILES overridden: {target_smiles}")
                    break

    def evaluate(self, sequence: str, step: int = 0, variant: int = 0) -> EvalResult:
        """Evaluate a single sequence (original single-best interface for training mode)."""
        job_name = f"step_{step}_variant_{variant}"

        # 1. Create input JSON
        json_path = self._create_input_json(sequence, job_name)

        # 2. Run AF3
        output_dir = os.path.join(self.prediction_dir, job_name)
        os.makedirs(output_dir, exist_ok=True)
        success = self._run_af3(json_path, output_dir, job_name)

        if not success:
            logger.warning(f"AF3 prediction failed for {job_name}")
            return EvalResult(score=0.0, metrics={"error": "prediction_failed"})

        # 3. Parse results
        metrics = self._parse_results(output_dir)
        if metrics is None:
            return EvalResult(score=0.0, metrics={"error": "parse_failed"})

        # 4. Find structure path (for iterative mode or RMSD calculation)
        structure_path = self._find_structure_path(output_dir)

        # 5. Calculate RMSD for the conformational-change reward, when the
        # Trainer has injected reference Cα coordinates (open/closed states).
        if structure_path and os.path.exists(structure_path) and (
            self.open_ref_ca is not None or self.closed_ref_ca is not None
        ):
            try:
                pred_ca = extract_ca_from_cif(structure_path, self.design_chain_id)
                if self.open_ref_ca is not None:
                    metrics["open_rmsd"] = calculate_rmsd(pred_ca, self.open_ref_ca, align=True)
                if self.closed_ref_ca is not None:
                    metrics["closed_rmsd"] = calculate_rmsd(pred_ca, self.closed_ref_ca, align=True)
            except Exception as e:
                logger.debug(f"Training RMSD calc failed: {e}")

        # 6. Calculate reward
        score = self._calculate_reward(metrics)

        return EvalResult(score=score, structure_path=structure_path, metrics=metrics)

    def evaluate_all_seeds(
        self,
        sequence: str,
        step: int,
        variant: int,
        num_seeds: int,
        open_ref_ca: Optional["np.ndarray"] = None,
        closed_ref_ca: Optional["np.ndarray"] = None,
        design_chain_id: str = "A",
        filter_config: Optional[FilterConfig] = None,
    ) -> list[SeedResult]:
        """Evaluate a sequence across ALL AF3 seeds and return per-seed results.

        This is the main method for sampling mode. It:
        1. Creates input JSON with `num_seeds` model seeds
        2. Runs AF3
        3. Iterates over ALL output seed directories
        4. Parses metrics + extracts Cα + computes RMSD for each seed
        5. Applies filter criteria to determine pass/fail

        Args:
            sequence: Amino acid sequence string.
            step: Current sampling step.
            variant: Variant index within the step.
            num_seeds: Number of AF3 model seeds.
            open_ref_ca: Cα coordinates of open/apo reference (N, 3).
            design_chain_id: Chain ID for Cα extraction.
            filter_config: Filtering thresholds.

        Returns:
            List of SeedResult, one per seed found in output.
        """
        job_name = f"step_{step}_variant_{variant}"

        # 1. Create input JSON with multiple seeds
        json_path = self._create_input_json(sequence, job_name, num_seeds=num_seeds)

        # 2. Run AF3
        output_dir = os.path.join(self.prediction_dir, job_name)
        os.makedirs(output_dir, exist_ok=True)
        success = self._run_af3(json_path, output_dir, job_name)

        if not success:
            logger.warning(f"AF3 prediction failed for {job_name}")
            return []

        # 3. Find ALL seed results in output directory
        seed_results = []
        seed_dirs = self._find_all_seed_outputs(output_dir)

        if not seed_dirs:
            # Fallback: try to find results directly in output_dir (single-seed layout)
            summary = self._find_summary_file(output_dir)
            if summary:
                seed_dirs = [(0, output_dir, summary)]

        for seed_idx, seed_dir, summary_path in seed_dirs:
            try:
                # Parse metrics
                metrics = self._parse_summary_file(summary_path)
                if metrics is None:
                    continue

                # Find structure CIF
                cif_path = self._find_structure_in_dir(seed_dir)

                # Compute RMSD if reference available
                open_rmsd = None
                closed_rmsd = None
                local_drmsd_val = None

                want_open = open_ref_ca is not None and len(open_ref_ca) > 0
                want_closed = closed_ref_ca is not None and len(closed_ref_ca) > 0
                rmsd_error = None

                if cif_path:
                    try:
                        pred_ca = extract_ca_from_cif(cif_path, design_chain_id)
                        if len(pred_ca) == 0:
                            rmsd_error = f"no Cα for chain {design_chain_id}"
                        else:
                            if want_open:
                                open_rmsd = calculate_rmsd(pred_ca, open_ref_ca, align=True)
                                local_drmsd_val = compute_local_drmsd(open_ref_ca, pred_ca, seq_sep=6)
                            if want_closed:
                                closed_rmsd = calculate_rmsd(pred_ca, closed_ref_ca, align=True)
                    except Exception as e:
                        rmsd_error = str(e)
                elif want_open or want_closed:
                    rmsd_error = "no predicted structure CIF found"

                # Fail closed: if an RMSD reference was configured but we could
                # not compute the required RMSD for this seed, the seed must not
                # pass on iPTM alone (that would silently weaken the filter).
                if (want_open or want_closed) and rmsd_error is not None:
                    logger.warning(
                        f"Seed {seed_idx} for {job_name}: RMSD required but not "
                        f"computable ({rmsd_error}); failing this seed (fail-closed)."
                    )
                    passed = False
                else:
                    passed = self._check_filter(
                        metrics, open_rmsd, closed_rmsd, local_drmsd_val, filter_config
                    )

                sr = SeedResult(
                    seed=seed_idx,
                    metrics=metrics,
                    open_rmsd=open_rmsd,
                    closed_rmsd=closed_rmsd,
                    local_drmsd=local_drmsd_val,
                    structure_path=cif_path,
                    passed=passed,
                    decoy_results=[], # Will be populated by Sampler after decoy testing
                )
                seed_results.append(sr)

            except Exception as e:
                logger.warning(f"Failed to process seed {seed_idx} for {job_name}: {e}")

        return seed_results

    @staticmethod
    def _check_filter(
        metrics: dict,
        open_rmsd: Optional[float],
        closed_rmsd: Optional[float],
        local_drmsd: Optional[float],
        filter_config: Optional[FilterConfig],
    ) -> bool:
        """Check if a seed result passes the filter criteria."""
        if filter_config is None:
            return True

        iptm = metrics.get("iptm", 0.0)
        ptm = metrics.get("ptm", 0.0)
        mean_pae = metrics.get("mean_pae", PAE_MAX)

        # Hard thresholds
        if iptm < filter_config.iptm_min:
            logger.debug(f"Filter reject: iptm {iptm:.3f} < {filter_config.iptm_min}")
            return False
        if ptm < filter_config.ptm_min:
            logger.debug(f"Filter reject: ptm {ptm:.3f} < {filter_config.ptm_min}")
            return False
        if mean_pae > filter_config.pae_max:
            logger.debug(f"Filter reject: mean_pae {mean_pae:.2f} > {filter_config.pae_max}")
            return False
        
        # Conformational change check
        if open_rmsd is not None and open_rmsd < filter_config.open_rmsd_min:
            logger.debug(f"Filter reject: open_rmsd {open_rmsd:.2f} < {filter_config.open_rmsd_min}")
            return False
        if closed_rmsd is not None and closed_rmsd > filter_config.closed_rmsd_max:
            logger.debug(f"Filter reject: closed_rmsd {closed_rmsd:.2f} > {filter_config.closed_rmsd_max}")
            return False
        
        # Fold integrity check
        if local_drmsd is not None and local_drmsd > filter_config.local_drmsd_max:
            logger.debug(f"Filter reject: local_drmsd {local_drmsd:.2f} > {filter_config.local_drmsd_max}")
            return False

        return True

    def _create_input_json(self, sequence: str, job_name: str,
                           num_seeds: Optional[int] = None) -> str:
        """Create AF3 input JSON with the designed sequence substituted in."""
        import copy
        data = copy.deepcopy(self.template)

        # Set job name in JSON
        s_name = self.scaffold_name or "protein"
        l_name = self.ligand_name or "ligand"
        data["name"] = f"{s_name}-{l_name}-{job_name}"

        # Set model seeds
        if num_seeds is not None and num_seeds > 0:
            data["modelSeeds"] = list(range(1, num_seeds + 1))

        self._apply_designed_sequence(data, sequence)

        job_dir = os.path.join(self.input_dir, job_name)
        os.makedirs(job_dir, exist_ok=True)
        json_path = os.path.join(job_dir, f"{job_name}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        return json_path

    def _apply_designed_sequence(self, data: dict, sequence: str) -> None:
        """Substitute the designed sequence (and its single-sequence MSA) into
        the design chain of an AF3 input ``data`` dict, in place.

        Shared by the target/decoy/apo input builders so none of them can
        silently drop the sequence. Raises on a misconfigured template.
        """
        idx = self.config.design_chain_index
        sequences = data.get("sequences", [])
        if idx >= len(sequences):
            raise ValueError(
                f"design_chain_index={idx} is out of range for template with "
                f"{len(sequences)} sequence entries ({self.config.template_json}). "
                f"Set environment.design_chain_index to the designed chain's position."
            )
        chain = sequences[idx].get("protein")
        if chain is None:
            raise ValueError(
                f"Template sequence entry at design_chain_index={idx} has no "
                f"'protein' block; it cannot be the designed chain. Check that "
                f"design_chain_index points to the redesigned protein chain."
            )
        old_seq = chain.get("sequence", "")
        if old_seq and len(old_seq) != len(sequence):
            # Fixed-backbone redesign keeps the chain length identical; a
            # mismatch almost always means design_chain_index points at the
            # wrong chain (e.g. the target instead of the binder).
            self._warn_once_chain_len(
                f"Designed sequence length ({len(sequence)}) != template chain "
                f"length at design_chain_index={idx} ({len(old_seq)}). "
                f"Verify design_chain_index matches the chain you redesign."
            )
        chain["sequence"] = sequence
        # Update the MSA. The designed chain must use a single-sequence MSA (a
        # ">name\nSEQUENCE" block) because its sequence changes every step: we
        # replace that query line. A multi-record MSA cannot be updated this way
        # without silently corrupting alignment columns, so reject it outright.
        for msa_key in ("unpairedMsa", "pairedMsa"):
            msa = chain.get(msa_key, "")
            if msa:
                lines = msa.splitlines()
                if sum(1 for ln in lines if ln.startswith(">")) > 1:
                    raise ValueError(
                        f"Designed chain (design_chain_index={idx}) has a "
                        f"multi-record {msa_key}; only a single-sequence MSA is "
                        f"supported for the redesigned chain. Provide a two-line "
                        f"'>name\\nSEQUENCE' block instead."
                    )
                if len(lines) >= 2:
                    lines[1] = sequence
                chain[msa_key] = "\n".join(lines)

    def _warn_once_chain_len(self, msg: str) -> None:
        """Emit the chain-length mismatch warning only once per run."""
        if not getattr(self, "_chain_len_warned", False):
            logger.warning(msg)
            self._chain_len_warned = True

    def _create_decoy_input_json(
        self, sequence: str, decoy_smiles: str, job_name: str,
        num_seeds: int = 5,
    ) -> str:
        """Create AF3 input JSON with a decoy ligand instead of the target.

        Replaces the ligand SMILES in the template with the decoy SMILES,
        keeping everything else (protein sequence, MSA) the same.
        """
        import copy
        data = copy.deepcopy(self.template)

        # Set job name in JSON
        s_name = self.scaffold_name or "protein"
        l_name = "decoy"
        data["name"] = f"{s_name}-{l_name}-{job_name}"

        # Set model seeds (fewer for decoy screening)
        if num_seeds > 0:
            data["modelSeeds"] = list(range(1, num_seeds + 1))

        # Update protein sequence (shared strict path)
        self._apply_designed_sequence(data, sequence)

        # Replace ligand SMILES with decoy
        for seq_entry in data.get("sequences", []):
            if "ligand" in seq_entry:
                seq_entry["ligand"]["smiles"] = decoy_smiles
                break

        # Update name
        data["name"] = f"{data.get('name', 'decoy')}_decoy"

        job_dir = os.path.join(self.input_dir, job_name)
        os.makedirs(job_dir, exist_ok=True)
        json_path = os.path.join(job_dir, f"{job_name}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        return json_path

    def _create_apo_input_json(
        self, sequence: str, job_name: str, num_seeds: int = 1
    ) -> str:
        """Create AF3 input JSON without any ligand (Apo state)."""
        import copy
        data = copy.deepcopy(self.template)

        s_name = self.scaffold_name or "protein"
        data["name"] = f"{s_name}-apo-{job_name}"

        if num_seeds > 0:
            data["modelSeeds"] = list(range(1, num_seeds + 1))

        # Update protein sequence (shared strict path)
        self._apply_designed_sequence(data, sequence)

        # Remove all ligand sequences (apo state)
        data["sequences"] = [seq for seq in data.get("sequences", []) if "ligand" not in seq]

        job_dir = os.path.join(self.input_dir, job_name)
        os.makedirs(job_dir, exist_ok=True)
        json_path = os.path.join(job_dir, f"{job_name}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        return json_path

    def evaluate_apo(
        self,
        sequence: str,
        step: int,
        variant: int,
        num_seeds: int = 1,
    ) -> Optional[str]:
        """Evaluate a sequence without any ligands to get the predicted Apo state."""
        job_name = f"step_{step}_variant_{variant}_apo"
        json_path = self._create_apo_input_json(sequence, job_name, num_seeds=num_seeds)
        
        output_dir = os.path.join(self.prediction_dir, job_name)
        os.makedirs(output_dir, exist_ok=True)
        success = self._run_af3(json_path, output_dir, job_name)
        
        if not success:
            logger.warning(f"Apo AF3 failed for {job_name}")
            return None
            
        seed_dirs = self._find_all_seed_outputs(output_dir)
        if not seed_dirs:
            summary = self._find_summary_file(output_dir)
            if summary:
                seed_dirs = [(0, output_dir, summary)]
                
        best_ptm = 0.0
        best_apo_cif = None
        
        for seed_idx, seed_dir, summary_path in seed_dirs:
            try:
                metrics = self._parse_summary_file(summary_path)
                if metrics is None:
                    continue
                ptm = metrics.get("ptm", 0.0)
                if ptm >= best_ptm:
                    best_ptm = ptm
                    best_apo_cif = self._find_structure_in_dir(seed_dir)
            except Exception as e:
                logger.warning(f"Apo seed {seed_idx} parse failed: {e}")
                
        return best_apo_cif

    def evaluate_decoys(
        self,
        sequence: str,
        decoy_smiles_list: List[str],
        step: int,
        variant: int,
        num_seeds: int = 5,
        open_ref_ca: Optional["np.ndarray"] = None,
        closed_ref_ca: Optional["np.ndarray"] = None,
        design_chain_id: str = "A",
        filter_config: Optional[FilterConfig] = None,
    ) -> List[DecoyResult]:
        """Evaluate a sequence against decoy/interferent ligands.

        For each decoy SMILES, runs AF3 with fewer seeds and checks:
        1. Decoy iPTM must be low (no strong binding)
        2. Decoy must NOT cause conformational closing

        Args:
            sequence: The designed protein sequence.
            decoy_smiles_list: List of decoy SMILES strings.
            step: Current step index.
            variant: Variant index.
            num_seeds: Seeds per decoy (fewer than target for speed).
            open_ref_ca: Open/apo reference Cα coordinates.
            closed_ref_ca: Closed/bound reference Cα coordinates.
            design_chain_id: Chain ID for Cα extraction.
            filter_config: Filter thresholds.

        Returns:
            List of DecoyResult, one per decoy.
        """
        decoy_results = []
        if not decoy_smiles_list:
            return decoy_results

        fc = filter_config or FilterConfig()

        for d_idx, decoy_smi in enumerate(decoy_smiles_list):
            job_name = f"step_{step}_variant_{variant}_decoy{d_idx}"

            # Create decoy input JSON
            json_path = self._create_decoy_input_json(
                sequence, decoy_smi, job_name, num_seeds=num_seeds
            )

            # Run AF3
            output_dir = os.path.join(self.prediction_dir, job_name)
            os.makedirs(output_dir, exist_ok=True)
            success = self._run_af3(json_path, output_dir, job_name)

            dr = DecoyResult(decoy_smiles=decoy_smi)
            best_decoy_cif = None

            if not success:
                # Fail closed: a decoy run we could not evaluate must not be
                # counted as evidence of specificity, or a non-specific binder
                # could slip through the screen on infrastructure flakiness.
                logger.warning(
                    f"Decoy AF3 failed for {job_name}; marking specificity as NOT passed (fail-closed)"
                )
                dr.passed_specificity = False
                decoy_results.append(dr)
                continue

            # Parse all seed outputs
            seed_dirs = self._find_all_seed_outputs(output_dir)
            if not seed_dirs:
                summary = self._find_summary_file(output_dir)
                if summary:
                    seed_dirs = [(0, output_dir, summary)]

            best_iptm = 0.0
            min_closed_rmsd = float("inf")  # worst-case (lowest) closed_rmsd for decoy
            parsed_seeds = 0

            for seed_idx, seed_dir, summary_path in seed_dirs:
                try:
                    metrics = self._parse_summary_file(summary_path)
                    if metrics is None:
                        continue
                    parsed_seeds += 1

                    iptm = metrics.get("iptm", 0.0)
                    if iptm > best_iptm:
                        best_iptm = iptm
                        best_decoy_cif = self._find_structure_in_dir(seed_dir)

                    # For decoy, we care about the smallest RMSD to closed ref (does it close?)

                    # Check closed_rmsd: decoy should NOT cause closing
                    cif_path = self._find_structure_in_dir(seed_dir)
                    if cif_path and closed_ref_ca is not None and len(closed_ref_ca) > 0:
                        try:
                            pred_ca = extract_ca_from_cif(cif_path, design_chain_id)
                            if len(pred_ca) > 0:
                                closed_rmsd = calculate_rmsd(pred_ca, closed_ref_ca, align=True)
                                min_closed_rmsd = min(min_closed_rmsd, closed_rmsd)
                        except Exception as e:
                            logger.debug(f"Decoy RMSD failed for seed {seed_idx}: {e}")

                except Exception as e:
                    logger.warning(f"Decoy seed {seed_idx} parse failed: {e}")

            # Check specificity criteria
            dr.best_iptm = best_iptm
            dr.min_closed_rmsd = min_closed_rmsd
            dr.structure_path = best_decoy_cif

            if parsed_seeds == 0:
                # AF3 "succeeded" but produced no parseable seed summary; we have
                # no evidence the variant is specific against this decoy, so fail
                # closed rather than certify specificity on default values.
                logger.warning(
                    f"Decoy {job_name}: no parseable AF3 seed output; marking "
                    f"specificity as NOT passed (fail-closed)"
                )
                dr.passed_specificity = False
                decoy_results.append(dr)
                continue

            # Evaluate specificity
            dr.causes_closing = (min_closed_rmsd <= fc.decoy_closed_rmsd_min)
            dr.passed_specificity = (
                best_iptm <= fc.decoy_iptm_max and not dr.causes_closing
            )

            logger.info(
                f"    Decoy {d_idx} ({decoy_smi[:30]}): "
                f"iptm={best_iptm:.3f} closed_rmsd={min_closed_rmsd:.2f} "
                f"{'✓ SPECIFIC' if dr.passed_specificity else '✗ FAIL'}"
            )

            decoy_results.append(dr)

        return decoy_results

    @staticmethod
    def _resolve_container_runtime(configured: str) -> str:
        """Resolve the container runtime binary.

        "auto" (or empty) detects apptainer/singularity on PATH, preferring
        apptainer. An explicit value is used as-is (falling back to PATH lookup
        if it is a bare name that is not directly executable).
        """
        if configured and configured not in ("auto", "singularity", "apptainer"):
            return configured
        if configured in ("singularity", "apptainer"):
            found = shutil.which(configured)
            if found:
                return found
        for candidate in ("apptainer", "singularity"):
            found = shutil.which(candidate)
            if found:
                return found
        # Last resort: return whatever was configured (or a sensible default)
        return configured if configured not in ("", "auto") else "apptainer"

    def cleanup_prediction_dir(self, prediction_dir: str) -> None:
        """Delete AF3 prediction directory to save disk space.

        Called after all results have been extracted and logged.
        """
        try:
            if os.path.isdir(prediction_dir):
                shutil.rmtree(prediction_dir)
                logger.debug(f"Cleaned up: {prediction_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {prediction_dir}: {e}")

    def _run_af3(self, json_path: str, output_dir: str, job_name: str) -> bool:
        """Run AlphaFold 3 via Singularity/Apptainer or direct Python."""
        cfg = self.config

        # Free GPU memory before running AF3 (JAX will need it)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cfg.af3_sif:
            # Container mode: run via Singularity/Apptainer
            runtime = self._resolve_container_runtime(cfg.apptainer_path)
            cmd = [
                runtime, "exec", "--nv",
                "-B", f"{os.path.dirname(json_path)}:/input",
                "-B", f"{cfg.af3_run_dir}:/af3_run",
                "-B", f"{cfg.af3_model_dir}:/model",
                "-B", f"{cfg.af3_db_dir}:/dataset",
                "-B", f"{os.path.abspath(output_dir)}:/output",
                cfg.af3_sif,
                "python", "/af3_run/run_alphafold.py",
                f"--json_path=/input/{os.path.basename(json_path)}",
                "--model_dir=/model",
                "--db_dir=/dataset",
                "--output_dir=/output",
                f"--run_data_pipeline={'true' if cfg.run_data_pipeline else 'false'}",
            ]
        else:
            # Direct mode: run AF3 as a subprocess with env_script. Paths are
            # shell-quoted so spaces/metacharacters can't break the command.
            run_script = os.path.join(cfg.af3_run_dir, "run_alphafold.py")
            shell_cmd = ""
            if cfg.af3_env_script:
                shell_cmd += f"source {shlex.quote(cfg.af3_env_script)} && "
            shell_cmd += (
                f"python {shlex.quote(run_script)}"
                f" --json_path={shlex.quote(json_path)}"
                f" --model_dir={shlex.quote(cfg.af3_model_dir)}"
                f" --output_dir={shlex.quote(os.path.abspath(output_dir))}"
                f" --run_data_pipeline={'true' if cfg.run_data_pipeline else 'false'}"
                f" --run_inference=true"
            )
            # The data pipeline needs the genetic databases; container mode always
            # binds them, so mirror that here when the pipeline is enabled.
            if cfg.run_data_pipeline and cfg.af3_db_dir:
                shell_cmd += f" --db_dir={shlex.quote(cfg.af3_db_dir)}"
            cmd = ["bash", "-c", shell_cmd]

        logger.info(f"Running AF3 for {job_name}...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600,
            )
            if result.returncode != 0:
                logger.error(f"AF3 failed (rc={result.returncode}): {result.stderr[-500:]}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"AF3 timed out for {job_name}")
            return False
        except Exception as e:
            logger.error(f"AF3 error: {e}")
            return False

    def _find_all_seed_outputs(self, output_dir: str) -> list[tuple[int, str, str]]:
        """Find all seed output directories and their summary files.

        AF3 output structure is typically:
          output_dir/<job_name>/seed-<N>_sample-0/
            *_summary_confidences.json
            *_model.cif

        Returns:
            List of (seed_index, seed_dir_path, summary_file_path)
        """
        results = []
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith("_summary_confidences.json"):
                    summary_path = os.path.join(root, f)
                    # Extract seed index from directory name
                    dir_name = os.path.basename(root)
                    seed_idx = 0
                    if "seed-" in dir_name:
                        try:
                            seed_part = dir_name.split("seed-")[1].split("_")[0]
                            seed_idx = int(seed_part)
                        except (IndexError, ValueError):
                            pass
                    results.append((seed_idx, root, summary_path))

        results.sort(key=lambda x: x[0])
        return results

    def _parse_results(self, output_dir: str) -> Optional[dict]:
        """Find and parse summary_confidences.json from AF3 output (legacy)."""
        summary_path = self._find_summary_file(output_dir)
        if not summary_path:
            logger.warning(f"No summary_confidences.json found in {output_dir}")
            return None
        return self._parse_summary_file(summary_path)

    def _parse_summary_file(self, summary_path: str) -> Optional[dict]:
        """Parse a single summary_confidences.json file.

        The interface-confidence metrics (``iptm``, ``ptm``) are required: if the
        file is missing them the schema/layout is wrong, and we return None
        (a parse failure) rather than silently substituting a low score that
        would look like a genuinely bad prediction. ``has_clash`` /
        ``ranking_score`` are treated as optional with conservative defaults.
        """
        try:
            with open(summary_path) as f:
                data = json.load(f)
            missing = [k for k in ("iptm", "ptm") if k not in data]
            if missing:
                logger.error(
                    f"AF3 summary {summary_path} is missing required field(s) "
                    f"{missing}; treating as a parse failure."
                )
                return None
            return {
                "iptm": float(data["iptm"]),
                "ptm": float(data["ptm"]),
                "has_clash": float(data.get("has_clash", 1.0)),
                "ranking_score": float(data.get("ranking_score", 0.0)),
                "mean_pae": self._extract_mean_pae(data),
                "chain_ptm": data.get("chain_ptm", []),
            }
        except Exception as e:
            logger.error(f"Failed to parse summary: {e}")
            return None

    def _extract_mean_pae(self, data: dict) -> float:
        """Mean inter-chain PAE between the designed chain and every other chain.

        Uses ``chain_pair_pae_min`` (a square matrix indexed by chain order in
        the template) and the configured ``design_chain_index`` rather than
        assuming the interface is chains 0/1, so it stays correct when the
        designed chain is not first or when extra chains/ligands are present.
        """
        m = data.get("chain_pair_pae_min", [])
        n = len(m)
        if n < 2:
            return PAE_MAX
        i = self.config.design_chain_index
        if not (0 <= i < n):
            i = 0  # fall back to the first chain if the index is out of range
        vals = []
        for j in range(n):
            if j == i:
                continue
            if i < len(m) and j < len(m[i]):
                vals.append(m[i][j])
            if j < len(m) and i < len(m[j]):
                vals.append(m[j][i])
        return sum(vals) / len(vals) if vals else PAE_MAX

    def _calculate_reward(self, metrics: dict) -> float:
        """Calculate weighted reward from structural metrics."""
        w = self.config.reward_weights
        pae_reward = 1.0 - (metrics["mean_pae"] / PAE_MAX)
        clash_penalty = -1.0 if metrics["has_clash"] > 0.5 else 0.0

        reward = (
            w.iptm * metrics["iptm"]
            + w.ptm * metrics["ptm"]
            + w.pae * pae_reward
            + w.clash_penalty * clash_penalty
        )

        # Optional RMSD rewards for fine-tuning conformational change
        if metrics.get("open_rmsd") is not None and w.open_rmsd != 0:
            # Reward higher RMSD vs open ref (deviating from apo)
            # Max expected RMSD is around 10A, normalize or cap? 
            # Let's use raw Å with a weight or a tanh-like scale.
            reward += w.open_rmsd * min(metrics["open_rmsd"], 10.0) / 10.0
        
        if metrics.get("closed_rmsd") is not None and w.closed_rmsd != 0:
            # Reward lower RMSD vs closed ref (reaching holo)
            reward += w.closed_rmsd * (1.0 - min(metrics["closed_rmsd"], 10.0) / 10.0)

        # Clamp to [0, 1] for a stable, comparable reward scale. Note: if many
        # variants in a group saturate at 1.0, the group std collapses and GRPO
        # advantages vanish for that step (no gradient). With the default
        # weights this is rare in practice (the PAE term keeps designs below the
        # ceiling), but raise the ceiling or rescale the weights if you observe
        # advantages flat-lining late in training.
        return max(0.0, min(1.0, reward))

    @staticmethod
    def _find_summary_file(directory: str) -> Optional[str]:
        """Recursively find the summary_confidences.json file."""
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith("_summary_confidences.json"):
                    return os.path.join(root, f)
        return None

    @staticmethod
    def _find_structure_path(directory: str) -> Optional[str]:
        """Find a predicted structure CIF file.

        AF3 writes a top-level ``*_model.cif`` (the ranked best) plus per-seed
        ``*_sample-*.cif`` files, so match either ``model`` or ``sample`` —
        matching only ``sample`` would miss the ranked model file.
        """
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(".cif") and ("model" in f or "sample" in f):
                    return os.path.join(root, f)
        return None

    @staticmethod
    def _find_structure_in_dir(directory: str) -> Optional[str]:
        """Find a CIF model file in a specific directory."""
        for f in os.listdir(directory):
            if f.endswith(".cif") and ("model" in f or "sample" in f):
                return os.path.join(directory, f)
        # Fallback: any CIF file
        for f in os.listdir(directory):
            if f.endswith(".cif"):
                return os.path.join(directory, f)
        return None
