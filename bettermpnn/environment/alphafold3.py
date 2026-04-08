"""AlphaFold 3 environment: evaluates sequences via AF3 structure prediction."""

import json
import logging
import os
import subprocess
from typing import Optional

import torch

from .base import Environment, EvalResult
from ..config import EnvironmentConfig

logger = logging.getLogger(__name__)

PAE_MAX = 31.75  # Maximum PAE value for normalization


class AlphaFold3Environment(Environment):
    """Evaluate protein sequences using AlphaFold 3 structure prediction.

    Runs AF3 in a Singularity/Apptainer container, parses confidence metrics
    from the output, and returns a weighted reward score.
    """

    def __init__(self, config: EnvironmentConfig, output_dir: str = "output"):
        self.config = config
        self.output_dir = output_dir
        self.input_dir = os.path.join(output_dir, "af3_inputs")
        self.prediction_dir = os.path.join(output_dir, "af3_predictions")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.prediction_dir, exist_ok=True)

        # Load template JSON
        with open(config.template_json) as f:
            self.template = json.load(f)

    def evaluate(self, sequence: str, step: int = 0, variant: int = 0) -> EvalResult:
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

        # 4. Calculate reward
        score = self._calculate_reward(metrics)

        # 5. Find structure path (for iterative mode)
        structure_path = self._find_structure_path(output_dir)

        return EvalResult(score=score, structure_path=structure_path, metrics=metrics)

    def _create_input_json(self, sequence: str, job_name: str) -> str:
        """Create AF3 input JSON with the designed sequence substituted in."""
        import copy
        data = copy.deepcopy(self.template)

        idx = self.config.design_chain_index
        sequences = data.get("sequences", [])
        if idx < len(sequences):
            chain = sequences[idx].get("protein", {})
            chain["sequence"] = sequence
            # Update MSA fields (replace the sequence line in each)
            for msa_key in ("unpairedMsa", "pairedMsa"):
                msa = chain.get(msa_key, "")
                if msa:
                    lines = msa.splitlines()
                    if len(lines) >= 2:
                        lines[1] = sequence
                    chain[msa_key] = "\n".join(lines)

        job_dir = os.path.join(self.input_dir, job_name)
        os.makedirs(job_dir, exist_ok=True)
        json_path = os.path.join(job_dir, f"{job_name}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        return json_path

    def _run_af3(self, json_path: str, output_dir: str, job_name: str) -> bool:
        """Run AlphaFold 3 via Singularity/Apptainer or direct Python."""
        cfg = self.config

        # Free GPU memory before running AF3 (JAX will need it)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if cfg.af3_sif:
            # Container mode: run via Singularity/Apptainer
            cmd = [
                cfg.apptainer_path, "exec", "--nv",
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
            # Direct mode: run AF3 as a subprocess with env_script
            run_script = os.path.join(cfg.af3_run_dir, "run_alphafold.py")
            shell_cmd = ""
            if cfg.af3_env_script:
                shell_cmd += f"source {cfg.af3_env_script} && "
            shell_cmd += (
                f"python {run_script}"
                f" --json_path={json_path}"
                f" --model_dir={cfg.af3_model_dir}"
                f" --output_dir={os.path.abspath(output_dir)}"
                f" --run_data_pipeline={'true' if cfg.run_data_pipeline else 'false'}"
                f" --run_inference=true"
            )
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

    def _parse_results(self, output_dir: str) -> Optional[dict]:
        """Find and parse summary_confidences.json from AF3 output."""
        summary_path = self._find_summary_file(output_dir)
        if not summary_path:
            logger.warning(f"No summary_confidences.json found in {output_dir}")
            return None
        try:
            with open(summary_path) as f:
                data = json.load(f)
            return {
                "iptm": float(data.get("iptm", 0.0)),
                "ptm": float(data.get("ptm", 0.0)),
                "has_clash": float(data.get("has_clash", 1.0)),
                "ranking_score": float(data.get("ranking_score", 0.0)),
                "mean_pae": self._extract_mean_pae(data),
                "chain_ptm": data.get("chain_ptm", []),
            }
        except Exception as e:
            logger.error(f"Failed to parse summary: {e}")
            return None

    def _extract_mean_pae(self, data: dict) -> float:
        """Extract mean inter-chain PAE from AF3 output."""
        chain_pair_pae = data.get("chain_pair_pae_min", [])
        if len(chain_pair_pae) >= 2:
            a_to_b = chain_pair_pae[0][1] if len(chain_pair_pae[0]) > 1 else PAE_MAX
            b_to_a = chain_pair_pae[1][0] if len(chain_pair_pae) > 1 else PAE_MAX
            return (a_to_b + b_to_a) / 2.0
        return PAE_MAX

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
        """Find the best predicted structure CIF file."""
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(".cif") and "sample" in f:
                    return os.path.join(root, f)
        return None
