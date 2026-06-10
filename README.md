<h1 align="center">BetterMPNN</h1>

BetterMPNN is a reinforcement learning driven framework for protein sequence deep optimization. We use **Group Relative Policy Optimization (GRPO)** to fine-tune ProteinMPNN in specific tasks, enabling efficient protein sequence design through an exploration-evaluation-optimization loop. With a pluggable structure prediction environment as the reward signal, the framework can complete the design process from backbone to high-performance binding proteins within hours.

## Design

Current protein design tools can generate sequences in a forward pass but have no mechanism to learn from downstream evaluation. BetterMPNN bridges this gap by introducing a reinforcement learning feedback loop: an **Environment** (any structure predictor) scores the generated sequences, and **GRPO** uses those scores to iteratively update ProteinMPNN — enabling the model to learn from evaluation outcomes and progressively converge toward higher-quality designs.

![BetterMPNN Workflow](assets/images/workflow.webp)

The three components:

- **Agent (ProteinMPNN)** — Generates sequences under fixed-backbone constraints
- **Environment** — Black-box scoring: `sequence → scalar reward`. Pluggable — AF3, Protenix, ESMFold, or any custom predictor
- **GRPO** — Computes group-relative advantages and updates the agent with KL-regularized policy gradient

Each step: sample N sequences → score them → compute advantages relative to the group mean → backpropagate GRPO loss → repeat.

**Two modes**

- **`train`** — GRPO fine-tuning. Updates ProteinMPNN so it converges toward higher-reward sequences for the given target.
- **`sample`** — Large-scale screening with a *frozen* model. Generates many variants, evaluates **every AF3 seed** per variant, and applies iPTM / conformational-RMSD / decoy-specificity filters to build a **screening pool** of candidates for experimental validation. Parallelizable across GPUs via a SLURM job array.

Both modes support **small-molecule ligands** (SMILES), **residue-level design scope** (redesign only chosen positions), and **conformational-change rewards** (open↔closed via reference RMSD).

**Reward Function (AlphaFold 3 example)**

The included AF3 environment combines: `Reward = a * (1 - PAE / PAE_MAX) + b * ipTM + c * pTM - d * clash_penalty`

By pre-computing the target's MSA and running AF3 in inference-only mode (no data pipeline), each evaluation takes on the order of a minute on a modern data-center GPU; exact timing depends on the GPU, sequence lengths, and the number of diffusion samples and recycles.

## Results

The following are wet-lab validation results from our own application of BetterMPNN (ShanghaiTech-iGEM-2025). The candidate sequences and assay data belong to that project and are not reproducible from this repository alone; the numbers are reported here for context.

**Binder design — GZMK binder for a colloidal-gold test strip.** Candidates were drawn at random from the final sequence pool **without any virtual screening** and characterized by surface plasmon resonance (SPR): **10 of 12 (83%)** bound with micromolar-level affinity. A conventional RFdiffusion pipeline on the same target gave 2 of 57 (3.5%) under comparable selection.

**Odorant-binding protein (OBP) for gas sensing.** Applied to OBP design as a gas-sensitive material, producing designs with micromolar-level affinity for the target odorant molecules.

**Training Trajectories**

(a)–(b): suboptimal backbones — noisy rewards, poor convergence. (c)–(d): viable backbones — clear reward improvement over training.

![Training Trajectories](assets/images/training_trajectories.webp)

**Included Example**

| | Details |
|:--|:--|
| **Target** | GZMK (238 aa, chain B) |
| **Starting binder** | Helical peptide (58 aa, chain A) |
| **Scaffold** | `examples/scaffold.pdb` |
| **Baseline** | iPTM = 0.64, pTM = 0.86, ranking_score = 0.69 |

## Usage

**Installation**

```bash
git clone https://github.com/Terry-Wang-Lynx/BetterMPNN.git
cd BetterMPNN
conda create -n bettermpnn python=3.11
conda activate bettermpnn
pip install -r requirements.txt
```

Download ProteinMPNN weights:

```bash
mkdir -p weights/vanilla
wget -P weights/vanilla \
    https://files.ipd.uw.edu/pub/training_sets/ProteinMPNN/v_48_020.pt
```

**Run**

```bash
python -m bettermpnn.cli --config configs/example.yaml
```

Override from CLI:

```bash
python -m bettermpnn.cli --config configs/example.yaml \
    --pdb my_scaffold.pdb --chain A --steps 200 --variants 8 --seed 0 -v
```

**Large-scale sampling (build a screening pool)**

```bash
# Single GPU
python -m bettermpnn.cli --config configs/example_sampling.yaml --mode sample

# Multi-GPU via SLURM job array (4 workers), then merge
sbatch scripts/run_sample_array.sh configs/example_sampling.yaml
bash scripts/merge_results.sh output_sampling_<job_id>
```

Sampling outputs land in `passed/` (cleared all filters incl. specificity),
`passed_structural_only/`, plus `screening_log*.csv` and `screening_summary*.json`.

**Input Files**

1. **Scaffold PDB** — target + binder complex
2. **AF3 template JSON** — sequences with pre-computed MSA (`configs/af3_template.json`)
3. **Config YAML** — training parameters (`configs/example.yaml`)

Only the target needs a rich MSA; the redesigned chain uses a single-sequence MSA. Obtain the target MSA from the [AlphaFold Server](https://alphafoldserver.com/) or Jackhmmer and paste it into the corresponding chain's `unpairedMsa`/`pairedMsa` field in the template JSON (AF3 A3M format). For the redesigned chain, keep a single-sequence MSA: a two-line `>name\n<SEQUENCE>` block whose second line is the query sequence — BetterMPNN substitutes each designed sequence into that line at runtime.

**Key Parameters**

The "Default" column below shows the values used in `configs/example.yaml`; the dataclass defaults in `bettermpnn/config.py` are more conservative (e.g. `steps=10`) and are overridden by any value you set in the config.

| Parameter | Default | Description |
|:--|:--|:--|
| `mode` | `"train"` | `train` (GRPO) or `sample` (frozen screening) |
| `pdb` | required | Scaffold PDB |
| `chain` | `"A"` | Chain to redesign |
| `steps` | `200` | Training / sampling steps |
| `variants` | `8` | Sequences per step |
| `lr` | `1e-4` | Learning rate (train mode) |
| `beta` | `0.01` | KL penalty weight (train mode) |
| `temperature` | `0.3` | Sampling temperature |
| `iterative` | `false` | Update backbone each step |
| `seed` | none | Random seed for reproducibility |
| `num_seeds` | `5` | AF3 seeds per variant (sample mode) |
| `design.redesign_residues` | `[]` | Restrict design to these residues, e.g. `["A10","A11"]` (empty = whole chain) |
| `ligand.smiles` | none | Target small-molecule SMILES |
| `decoy_smiles` | `[]` | Interferent SMILES for specificity filtering (sample mode) |
| `open_ref_pdb` / `closed_ref_pdb` | none | Reference structures for conformational RMSD |

> `environment.design_chain_index` must point to the same chain (by position in the template JSON `sequences` list) that you redesign via `chain`; a length mismatch is flagged at runtime.

**Output**

```
output/
├── training_results.json   # Per-step summary
├── variant_log.csv         # Per-variant detail (sequence, score, metrics)
├── training_plot.png       # Training curves
├── mpnn_step_*.pt          # Checkpoints
└── mpnn_final.pt           # Final model
```

**Environment Setup**

AF3 (Singularity):
```yaml
environment:
  af3_sif: "/path/to/alphafold3.sif"
  af3_run_dir: "/path/to/alphafold3"
  af3_model_dir: "/path/to/model"
  af3_db_dir: "/path/to/dataset"
```

AF3 (conda, no container):
```yaml
environment:
  af3_sif: ""
  af3_run_dir: "/path/to/alphafold3"
  af3_model_dir: "/path/to/model"
  af3_env_script: "/path/to/af3_env.sh"
```

Custom environment:
```python
from bettermpnn.environment.base import Environment, EvalResult

class MyPredictor(Environment):
    def evaluate(self, sequence, step=0, variant=0):
        return EvalResult(score=my_model.predict(sequence).confidence)
```

## Development

Run the unit tests (GRPO math, config loading, RMSD — no GPU needed):

```bash
pip install -e ".[dev]"
pytest -q
```

`bettermpnn/mpnn/protein_mpnn_utils.py` is vendored from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) (MIT License) © the ProteinMPNN authors.

## References

- **ProteinMPNN:** J. Dauparas, et al. *Science* (2022). [Paper](https://www.science.org/doi/10.1126/science.add2187) | [Code](https://github.com/dauparas/ProteinMPNN)
- **AlphaFold 3:** J. Abramson, et al. *Nature* (2024). [Paper](https://www.nature.com/articles/s41586-024-07487-w) | [Code](https://github.com/google-deepmind/alphafold3)
- **GRPO:** DeepSeek-AI. (2025). [Paper](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)

## Contributors

**Design & Development**
- Tianyi Wang (王天颐)
- Yafei Chang (常雅斐)

**Wet Lab Validation**
- Tianyi Wang (王天颐)
- Yuchen Hao (郝郁晨)
- ShanghaiTech-iGEM-2025

## License

MIT License. See [LICENSE](LICENSE).
