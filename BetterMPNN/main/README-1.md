# BetterMPNN

## Overview

BetterMPNN is an innovative protein sequence optimization framework that combines **ProteinMPNN** and **AlphaFold 3**. The tool employs **Group Relative Policy Optimization (GRPO) algorithm**, using AlphaFold 3's comprehensive structural metrics as reward signals to fine-tune the ProteinMPNN model, aiming to generate protein sequence variants with improved structural properties.

### Core Features

**·Structure-Based Sequence Design**: Utilizes PDB scaffold information for informed sequence generation

**·GRPO Optimization**: Implements Group Relative Policy Optimization for stable reinforcement learning

**·Comprehensive Reward System**: Combines multiple AF3 metrics (pTM, ipTM, PAE, clashes) for balanced optimization

**·Singularity Container Support**: Runs AlphaFold 3 in isolated container environments

**·End-to-End Workflow**: Complete solution from scaffold-based design to structure evaluation

### Workflow

**1.Scaffold-Based Design**: Use ProteinMPNN to generate diverse protein sequence variants based on PDB structure (backbones generated from RFdiffusion)

**2.Structure Prediction**: Evaluate each variant's predicted structure quality through AlphaFold 3 in Singularity

**3.Multi-Metric Reward Calculation**: Extract comprehensive structural metrics as optimization targets

**4.GRPO Fine-tuning**: Iteratively optimize ProteinMPNN using GRPO

**5.Token-Level Optimization**: Apply reinforcement learning at the amino acid token level for precise control

## Installation Guide

**!! Note:** The following operations involve the use of external software. However, due to space limitations, we **will not** provide detailed installation and usage tutorials for ProteinMPNN and AlphaFold 3 here. **Please read the official documentation for detailed guidance.** As long as you can successfully use ProteinMPNN and AlphaFold 3, you can run and use this tool according to the code, structure directory, and configuration instructions we provide.

### Step 1: Project and Environment Setup

#### Create Conda Environment

```bash
# Clone the project repository and switch to the main directory of BetterMPNN
# Create environment using environment.yml (recommended method)
conda env create -f environment.yml
conda activate protein-mpnn
```

### Step 2: ProteinMPNN Setup

#### Obtain ProteinMPNN Code and Weights

```bash
# Clone the ProteinMPNN repository
git clone https://github.com/dauparas/ProteinMPNN.git

# Install ProteinMPNN
cd ProteinMPNN
pip install

# Verify installation
python -c "import protein_mpnn; print('ProteinMPNN installation successful')"

# Ensure protein_mpnn_utils.py and model_utils.py are in scripts directory

# Download pre-trained weights to vanilla_model_weights directory
# Available weights: v_48_002.pt, v_48_010.pt, v_48_020.pt, v_48_030.pt

# Download pre-trained weights to soluble_model_weights directory
# Available weights: v_48_002.pt, v_48_010.pt, v_48_020.pt, v_48_030.pt
```

### Step 3: AlphaFold 3 Setup with Singularity

#### System Requirements

##### Hardware Requirements

1. Computing Device: A Linux system must be used (Ubuntu 22.04 LTS is recommended).
2. GPU: NVIDIA GPU (e.g., RTX 5090, A100, H100 80GB or higer).
3. Memory: At least 64GB of RAM (more is required when processing long - sequence targets).
4. Storage: It is recommended to have 1TB of SSD space for storing the genetic database.

##### Software dependencies

* CUDA 12.6
* cuDNN
* Docker or Singularity
* NVIDIA Driver

**Note:**

* Please refer to the official installation.md of AlphaFold3 for pre - installation and configuration.

  ```
  https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md
  ```
* We used Singularity, so the specific code will be based on Singularity

#### Obtaining AlphaFold 3 Source Code

Install git and download the AlphaFold 3 repository:

```bash
git clone https://github.com/google-deepmind/alphafold3.git
```

#### Obtain AlphaFold 3 Singularity Image

```bash
# Obtain AlphaFold 3 Singularity image (.sif file)
# Place it in your designated directory

# Verify Singularity installation
singularity --version
```

#### Obtain Model Parameters

> **Important**: AlphaFold 3 model parameters must be obtained following official procedures
>
> 1. Visit: https://github.com/google-deepmind/alphafold3
> 2. Follow the official process to request model parameter access
> 3. Download model parameters to the specified directory

#### Download Genetic Databases

```bash
# Use official script to download databases
./fetch_databases.sh [<database_directory>]

# Or specify custom directory
./fetch_databases.sh /path/to/your/databases
```

### Step 4: Project Configuration

#### Path Configuration

Edit `scripts/run_bettermpnn.py` and update the following paths:

```python
# Update to your actual paths
BASE_DIR = "/path/to/your/project/root"  # You must modify
AF3_MODEL_DIR = "/path/to/your/alphafold3/model"  # You must modify
AF3_DB_DIR = "/path/to/your/alphafold3/databases"  # You must modify
AF3_SIF_PATH = "/path/to/your/alphafold3.sif"  # You must modify
SINGULARITY_PATH = "/path/to/your/singularity/bin/singularity"  # You must modify

# Scaffold PDB path (relative to project root, no modification needed if following structure)
SCAFFOLD_PDB_PATH = os.path.join(os.path.dirname(__file__), '../input/fold_2_620.pdb')
PATH_TO_MODEL_WEIGHTS = os.path.join(os.path.dirname(__file__), '../vanilla_model_weights/v_48_020.pt')
```

#### Input PDB Preparation

Place your scaffold PDB file in the `input/` directory:

```bash
# Ensure your scaffold PDB is available
cp your_scaffold.pdb input/XXX.pdb
```

#### JSON Template Configuration

Configure `config/test.json` for your target complex:

```json
{
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "YOUR_BINDING_PARTNER_SEQUENCE",
        "unpairedMsa": ""
      }
    },
    {
      "protein": {
        "id": "B", 
        "sequence": "YOUR_TARGET_SEQUENCE_TO_OPTIMIZE",
        "unpairedMsa": ""
      }
    }
  ]
}
```

## Usage Guide

### Basic Usage

#### Quick Start

```bash
# Ensure environment is activated
conda activate protein-mpnn

# Run the optimization workflow
python scripts/run_bettermpnn.py
```

Referring to the script we actually use, a `run.sh` is provided here as the script for submitting jobs on the cluster (a Slurm cluster.). After submitting this script, run_bettermpnn.py will automatically start running.

#### Custom Scaffold and Design Chain

Modify the configuration in `scripts/run_bettermpnn.py`:

```python
# Change scaffold PDB file
SCAFFOLD_PDB_PATH = "input/your_scaffold.pdb"

# Change chain to design
CHAIN_ID_TO_DESIGN = "B"  # Modify to your target chain
```

### Advanced Configuration

#### Optimization Parameter Tuning

Modify training parameters in `scripts/run_bettermpnn.py`:

```python
# Training control parameters
TRAINING_STEPS = 3000          # Total training steps, 
NUM_GENERATIONS = 8            # Number of variants generated per step

# GRPO algorithm parameters
BETA = 0.01                    # KL divergence weight
ADVANTAGE_SCALE_FACTOR = 5.0   # Advantage scaling factor
REWARD_SHAPING_ALPHA = 0.7     # Reward shaping exponent
SAMPLING_TEMPERATURE = 0.3     # Sampling temperature

# Optimization parameters
LEARNING_RATE = 1e-4           # Learning rate
SAVE_CHECKPOINT_EVERY = 15     # Checkpoint frequency
```

#### Reward Weight Configuration

Customize the reward function in `scripts/reward_utils.py`:

```python
# Adjust weights for different structural metrics
# Here is a set of examples.
REWARD_WEIGHTS = {
    'pae': 0.4,         # Predicted Aligned Error
    'iptm': 0.55,        # Interface TM-score
    'ptm_reward': 0.05,         # Protein TM-score
    'clash_penalty': 0.1      # Steric clash penalty
}
```

**In fact, you can define your own reward function, including its composition and parameters.** Since our code related to the reward function is presented in an independent Python file (scripts/reward_utils.py), you can replace the code inside with your own defined reward function.   Please make sure the path and parameter passing are correct. Note that our default reward function is composed of the prediction metrics of AlphaFold 3. If you want to design your own reward function, you also need to modify the relevant content in run_bettermpnn.py synchronously.

#### MSA Configuration (Optional Optimization)

**Critical Finding:** For small binder design tasks, only the target protein (Binding Partner) requires MSA to achieve good results, significantly reducing computational cost and time.

Therefore, our project supports two MSA processing methods:

##### Option 1: Skip MSA Computation (Recommended for Performance)

If you have pre-computed MSA data, you can configure to skip MSA computation:

**1.Prepare MSA Data**

·Ensure MSA uses A3M format

·First sequence must be the original sequence

**Note:** If you are unsure how to quickly obtain MSA information, we recommend using the online tool AlphaFold Server (https://alphafoldserver.com/) to predict the two proteins and replace the resulting MSA information into the JSON template file as required.

**2.Configure Input JSON** (`config/test.json`)

The JSON file contains two protein sequences that need to be configured separately:

**- Sequence A (Binding Partner Protein)**

·If you want to obtain the optimized binding protein for the id=A protein, replace the "sequence" field with your target protein sequence

·If you have MSA information and want to skip AF3's MSA calculation process, fill in your existing MSA information in the "unpairedMsa" field

**- Sequence B (Protein to be Optimized)**

·For the protein with id=B that you want to optimize, replace the "sequence" field with your protein sequence to be optimized

·If you have MSA information and want to skip AF3's MSA calculation process, fill in your existing MSA information in the "unpairedMsa" field

```json
{
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "YOUR_BINDING_PARTNER_SEQUENCE",
        "unpairedMsa": ">A\nYOUR_BINDING_PARTNER_SEQUENCE\nOther MSA sequences..."
      }
    },
    {
      "protein": {
        "id": "B", 
        "sequence": "YOUR_TARGET_SEQUENCE_TO_OPTIMIZE",
        "unpairedMsa": ">B\nYOUR_TARGET_SEQUENCE_TO_OPTIMIZE\nOther MSA sequences..."
      }
    }
  ]
}
```

**3.Enable Skip MSA Mode**: Ensure AlphaFold 3 call includes `--run_data_pipeline=false`

##### Option 2: Use AlphaFold 3 MSA Computation (Standard Mode)

If you want AlphaFold 3 to automatically compute MSA:

**1.Configure Input JSON** (`config/test.json`)

Only configure sequence information:

```json
{
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "YOUR_BINDING_PARTNER_SEQUENCE"
      }
    },
    {
      "protein": {
        "id": "B", 
        "sequence": "YOUR_TARGET_SEQUENCE_TO_OPTIMIZE"
      }
    }
  ]
}
```

**2.Modify Run Script**

In the AlphaFold 3 call section of `scripts/run_bettermpnn.py`, change:

```python
--run_data_pipeline=false
```

to:

```python
--run_data_pipeline=true
```

**3.Ensure Database Configuration is Correct**

Verify the following database paths are available on your system:

```python
AF3_DB_DIR = "/path/to/your/alphafold3/databases"
```

**Important Notes**:

·Using standard MSA mode will significantly increase computation time (each variant may take several hours)

·Ensure sufficient computational resources and time budget

·For rapid prototyping and testing, we recommend using Option 1 (skip MSA)

### Output Files Description

After completion, the following outputs will be generated:

```
project_root/
├── rl_checkpoint/               # Model checkpoints
│   ├── mpnn_model_step_XX.pt    # Periodically saved models
│   └── mpnn_model_final.pt      # Final trained model
├── rl_graph/                    # Training process visualization
│   └── grpo_progress_step_*.png # Training charts for each step
├── rl_rewards_log.csv           # Detailed reward metrics records
├── af3_summary_log.txt          # AF3 summary confidence logs
├── rl_prediction/               # AlphaFold 3 prediction results
│   └── step_*_variant_*/        # Structure predictions for each variant
└── rl_input/                    # AlphaFold 3 input files
    └── step_*_variant_*/        # Input configurations for each variant
```

### Test Your Trained Model

After the training is completed, you can test the model (the .pt file in the directory rl_checkpoint) saved during the training process through the following operations:

1.Put the `checkpoint/mpnn_model_step_XX.pt` generated during training into the `rl_model` folder

2.Modify the path and model file name in `scripts/test-model-1.py` to custom paths and models:

```python
# Put your checkpoint-model into the 'rl_model' directory
PATH_TO_MODEL_WEIGHTS = os.path.join(os.path.dirname(__file__), '../rl_model/mpnn_model_step_XX.pt')  # Your checkpoint-model
```

3.run `scripts/test_model.sh`

4.You will obtain the sequence results generated by the trained model in one round and the reward evaluation

## Core Algorithm Features

### Group Relative Policy Optimization (GRPO)

The key innovation of this project is the **Group Relative Policy Optimization**, which provides stable reinforcement learning for protein sequence design:

#### Implementation Principle

1.**Group-Based Advantage Calculation**: Compute advantages relative to group performance rather than absolute rewards

2.**Reward Shaping**: Apply non-linear transformation to increase reward differentiation

3.**Token-Level Policy Optimization**: Optimize at the individual amino acid level for precise control

4.**KL Divergence Regularization**: Maintain policy stability through divergence constraints

#### Algorithm Advantages

**·Stable Training**: Group-relative advantages reduce variance and improve convergence

**·Fine-Grained Control**: Token-level optimization enables precise sequence engineering

**·Multi-Metric Optimization**: Balanced reward function considers multiple structural properties

**·Containerized Execution**: Singularity support ensures reproducible AF3 execution

### Comprehensive Reward System

Advanced reward calculation combining multiple AlphaFold 3 metrics:

**·pTM Score**: Global structure quality assessment

**·ipTM Score**: Interface quality for complexes

**·PAE Metrics**: Local structural confidence

**·Clash Detection**: Steric hindrance penalty

**·Custom Weighting**: Configurable balance between different metrics

## Contributing

We welcome and encourage contributions to this project! Whether you want to report bugs, suggest new features, or directly submit code improvements, we greatly appreciate your input.

### How to Contribute

#### Reporting Issues

If you find bugs or have improvement suggestions:

1.Create a new Issue on GitLab

2.Describe the problem or suggestion in detail

3.For bugs, please provide:

·System environment information (OS, CUDA version, Singularity version, etc.)

·Error logs or screenshots

·Steps to reproduce

### Development Guidelines

#### Code Style

·**Python Code**: Follow PEP 8 standards

·**Variable Naming**: Use descriptive variable names

·**Comments**: Add clear comments for complex logic

·**Docstrings**: Add docstring descriptions for functions

### Code of Conduct

To maintain a friendly, inclusive open source community, we expect all participants to:

·Respect different viewpoints and experience levels

·Accept constructive criticism

·Focus on what is best for the community

·Show empathy and patience towards other community members

### Contact

If you have any questions or need help:

·Create GitLab Issues for public discussion

·Contact the maintenance team through the project repository

Thank you for contributing to protein sequence optimization research!

## References

### Core Papers

**1.ProteinMPNN:** J. Dauparas, et al. "Robust deep learning-based protein sequence design using ProteinMPNN." *Science* (2022).

·Paper: https://www.science.org/doi/10.1126/science.add2187

·Code: https://github.com/dauparas/ProteinMPNN

**2.AlphaFold 3:** Josh Abramson, et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature* (2024).

·Paper: https://www.nature.com/articles/s41586-024-07487-w

·Code: https://github.com/google-deepmind/alphafold3

### Algorithmic Foundation (GRPO)

**3.DeepSeek-R1:** DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." *arXiv* (2025).

·Paper: https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf

·Code: https://github.com/deepseek-ai/DeepSeek-R1

**4.DeepSeek-Math:** Zhihong Shao, et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." *arXiv* (2024).

·Paper: https://arxiv.org/abs/2402.03300

·Code: https://github.com/deepseek-ai/DeepSeek-Math

License and Acknowledgments

### License

This project is released under the [CC BY 4.0](LICENSE.txt) license.

### Acknowledgments

**·ProteinMPNN Developers**: For developing and open-sourcing ProteinMPNN

**·Google DeepMind**: For creating AlphaFold 3 and providing model access

**·ShanghaiTech University**: For providing computational resources and support

### Authors

**·Tianyi Wang**: the core strategist of our project, who provided a large number of high-quality ideas (we have suggested that he buy a T-shirt and a coffee cup, even though he majors in biology)

**·Yafei Chang**: has something to say to some thing: "Thank you, my penguin doll. You accompanied me while debugging code late at night. I really want to include you in my list of thanks"

**·Claude** (our AI Assistant): special thanks to Claude for its invaluable assistance in code writing and debugging, providing concrete technical support throughout the development process

---

## Disclaimer

1.This software requires access to AlphaFold 3 model parameters; please ensure proper authorization from Google DeepMind before use

2.This software requires Singularity/Apptainer for containerized execution

3.This software is provided "as is" without any express or implied warranties

4.Users are responsible for validating results for their specific applications

5.Please comply with all applicable software licenses and terms of use

---

**Contact:** For technical questions or suggestions, please contact us through GitHub Issues or the project repository.
