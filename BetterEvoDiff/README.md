# BetterEvoDiff

In addition to BetterMPNN, we also provide another tool called **BetterEvoDiff**.

## Overview

BetterEvoDiff is an innovative protein sequence optimization framework that combines **EvoDiff** and **AlphaFold 3**. The tool employs **reinforcement learning strategies**, using AlphaFold 3's ranking scores as reward signals to fine-tune the EvoDiff model, aiming to generate protein sequence variants with higher predicted structural stability.

### Core Features

·**Sequence Space Optimization**: Direct optimization in protein sequence space without requiring structural information

·**Multi-Path Estimation Strategy**: Uses multi-path decoding to reduce gradient estimation variance and improve training stability

·**MSA Computation Bypass**: Optional performance optimization that significantly reduces computation time

·**Reinforcement Learning Fine-tuning**: Employs policy gradient methods with KL divergence regularization

·**End-to-End Workflow**: Complete solution from sequence generation to structure evaluation

### Workflow

**1.Sequence Generation:** Use EvoDiff to generate diverse protein sequence variants

**2.Structure Prediction:** Evaluate each variant's predicted structure quality through AlphaFold 3

**3.Reward Calculation:** Extract AlphaFold 3's ranking scores as optimization targets

**4.Model Fine-tuning:** Iteratively optimize EvoDiff using reinforcement learning methods

**5.Multi-Path Training:** Generate multiple decoding paths for each variant to stabilize gradient estimation

## Installation Guide

**!! Note:** The following operations involve the use of external software. However, due to space limitations, we **will not** provide detailed installation and usage tutorials for EvoDiff and AlphaFold 3 here. **Please read the official documentation for detailed guidance.** As long as you can successfully use EvoDiff and AlphaFold 3, you can run and use this tool according to the code, structure directory, and configuration instructions we provide.

### Step 1: Project and Environment Setup

```bash
# Clone the project repository and switch to the main directory of BetterEvoDiff
# Create a Conda environment using environment.yml (recommended method)
conda env create -f environment.yml
conda activate evodiff
```

### Step 2: EvoDiff Installation

```bash
# Install EvoDiff from the official repository
pip install git+https://github.com/microsoft/evodiff.git

# Verify installation
python -c "from evodiff.pretrained import OA_DM_38M; print('EvoDiff installation successful')"
```

### Step 3: AlphaFold 3 Setup

#### Obtaining AlphaFold 3 Source Code

Install git and clone the AlphaFold 3 repository:

```bash
# Clone into a directory of your choice, for example, alongside your project
git clone https://github.com/google-deepmind/alphafold3.git /path/to/your/alphafold3
cd /path/to/your/alphafold3
```

Install AlphaFold 3 Dependencies: Refer to the official AlphaFold 3 documentation for the most accurate installation commands.

#### Obtain Model Parameters

> **Important**: AlphaFold 3 model parameters must be obtained directly from Google DeepMind
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

Configure your project to use the installed tools and data.

#### Path Configuration

Edit `scripts/run_betterevodiff.py` and update the following paths:

```python
# Update to your actual paths
AF3_MODEL_DIR = "/path/to/your/alphafold3/model"
AF3_DB_DIR = "/path/to/your/alphafold3/databases"  
PDB_DATABASE_PATH = "/path/to/your/pdb/dataset"
```

#### Run Script Configuration

Edit `run.sh` to adjust for your cluster environment:

```bash
# Update proxy settings (if needed)
export http_proxy=your_proxy_settings
export https_proxy=your_proxy_settings

# Update module loading
module load your_required_modules

# Update conda environment path
source activate evodiff
```

## Usage Guide

### Basic Usage

#### Quick Start

```bash
# Ensure environment is activated
conda activate evodiff

# Run complete optimization workflow
chmod +x run.sh
./run.sh
```

#### Custom Input Sequence

Edit the base sequence in `scripts/run_betterevodiff.py`:

```python
# Replace with your target sequence to optimize
BASE_SEQUENCE = "YOUR_PROTEIN_SEQUENCE_HERE"
```

### Advanced Configuration

#### Optimization Parameter Tuning

Modify training parameters in `scripts/run_betterevodiff.py`:

```python
# Training control parameters
TRAINING_STEPS = 3000          # Total training steps
NUM_GENERATIONS = 8            # Number of variants generated per step
NUM_MUTATIONS = 10             # Number of positions to mutate each time

# Multi-path strategy parameters
NUM_PATHS_PER_VARIANT = 16     # Number of paths per variant

# Loss function parameters
BETA = 0.1                     # KL divergence weight
LEARNING_RATE = 1e-5           # Learning rate
```

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

Only configure sequence information, leave MSA fields empty or set to empty strings:

```json
{
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "YOUR_BINDING_PARTNER_SEQUENCE",
        "unpairedMsa": "",
        "pairedMsa": ""
      }
    },
    {
      "protein": {
        "id": "B", 
        "sequence": "YOUR_TARGET_SEQUENCE_TO_OPTIMIZE",
        "unpairedMsa": "",
        "pairedMsa": ""
      }
    }
  ]
}
```

**2.Modify Run Script**

In the AlphaFold 3 call section of `scripts/run_betterevodiff.py`, change:

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
├── checkpoints/              # Model checkpoints
│   ├── model_step_XX.pth     # Periodically saved models
│   └── model_final.pth       # Final trained model
├── training_graphs/          # Training process visualization
│   └── progress_step_*.png   # Training charts for each step
├── rewards_log.csv           # Reward score records with sequences
├── af3_summary_log.txt       # AF3 summary confidence logs
├── prediction/               # AlphaFold 3 prediction results
│   └── step_*_variant_*/     # Structure predictions for each variant
└── input/                    # AlphaFold 3 input files
    └── step_*_variant_*/     # Input configurations for each variant
```

### Test Your Trained Model

After the training is completed, you can test the model (the `.pt` file in the `rl_checkpoint` folder) saved during the training process through the following operations:

1.Put the `checkpoint/model_step_XX.pth` generated during training into the `rl_model` folder

2.Modify the path and model file name in `scripts/test-model-2.py` to custom paths and models:

```python
# Put your checkpoint-model into the 'rl_model' directory
PATH_TO_MODEL_WEIGHTS = os.path.join(os.path.dirname(__file__), '../rl_model/model_step_XX.pt')  # Your checkpoint-model
```

3.run `scripts/test_model.sh`

4.You will obtain the sequence results generated by the trained model in one round and the reward evaluation indicators

## Core Algorithm Features

### Multi-Path Estimation Strategy

The key innovation of this project is the **Multi-Path Estimation Strategy**, which significantly improves reinforcement learning training stability:

#### Implementation Principle

1.**Multiple Decoding**: Perform multiple (default 16) independent reconstructions for each generated sequence variant

2.**Random Shuffling**: Randomly reorder amino acid positions for each reconstruction

3.**Gradient Averaging**: Calculate policy gradients based on log probabilities from all paths

#### Algorithm Advantages

·**Variance Reduction**: Multi-path estimation significantly reduces policy gradient estimation variance

·**Enhanced Exploration**: Different decoding orders explore different generative paths in sequence space

·**Improved Credit Assignment**: More accurately attribute rewards to specific amino acid positions

### MSA Computation Optimization

Performance optimization achieved by bypassing AlphaFold 3's MSA computation process:

·**Time Savings**: Reduces single prediction from hours to minutes

·**Resource Optimization**: Reduces CPU and database access requirements

·**Accuracy Preservation**: Validation shows minimal impact on optimization effectiveness

## Contributing

We welcome and encourage contributions to this project! Whether you want to report bugs, suggest new features, or directly submit code improvements, we greatly appreciate your input.

### How to Contribute

#### Reporting Issues

If you find bugs or have improvement suggestions:

1.Create a new Issue on GitHub

2.Describe the problem or suggestion in detail

3.For bugs, please provide:

·System environment information (OS, CUDA version, etc.)

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

·Create GitHub Issues for public discussion

·Contact the maintenance team through the project repository

Thank you for contributing to protein sequence optimization research!

## References

### Core Papers

**1.EvoDiff**: Alamdari, S., et al. "Protein generation with evolutionary diffusion: sequence is all you need." *bioRxiv* (2023).

·Paper: https://www.biorxiv.org/content/10.1101/2023.09.11.556673v2

·Code: https://github.com/microsoft/evodiff

**2.AlphaFold 3**: Josh Abramson, et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature* (2024).

·Paper: https://www.nature.com/articles/s41586-024-07487-w

·Code: https://github.com/google-deepmind/alphafold3

### Algorithmic Foundation (GRPO)

**3.DeepSeek-R1**: DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." *arXiv* (2025).

·Paper: https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf

·Code: https://github.com/deepseek-ai/DeepSeek-R1

**4.DeepSeek-Math**: Zhihong Shao, et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." *arXiv* (2024).

·Paper: https://arxiv.org/abs/2402.03300

·Code: https://github.com/deepseek-ai/DeepSeek-Math

## License and Acknowledgments

### License

This project is released under the CC BY 4.0 license.

### Acknowledgments

·**Microsoft Research**: For developing and open-sourcing EvoDiff

·**Google DeepMind**: For creating AlphaFold 3 and providing model access

·**ShanghaiTech University**: For providing computational resources and support

### Authors

·**Tianyi Wang**: the core strategist of our project, who provided a large number of high-quality ideas (we have suggested that he buy a T-shirt and a coffee cup, even though he majors in biology)

·**Yafei Chang**: has something to say to some thing: "Thank you, my penguin doll. You accompanied me while debugging code late at night. I really want to include you in my list of thanks"

·**Claude** (Our AI Assistant): special thanks to Claude for its invaluable assistance in code writing and debugging, providing concrete technical support throughout the development process

---

## Disclaimer

1.This software requires access to AlphaFold 3 model parameters; please ensure proper authorization from Google DeepMind before use

2.This software is provided "as is" without any express or implied warranties

3.Users are responsible for validating results for their specific applications

4.Please comply with all applicable software licenses and terms of use

---

**Contact**: For technical questions or suggestions, please contact us through GitHub Issues or the project repository.
