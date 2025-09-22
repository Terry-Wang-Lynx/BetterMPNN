# Protein Sequence Optimization Tool Based on EvoDiff and AlphaFold 3

## Project Overview

This project develops an innovative protein sequence optimization framework that combines **EvoDiff** and **AlphaFold 3**. The tool employs reinforcement learning strategies, using AlphaFold 3's ranking scores as reward signals to fine-tune the EvoDiff model, aiming to generate protein sequence variants with higher predicted structural stability.

### Core Features

·**Sequence Space Optimization**: Direct optimization in protein sequence space without requiring structural information

·**Multi-Path Estimation Strategy**: Uses multi-path decoding to reduce gradient estimation variance and improve training stability

·**MSA Computation Bypass**: Optional performance optimization that significantly reduces computation time

·**Reinforcement Learning Fine-tuning**: Employs policy gradient methods with KL divergence regularization

·**End-to-End Workflow**: Complete solution from sequence generation to structure evaluation

### Workflow

1.**Sequence Generation**: Use EvoDiff to generate diverse protein sequence variants

2.**Structure Prediction**: Evaluate each variant's predicted structure quality through AlphaFold 3

3.**Reward Calculation**: Extract AlphaFold 3's ranking scores as optimization targets

4.**Model Fine-tuning**: Iteratively optimize EvoDiff using reinforcement learning policy gradient methods

5.**Multi-Path Training**: Generate multiple decoding paths for each variant to stabilize gradient estimation

## Installation Guide

### Step 1: Environment Setup

#### Create Conda Environment

```bash
# Create EvoDiff environment
conda env create -f environment.yml
conda activate evodiff

# Or create manually
conda create -n evodiff python=3.10
conda activate evodiff
```

#### Install Base Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
conda install pytorch=2.0.1 torchvision=0.15.2 cudatoolkit=11.8 -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

### Step 2: EvoDiff Installation

```bash
# Clone EvoDiff repository
git clone https://github.com/microsoft/evodiff.git
cd evodiff

# Install from source
pip install -e .

# Verify installation
python -c "from evodiff.pretrained import OA_DM_38M; print('EvoDiff installation successful')"
```

### Step 3: AlphaFold 3 Setup

#### Obtain AlphaFold 3 Code

```bash
git clone https://github.com/google-deepmind/alphafold3.git
cd alphafold3
```

#### Obtain Model Parameters

> **Important**: AlphaFold 3 model parameters must be obtained directly from Google DeepMind
>
> 1. Visit: https://github.com/google-deepmind/alphafold3
> 2. Follow the official process to request model parameter access
> 3. Download model parameters to the specified directory

#### Download Genetic Databases

```bash
# Use official script to download databases (~45 minutes, 627GB)
./fetch_databases.sh [<database_directory>]

# Or specify custom directory
./fetch_databases.sh /path/to/your/databases
```

#### Environment Configuration

Configure appropriate modules based on your cluster environment:

```bash
# Example: Load necessary modules
module load alphafold/3.0.0
module load cuda/12.8
module load cudnn/9.6.0.74_cuda12
```

### Step 4: Project Configuration

#### Path Configuration

Edit `scripts/test-1.py` and update the following paths:

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

Edit the base sequence in `scripts/test-1.py`:

```python
# Replace with your target sequence to optimize
BASE_SEQUENCE = "YOUR_PROTEIN_SEQUENCE_HERE"
```

### Advanced Configuration

#### Optimization Parameter Tuning

Modify training parameters in `scripts/test-1.py`:

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

This project supports two MSA processing methods:

##### Option 1: Skip MSA Computation (Recommended for Performance)

If you have pre-computed MSA data, you can configure to skip MSA computation:

**1.Prepare MSA Data**

·Ensure MSA uses A3M format

·First sequence must be the original sequence

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

In the AlphaFold 3 call section of `scripts/test.py`, change:

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
│   ├── model_step_25.pth     # Periodically saved models
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

#### Project Structure

·`scripts/`: Core algorithm implementations

·`config/`: Configuration file templates

·`checkpoints/`: Model checkpoint storage

·`training_graphs/`: Training visualization outputs

#### Testing Environment

We recommend testing your changes in the following environments:

·**Minimal Configuration**: Use smaller protein sequences for quick testing

·**Parameter Adjustment**: Reduce TRAINING_STEPS to shorten test time

·**Error Handling**: Ensure program exits gracefully under exceptional conditions

#### Priority Improvement Areas

We particularly welcome contributions in the following areas:

1.**Performance Optimization**: Improve algorithm efficiency or memory usage

2.**Error Handling**: Enhance program robustness

3.**Documentation Improvement**: Complete usage instructions or API documentation

4.**Feature Extension**: Support more protein types or structures

5.**Visualization Enhancement**: Improve training process visualization

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

**2.AlphaFold 3**: Abramson, J., et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." *Nature* (2024).

·Paper: https://www.nature.com/articles/s41586-024-07487-w

·Code: https://github.com/google-deepmind/alphafold3

### Algorithmic Foundation (GRPO)

**3.DeepSeek-R1**: DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv preprint (2024).

·Paper: https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf

·Code: https://github.com/deepseek-ai/DeepSeek-R1

**4.DeepSeek-Math**: Xie, T., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." *arXiv* (2024).

·Paper: https://arxiv.org/abs/2402.03300

·Code: https://github.com/deepseek-ai/DeepSeek-Math

## License and Acknowledgments

### License

This project is released under the [CC BY 4.0](LICENSE.txt) license.

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
