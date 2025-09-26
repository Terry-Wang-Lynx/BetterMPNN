import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import subprocess
import json
import time
import csv
import shutil
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
import logging

# ProteinMPNN Related Module Imports
from protein_mpnn_utils import parse_PDB, StructureDatasetPDB, _scores, _S_to_seq, tied_featurize, ProteinMPNN
from model_utils import loss_nll

# Custom Module Imports
from reward_utils import (
    exponential_decay_reward,
    find_summary_file,
    parse_summary_file,
    calculate_comprehensive_reward,
    REWARD_WEIGHTS
)

## !!Attention!! Configuration Section Here!!

# Working directory
# You must modify according to your project location
BASE_DIR = "/path/to/your/project/root"  # You must modify

# AlphaFold 3 Configuration
# You must modify these paths
AF3_MODEL_DIR = "/path/to/your/alphafold3/model"  # You must modify
AF3_DB_DIR = "/path/to/your/alphafold3/dataset"   # You also must modify
AF3_SIF_PATH = "/path/to/your/alphafold3/alphafold3.sif"  # You still must modify
SINGULARITY_PATH = "/path/to/your/apptainer(or singularity)"  # Yes, you must modify

# JSON template file path (relative to script location, no modification needed if following project structure)
JSON_TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), '../config/test.json')

# ProteinMPNN Configuration (relative paths, no modification needed if following project structure)
SCAFFOLD_PDB_PATH = os.path.join(os.path.dirname(__file__), '../input/XXX.pdb')  # Your .pdb
# Put your checkpoint-model into the 'rl_model' directory
PATH_TO_MODEL_WEIGHTS = os.path.join(os.path.dirname(__file__), '../rl_model/mpnn_model_step_XX.pt')  # Your checkpoint-model

# Output Directory Configuration
ROOT_INPUT_DIR = "rl_input"       
ROOT_PREDICTION_DIR = "test_rl_prediction" 
REWARD_LOG_FILE = "test_rl_rewards_log.csv"
AF3_SUMMARY_LOG_FILE = "test_af3_summary_log.txt"

# Generation Parameters
NUM_VARIANTS = 8  # Generate 8 sequences
SAMPLING_TEMPERATURE = 0.3

# Sequence Design Region Definition
CHAIN_ID_TO_DESIGN = "B"  # Modify if designing different chain

## End Configuration!! Congratulation!!


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

### SECTION 1: Core Functions

def generate_sequences_with_mpnn(model, pdb_path, chain_to_design, positions_to_design, num_variants, temperature=SAMPLING_TEMPERATURE, device='cuda'):
    """
    Generate new sequence variants using ProteinMPNN based on PDB scaffold
    """
    pdb_dict_list = parse_PDB(pdb_path, ca_only=False)
    total_residues = len(pdb_dict_list[0][f'seq_chain_{chain_to_design}'])
    all_positions = set(range(total_residues))
    fixed_positions = sorted(list(all_positions - set(positions_to_design)))
    
    fixed_positions_dict = {
        pdb_dict_list[0]['name']: {
            chain_to_design: fixed_positions
        }
    } if fixed_positions else None

    chain_id_dict = {
        pdb_dict_list[0]['name']: ([chain_to_design], [])
    }
    
    dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=200000)
    batch_clones = [copy.deepcopy(dataset[0]) for _ in range(num_variants)]

    featurized_batch = tied_featurize(
        batch_clones, device, chain_id_dict, 
        fixed_positions_dict, None, None, None, None
    )
    
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
    visible_list_list, masked_list_list, masked_chain_length_list_list, \
    chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
    tied_pos_list_of_lists_list, pssm_coef, pssm_bias, \
    pssm_log_odds_all, bias_by_res_all, tied_beta = featurized_batch
    
    with torch.no_grad():
        randn_2 = torch.randn(chain_M.shape, device=X.device)
        sample_dict = model.sample(
            X, randn_2, S, chain_M, chain_encoding_all, residue_idx, 
            mask=mask, temperature=temperature, chain_M_pos=chain_M_pos,
            omit_AAs_np=np.zeros(21), bias_AAs_np=np.zeros(21),
            omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
            pssm_bias=pssm_bias, bias_by_res=bias_by_res_all
        )
    
    S_sample = sample_dict["S"]
    variants = [_S_to_seq(S_sample[i], chain_M[i]) for i in range(num_variants)]
                        
    return variants


def create_af3_input_json(variant_sequence, input_json_path):
    """
    Create AF3 input JSON file
    """
    if not os.path.exists(JSON_TEMPLATE_FILE):
        logger.error(f"JSON template file '{JSON_TEMPLATE_FILE}' not found!")
        return False

    with open(JSON_TEMPLATE_FILE, 'r') as f:
        af3_input_data = json.load(f)
        
    af3_input_data['sequences'][1]['protein']['sequence'] = variant_sequence
    
    msa_lines = af3_input_data['sequences'][1]['protein']['unpairedMsa'].splitlines()
    msa_lines[1] = variant_sequence
    af3_input_data['sequences'][1]['protein']['unpairedMsa'] = "\n".join(msa_lines)

    with open(input_json_path, 'w') as f:
        json.dump(af3_input_data, f, indent=2)
    
    return os.path.exists(input_json_path)


def run_af3_prediction(input_json_path, output_dir, job_name):
    """
    Run AlphaFold3 prediction
    """
    temp_runner_script_path = os.path.join(os.path.dirname(input_json_path), '_run_af3.sh')
    
    # Dynamically generate temporary bash script for executing AF3
    script_content = f"""
#!/bin/bash
SINGULARITY_PATH="{SINGULARITY_PATH}"
mkdir -p {output_dir}

# Run AlphaFold3
{SINGULARITY_PATH} exec \\
  --nv \\
  -B {BASE_DIR}:/input,{AF3_MODEL_DIR}:/model,{AF3_DB_DIR}:/dataset,{os.path.abspath(output_dir)}:/output \\
  {AF3_SIF_PATH} python /input/run_alphafold.py \\
  --json_path=/input/{ROOT_INPUT_DIR}/{job_name}/input.json \\
  --model_dir=/model \\
  --db_dir=/dataset \\
  --output_dir=/output \\
  --run_data_pipeline=false
"""

    with open(temp_runner_script_path, 'w') as f:
        f.write(script_content)
    os.chmod(temp_runner_script_path, 0o755)
    
    # Clean GPU memory before running AF3
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        logger.info(f"Starting AF3 for {job_name}...")
        result = subprocess.run(['bash', temp_runner_script_path], 
                              check=False, capture_output=True, text=True, timeout=1800)
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"AF3 for {job_name} timed out")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during AF3 execution for {job_name}: {e}")
        return None
    finally:
        # Clean up temporary script
        if os.path.exists(temp_runner_script_path):
            os.remove(temp_runner_script_path)


def log_reward_details(variant_idx, reward_info, variant_sequence):
    """
    Log detailed reward information
    """
    with open(REWARD_LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            "eval", variant_idx + 1, reward_info['total_reward'],
            reward_info['mean_pae'],
            reward_info['pae_reward'],
            reward_info['iptm'],
            reward_info['iptm_reward'],
            reward_info['ptm_designed'],
            reward_info['ptm_reward'],
            reward_info['has_clash'],
            variant_sequence, 
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])


def log_af3_summary(output_dir, variant_idx, variant_sequence):
    """
    Log AF3 summary information to log file
    """
    summary_file = find_summary_file(output_dir)
    if not summary_file:
        logger.warning(f"summary_confidences.json file not found in {output_dir}")
        return
    
    summary_data = parse_summary_file(summary_file)
    if not summary_data:
        return
    
    # Log to AF3 summary log
    with open(AF3_SUMMARY_LOG_FILE, 'a') as f:
        f.write(f"Evaluation, Variant: {variant_idx}, Sequence: {variant_sequence}\n")
        f.write(f"Summary file: {summary_file}\n")
        f.write(json.dumps(summary_data, indent=2))
        f.write("\n" + "="*80 + "\n")
    
    logger.info(f"AF3 summary information logged: {summary_file}")


def get_af3_reward(variant_sequence, variant_idx):
    """
    Get AlphaFold3 structure prediction score as reward
    """
    job_name = f"eval_variant_{variant_idx}"
    input_dir = os.path.join(ROOT_INPUT_DIR, job_name)
    output_dir = os.path.join(ROOT_PREDICTION_DIR, job_name)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    input_json_path = os.path.join(input_dir, 'input.json')
    
    # Create input JSON file
    if not create_af3_input_json(variant_sequence, input_json_path):
        return 0.0, {}
    
    # Run AF3 prediction
    result = run_af3_prediction(input_json_path, output_dir, job_name)
    if not result or result.returncode != 0:
        if result:
            logger.error(f"AF3 for {job_name} failed, return code: {result.returncode}")
            logger.error(f"AF3 STDOUT: {result.stdout}")
            logger.error(f"AF3 STDERR: {result.stderr}")
        return 0.0, {}
    
    # Find and parse summary file
    summary_file = find_summary_file(output_dir)
    if not summary_file:
        logger.warning(f"AF3 for {job_name} ran successfully but no summary_confidences.json file found")
        return 0.0, {}
    
    summary_data = parse_summary_file(summary_file)
    if not summary_data:
        return 0.0, {}
    
    # Calculate comprehensive reward
    total_reward, reward_info = calculate_comprehensive_reward(summary_data, REWARD_WEIGHTS)
    
    logger.info(f"Successfully calculated comprehensive reward: {total_reward:.4f}")
    logger.info(f"Reward components: ranking_score={reward_info['ranking_score']:.4f}, "
               f"mean_pae={reward_info['mean_pae']:.4f}, "
               f"iptm={reward_info['iptm']:.4f}, "
               f"ptm={reward_info['ptm']:.4f}, "
               f"has_clash={reward_info['has_clash']}")
    
    # Log AF3 summary information
    log_af3_summary(output_dir, variant_idx, variant_sequence)
    
    # Log detailed reward information
    log_reward_details(variant_idx, reward_info, variant_sequence)
    
    return total_reward, reward_info


### SECTION 2: Main Function

def generate_and_evaluate_sequences():
    """
    Generate sequences using the trained ProteinMPNN model and evaluate with AlphaFold3
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Check necessary files
    if not os.path.exists(SCAFFOLD_PDB_PATH):
        raise FileNotFoundError(f"Error: Input PDB scaffold file not found at {SCAFFOLD_PDB_PATH}")
    
    # Parse PDB file
    pdb_dict_list = parse_PDB(SCAFFOLD_PDB_PATH, ca_only=False)
    try:
        total_residues = len(pdb_dict_list[0][f'seq_chain_{CHAIN_ID_TO_DESIGN}'])
        DESIGN_POSITIONS_ALL = list(range(total_residues))
        logger.info(f"Target protein: {pdb_dict_list[0]['name']}")
        logger.info(f"Target chain: '{CHAIN_ID_TO_DESIGN}', length: {total_residues} residues.")
        logger.info(f"Will perform global sequence design on all {len(DESIGN_POSITIONS_ALL)} positions.")
    except KeyError:
        raise KeyError(f"Error: Chain '{CHAIN_ID_TO_DESIGN}' not found in PDB file {SCAFFOLD_PDB_PATH}.")

    # Load model
    logger.info("Loading pre-trained ProteinMPNN model...")
    checkpoint = torch.load(PATH_TO_MODEL_WEIGHTS, map_location=device)
    
    hidden_dim = 128
    num_layers = 3
    
    model = ProteinMPNN(
        ca_only=False, num_letters=21, node_features=hidden_dim, 
        edge_features=hidden_dim, hidden_dim=hidden_dim, 
        num_encoder_layers=num_layers, num_decoder_layers=num_layers, 
        augment_eps=0.00, k_neighbors=checkpoint['num_edges']
    )
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    # Create directories
    for dir_path in [ROOT_INPUT_DIR, ROOT_PREDICTION_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize reward log
    if not os.path.exists(REWARD_LOG_FILE):
        with open(REWARD_LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                "step", "variant_index", "total_reward", 
                "mean_pae", "pae_reward", "iptm", "iptm_reward",
                "ptm_designed", "ptm_reward", "has_clash", "sequence", "timestamp"
            ])

    # Initialize AF3 summary log
    if not os.path.exists(AF3_SUMMARY_LOG_FILE):
        with open(AF3_SUMMARY_LOG_FILE, 'w') as f:
            f.write("AF3 Summary Confidence Log\n")
            f.write("=" * 80 + "\n")

    # Generate sequences
    logger.info(f"Generating {NUM_VARIANTS} new sequences using ProteinMPNN...")
    variants = generate_sequences_with_mpnn(
        model, SCAFFOLD_PDB_PATH, CHAIN_ID_TO_DESIGN, 
        DESIGN_POSITIONS_ALL, NUM_VARIANTS, device=device
    )
    
    # Output generated sequences
    logger.info("\nGenerated sequences:")
    for i, variant in enumerate(variants):
        logger.info(f"Sequence {i+1}: {variant}")
    
    # Evaluate each sequence with AlphaFold3
    logger.info("\nStarting AlphaFold3 evaluation for generated sequences...")
    results = []
    
    for i, variant in enumerate(variants):
        logger.info(f"Evaluating sequence {i+1}/{len(variants)}...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        reward, reward_info = get_af3_reward(variant, i)
        results.append({
            "sequence": variant,
            "reward": reward,
            "reward_info": reward_info
        })
        
        logger.info(f"Sequence {i+1} evaluation completed. Reward: {reward:.4f}")
    
    # Print summary of results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    
    for i, result in enumerate(results):
        logger.info(f"Sequence {i+1}:")
        logger.info(f"  Sequence: {result['sequence']}")
        logger.info(f"  Total Reward: {result['reward']:.4f}")
        if result['reward_info']:
            logger.info(f"  Mean PAE: {result['reward_info']['mean_pae']:.4f}")
            logger.info(f"  PAE Reward: {result['reward_info']['pae_reward']:.4f}")
            logger.info(f"  ipTM: {result['reward_info']['iptm']:.4f}")
            logger.info(f"  ipTM Reward: {result['reward_info']['iptm_reward']:.4f}")
            logger.info(f"  pTM (designed): {result['reward_info']['ptm_designed']:.4f}")
            logger.info(f"  pTM Reward: {result['reward_info']['ptm_reward']:.4f}")
            logger.info(f"  Has Clash: {result['reward_info']['has_clash']}")
        logger.info("-" * 40)
    
    # Calculate and display statistics
    rewards = [result['reward'] for result in results]
    if rewards:
        logger.info(f"Average Reward: {np.mean(rewards):.4f}")
        logger.info(f"Max Reward: {np.max(rewards):.4f}")
        logger.info(f"Min Reward: {np.min(rewards):.4f}")
    
    return results

if __name__ == '__main__':

    generate_and_evaluate_sequences()
