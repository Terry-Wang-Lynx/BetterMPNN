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
BASE_DIR = "/data/home/scvi041/run/mpnn-620-lr"  # You must modify

# AlphaFold 3 Configuration
# You must modify these paths
AF3_MODEL_DIR = "/path/to/your/alphafold3/model"  # You must modify
AF3_DB_DIR = "/path/to/your/alphafold3/dataset"   # You also must modify
AF3_SIF_PATH = "/path/to/your/alphafold3/alphafold3.sif"  # You still must modify
SINGULARITY_PATH = "/path/to/your/apptainer(or singularity)"  # Yes, you must modify

# JSON template file path (relative to script location, no modification needed if following project structure)
JSON_TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), '../config/test.json')

# ProteinMPNN Configuration (relative paths, no modification needed if following project structure)
SCAFFOLD_PDB_PATH = os.path.join(os.path.dirname(__file__), '../input/fold_2_620.pdb')
PATH_TO_MODEL_WEIGHTS = os.path.join(os.path.dirname(__file__), '../vanilla_model_weights/v_48_020.pt')

# Output Directory Configuration (no modification needed)
ROOT_INPUT_DIR = "rl_input"       
ROOT_PREDICTION_DIR = "rl_prediction" 
CHECKPOINT_DIR = "rl_checkpoint"    
GRAPH_DIR = "rl_graph"              
REWARD_LOG_FILE = "rl_rewards_log.csv"
AF3_SUMMARY_LOG_FILE = "af3_summary_log.txt"

# GRPO Training Hyperparameters (tuning recommended)
LEARNING_RATE = 1e-5
TRAINING_STEPS = 3000
NUM_GENERATIONS = 8
BETA = 0.01
ADVANTAGE_SCALE_FACTOR = 5.0
REWARD_SHAPING_ALPHA = 0.7
SAMPLING_TEMPERATURE = 0.3
SAVE_CHECKPOINT_EVERY = 15
CLEANUP_INPUT_EVERY = 3

# Sequence Design Region Definition
CHAIN_ID_TO_DESIGN = "B"  # Modify if designing different chain

## End Configuration!! Congratulation!!


### SECTION 1: Logging Setup


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


### SECTION 2: Core Functions


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
                        
    return variants, featurized_batch, S_sample, sample_dict, randn_2


def get_per_token_log_probs(model, featurized_batch, S_sample, sample_dict, randn_2):
    """
    Get log probabilities for each sampled token
    """
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
    visible_list_list, masked_list_list, masked_chain_length_list_list, \
    chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
    tied_pos_list_of_lists_list, pssm_coef, pssm_bias, \
    pssm_log_odds_all, bias_by_res_all, tied_beta = featurized_batch
    
    mask_for_loss = mask * chain_M * chain_M_pos
    
    # Get token-level log probability distribution
    token_log_probs_dist = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, 
                             chain_encoding_all, randn_2, 
                             use_input_decoding_order=True, 
                             decoding_order=sample_dict["decoding_order"])
    
    # Collect log probabilities of sampled tokens from distribution
    per_token_logps = torch.gather(token_log_probs_dist, 2, S_sample.unsqueeze(-1)).squeeze(-1)

    return per_token_logps, mask_for_loss


def reshape_rewards(rewards, alpha=REWARD_SHAPING_ALPHA):
    """
    Apply non-linear transformation to rewards to increase differentiation
    """
    # Use exponential transformation to increase reward differentiation
    return torch.sign(rewards) * torch.pow(torch.abs(rewards), alpha)


def compute_group_relative_advantages(rewards, scale_rewards=True, scale_factor=ADVANTAGE_SCALE_FACTOR):
    """
    Calculate group relative advantage function
    """
    if len(rewards) <= 1:
        return torch.zeros_like(rewards)
    
    # First reshape rewards
    reshaped_rewards = reshape_rewards(rewards)
    mean_reward = reshaped_rewards.mean()
    
    if scale_rewards:
        std_reward = reshaped_rewards.std()
        if std_reward > 1e-8:
            advantages = (reshaped_rewards - mean_reward) / std_reward
        else:
            # Use fixed scaling factor when standard deviation is small
            advantages = (reshaped_rewards - mean_reward) * scale_factor
    else:
        advantages = reshaped_rewards - mean_reward
    
    return advantages


def compute_grpo_loss(current_per_token_logps, ref_per_token_logps, advantages, mask, beta=BETA):
    """
    Calculate loss function according to official GRPO implementation
    """
    # 1. Calculate KL divergence (using unbiased estimator)
    per_token_kl = torch.exp(ref_per_token_logps - current_per_token_logps) - (ref_per_token_logps - current_per_token_logps) - 1
    
    # 2. Calculate policy term
    policy_gradient_term = torch.exp(current_per_token_logps - current_per_token_logps.detach()) * advantages.unsqueeze(1)
    
    # 3. Combine loss terms
    per_token_loss = -(policy_gradient_term - beta * per_token_kl)
    
    # 4. Apply mask and calculate average loss
    masked_loss = per_token_loss * mask
    summed_loss_per_seq = masked_loss.sum(dim=1)
    num_valid_tokens_per_seq = mask.sum(dim=1)
    num_valid_tokens_per_seq = torch.clamp(num_valid_tokens_per_seq, min=1.0)
    loss_per_seq = summed_loss_per_seq / num_valid_tokens_per_seq
    loss = loss_per_seq.mean()

    # 5. Calculate and return average KL divergence for logging
    masked_kl = per_token_kl * mask
    mean_kl_per_seq = (masked_kl.sum(dim=1) / num_valid_tokens_per_seq)
    mean_kl = mean_kl_per_seq.mean()
    
    # 6. Calculate policy loss (for logging)
    policy_term = torch.exp(current_per_token_logps - current_per_token_logps.detach()) * advantages.unsqueeze(1)
    masked_policy_term = policy_term * mask
    mean_policy_term_per_seq = (masked_policy_term.sum(dim=1) / num_valid_tokens_per_seq)
    policy_loss = -mean_policy_term_per_seq.mean()

    return loss, policy_loss, mean_kl


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


def log_reward_details(step, variant_idx, reward_info, variant_sequence):
    """
    Log detailed reward information
    """
    with open(REWARD_LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            step + 1, variant_idx + 1, reward_info['total_reward'],
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


def log_af3_summary(output_dir, step, variant_idx, variant_sequence):
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
        f.write(f"Step: {step}, Variant: {variant_idx}, Sequence: {variant_sequence}\n")
        f.write(f"Summary file: {summary_file}\n")
        f.write(json.dumps(summary_data, indent=2))
        f.write("\n" + "="*80 + "\n")
    
    logger.info(f"AF3 summary information logged: {summary_file}")


def get_af3_reward(variant_sequence, step, variant_idx):
    """
    Get AlphaFold3 structure prediction score as reward
    """
    job_name = f"step_{step}_variant_{variant_idx}"
    input_dir = os.path.join(ROOT_INPUT_DIR, job_name)
    output_dir = os.path.join(ROOT_PREDICTION_DIR, job_name)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    input_json_path = os.path.join(input_dir, 'input.json')
    
    # Create input JSON file
    if not create_af3_input_json(variant_sequence, input_json_path):
        return torch.tensor(0.0)
    
    # Run AF3 prediction
    result = run_af3_prediction(input_json_path, output_dir, job_name)
    if not result or result.returncode != 0:
        if result:
            logger.error(f"AF3 for {job_name} failed, return code: {result.returncode}")
            logger.error(f"AF3 STDOUT: {result.stdout}")
            logger.error(f"AF3 STDERR: {result.stderr}")
        return torch.tensor(0.0)
    
    # Find and parse summary file
    summary_file = find_summary_file(output_dir)
    if not summary_file:
        logger.warning(f"AF3 for {job_name} ran successfully but no summary_confidences.json file found")
        return torch.tensor(0.0)
    
    summary_data = parse_summary_file(summary_file)
    if not summary_data:
        return torch.tensor(0.0)
    
    # Calculate comprehensive reward
    total_reward, reward_info = calculate_comprehensive_reward(summary_data, REWARD_WEIGHTS)
    
    logger.info(f"Successfully calculated comprehensive reward: {total_reward:.4f}")
    logger.info(f"Reward components: ranking_score={reward_info['ranking_score']:.4f}, "
               f"mean_pae={reward_info['mean_pae']:.4f}, "
               f"iptm={reward_info['iptm']:.4f}, "
               f"ptm={reward_info['ptm']:.4f}, "
               f"has_clash={reward_info['has_clash']}")
    
    # Log AF3 summary information
    log_af3_summary(output_dir, step, variant_idx, variant_sequence)
    
    # Log detailed reward information
    log_reward_details(step, variant_idx, reward_info, variant_sequence)
    
    return torch.tensor(total_reward)


def plot_and_save_graph(losses, rewards, kls, policy_losses, save_path):
    """Plot and save training progress charts"""
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title('Total Loss Curve')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(rewards, color='g')
    plt.title('Average Reward Curve (AF3 Score)')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(kls, color='r')
    plt.title('KL Divergence Curve')
    plt.xlabel('Step')
    plt.ylabel('KL Divergence')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(policy_losses, color='orange')
    plt.title('Policy Loss Curve')
    plt.xlabel('Step')
    plt.ylabel('Policy Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


### SECTION 3: Reinforcement Learning Main Process


def grpo_finetune_proteinmpnn():
    """
    Main function for GRPO algorithm reinforcement learning fine-tuning of ProteinMPNN model
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
    model.train()
    
    # Create reference model (frozen)
    ref_model = copy.deepcopy(model)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training history tracking
    losses_history = []
    rewards_history = []
    kls_history = []
    policy_losses_history = []

    # Create directories
    for dir_path in [CHECKPOINT_DIR, GRAPH_DIR, ROOT_INPUT_DIR, ROOT_PREDICTION_DIR]:
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

    logger.info("Starting GRPO training...")
    for step in range(TRAINING_STEPS):
        logger.info(f"\n===== Starting training step {step+1}/{TRAINING_STEPS} =====")
        
        # 1. Generate sequence group
        logger.info(f"Step {step+1}: Generating {NUM_GENERATIONS} new sequences using ProteinMPNN...")
        variants, featurized_batch, S_sample, sample_dict, randn_2 = generate_sequences_with_mpnn(
            model, SCAFFOLD_PDB_PATH, CHAIN_ID_TO_DESIGN, 
            DESIGN_POSITIONS_ALL, NUM_GENERATIONS, device=device
        )

        # 2. Calculate rewards
        logger.info(f"Step {step+1}: Starting AF3 reward calculation for {NUM_GENERATIONS} variants...")
        rewards_list = []
        for i, variant in enumerate(variants):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            reward = get_af3_reward(variant, step, i)
            rewards_list.append(reward)
        
        rewards = torch.stack(rewards_list).to(device)

        # 3. Calculate group relative advantages
        advantages = compute_group_relative_advantages(rewards)
        
        # Add detailed monitoring logs
        logger.info(f"Reward statistics: mean={rewards.mean().item():.4f}, std={rewards.std().item():.4f}, "
                   f"min={rewards.min().item():.4f}, max={rewards.max().item():.4f}")
        logger.info(f"Advantage statistics: mean={advantages.mean().item():.6f}, std={advantages.std().item():.6f}, "
                   f"min={advantages.min().item():.6f}, max={advantages.max().item():.6f}")

        # 4. Calculate log probabilities for each sampled token
        with torch.set_grad_enabled(True):
            current_per_token_logps, mask_for_loss = get_per_token_log_probs(
                model, featurized_batch, S_sample, sample_dict, randn_2
            )
        
        with torch.no_grad():
            ref_per_token_logps, _ = get_per_token_log_probs(
                ref_model, featurized_batch, S_sample, sample_dict, randn_2
            )
        
        # 5. Calculate loss
        total_loss, policy_loss, kl_div = compute_grpo_loss(
            current_per_token_logps, ref_per_token_logps, advantages, mask_for_loss,
            beta=BETA
        )

        # 6. Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Gradient monitoring
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        logger.info(f"Total gradient norm: {total_norm:.8f}")
        
        optimizer.step()
        
        # 7. Record training history
        mean_reward = rewards.mean()
        losses_history.append(total_loss.item())
        rewards_history.append(mean_reward.item())
        kls_history.append(kl_div.item())
        policy_losses_history.append(policy_loss.item())
        
        logger.info(f"===== Step {step+1} Summary =====")
        logger.info(f"Average reward: {mean_reward.item():.4f}")
        logger.info(f"Total loss: {total_loss.item():.4f}")
        logger.info(f"Policy loss: {policy_loss.item():.6f}")
        logger.info(f"KL divergence: {kl_div.item():.6f}")

        # 8. Save checkpoint
        if (step + 1) % SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"mpnn_model_step_{step+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(), 
                'num_edges': checkpoint['num_edges'],
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step + 1,
                'losses_history': losses_history,
                'rewards_history': rewards_history,
                'kls_history': kls_history,
                'policy_losses_history': policy_losses_history
            }, checkpoint_path)
            logger.info(f"Model checkpoint saved to: {checkpoint_path}")
        
        # 9. Plot training charts
        plot_and_save_graph(
            losses_history, rewards_history, kls_history, policy_losses_history,
            os.path.join(GRAPH_DIR, f"grpo_progress_step_{step+1}.png")
        )
        
        # 10. Regular cleanup of temporary files
        if (step + 1) % CLEANUP_INPUT_EVERY == 0 and step > 0:
            for dir_to_clean in [ROOT_INPUT_DIR, ROOT_PREDICTION_DIR]:
                if os.path.exists(dir_to_clean):
                    shutil.rmtree(dir_to_clean)
                    os.makedirs(dir_to_clean, exist_ok=True)
            logger.info("Temporary files cleaned up")
        
        # 11. Early stopping check
        if kl_div.item() > 4 * BETA:
            logger.warning(f"KL divergence too large ({kl_div.item():.4f}), may need to reduce learning rate")
        
        # 12. Clean up GPU memory
        del variants, featurized_batch, S_sample, sample_dict, randn_2
        del rewards, advantages, current_per_token_logps, ref_per_token_logps, total_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("\nTraining complete!")
    
    # Save final model
    final_checkpoint_path = os.path.join(CHECKPOINT_DIR, "mpnn_model_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(), 
        'num_edges': checkpoint['num_edges'],
        'optimizer_state_dict': optimizer.state_dict(),
        'final_step': TRAINING_STEPS,
        'losses_history': losses_history,
        'rewards_history': rewards_history,
        'kls_history': kls_history,
        'policy_losses_history': policy_losses_history
    }, final_checkpoint_path)
    logger.info(f"Final model saved to: {final_checkpoint_path}")


if __name__ == '__main__':
    grpo_finetune_proteinmpnn()