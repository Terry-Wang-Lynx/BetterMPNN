import torch
import torch.optim as optim
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

from evodiff.pretrained import OA_DM_38M
from evodiff.utils import Tokenizer

## !!Attention!! Configuration Section Here!!

# Update these paths according to your actual environment setup
AF3_MODEL_DIR = "/path/to/your/alphafold3/model"  # You must modify
AF3_DB_DIR = "/path/to/your/alphafold3/dataset"   # You also must modify
PDB_DATABASE_PATH = "/path/to/your/pdb/dataset"   # You still must modify

# Script and config file paths (they are relative paths, so no modification needed)
AF3_RUN_SCRIPT = os.path.join(os.path.dirname(__file__), 'run_alphafold.py')
JSON_TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), '../config/test.json')

# Output directories
ROOT_INPUT_DIR = "input"
ROOT_PREDICTION_DIR = "prediction"

# Training process files
CHECKPOINT_DIR = "checkpoints"
GRAPH_DIR = "training_graphs"
REWARD_LOG_FILE = "rewards_log.csv"

# Training parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-5
TRAINING_STEPS = 3000
NUM_GENERATIONS = 8
NUM_PATHS_PER_VARIANT = 16
BETA = 0.1
SAVE_CHECKPOINT_EVERY = 25
CLEANUP_INPUT_EVERY = 3

# Base sequence (protein sequence to be optimized)
# If you haven't modified this place and test.json,
# you are still optimizing our protein! thx!
BASE_SEQUENCE = "SKRIEEQKKNIEKSKKATEELIKNKEELTEEELEGVKEYSKEVEKAEKELEKEK"
NUM_MUTATIONS = 10  # Number of positions to mutate each iteration

## End Configuration!! Congratulation!!


### SECTION 1: Core Functions


def get_log_probs(model, sequence, decode_order, tokenizer, device=DEVICE):
    """Calculate log probabilities of a given sequence under the model"""
    all_aas = tokenizer.all_aas
    target_tokens = torch.tensor(tokenizer.tokenizeMSA(sequence), dtype=torch.long, device=device)
    sequence_list = list(sequence)
    for pos in decode_order:
        sequence_list[pos] = '#'
    masked_sequence = ''.join(sequence_list)
    input_tokens = torch.tensor(tokenizer.tokenizeMSA(masked_sequence), dtype=torch.long, device=device)
    log_p_list = []
    for i in decode_order:
        timestep = torch.tensor([0], device=device)
        input_for_model = input_tokens.clone()
        prediction = model(input_for_model.unsqueeze(0), timestep)
        logits = prediction[:, i, :len(all_aas) - 6]
        log_p_distribution = torch.nn.functional.log_softmax(logits, dim=1)
        true_token_id = target_tokens[i]
        token_log_p = log_p_distribution[:, true_token_id]
        log_p_list.append(token_log_p)
        input_tokens[i] = true_token_id
    return torch.cat(log_p_list)

def generate_variant(model, sequence: str, mask_positions: list[int], decode_order: list[int], tokenizer, device: str = DEVICE):
    """Generate protein sequence variants using the model"""
    all_aas = tokenizer.all_aas
    sequence_list = list(sequence)
    for pos in mask_positions:
        sequence_list[pos] = '#'
    masked_sequence = ''.join(sequence_list)
    sample = torch.tensor(tokenizer.tokenizeMSA(masked_sequence), dtype=torch.long, device=device)
    for i in decode_order:
        timestep = torch.tensor([0], device=device)
        with torch.no_grad():
            prediction = model(sample.unsqueeze(0), timestep)
        logits = prediction[:, i, :len(all_aas) - 6]
        p = torch.nn.functional.softmax(logits, dim=1)
        new_token_id = torch.multinomial(p, num_samples=1)
        sample[i] = new_token_id.squeeze()
    return tokenizer.untokenize(sample)

def get_value_as_tensor(file_path: str) -> torch.Tensor:
    """Extract reward value from CSV file"""
    if not os.path.exists(file_path):
        print(f"  [Warning] Reward file not found: {file_path}. Returning score of 0.")
        return torch.tensor(0.0)
    try:
        df = pd.read_csv(file_path)
        value = df.iloc[0, 2]
        return torch.tensor(float(value))
    except Exception as e:
        print(f"  [Warning] Error reading {file_path}: {e}. Returning score of 0.")
        return torch.tensor(0.0)

def get_af3_ranking_score(variant_sequence: str, step: int, variant_idx: int) -> torch.Tensor:
    """Call AlphaFold3 to compute protein structure ranking score"""
    job_name = f"step_{step}_variant_{variant_idx}"
    input_dir = os.path.join(ROOT_INPUT_DIR, job_name)
    output_dir = os.path.join(ROOT_PREDICTION_DIR, job_name)
    os.makedirs(input_dir, exist_ok=True)

    input_json_path = os.path.join(input_dir, 'input.json')
    temp_runner_script_path = os.path.join(input_dir, '_run_af3.sh')

    if not os.path.exists(JSON_TEMPLATE_FILE):
        print(f"  [Error] JSON template file '{JSON_TEMPLATE_FILE}' not found! Cannot proceed.")
        return torch.tensor(0.0)

    with open(JSON_TEMPLATE_FILE, 'r') as f:
        af3_input_data = json.load(f)

    # These are the codes that help you automatically replace the generated sequences
    # into the corresponding positions of the input json of alphafold3.
    # Generally, please do not modify them. It's best to thank them;)    
    af3_input_data['sequences'][1]['protein']['sequence'] = variant_sequence
    
    msa_lines = af3_input_data['sequences'][1]['protein']['unpairedMsa'].splitlines()
    msa_lines[1] = variant_sequence
    af3_input_data['sequences'][1]['protein']['unpairedMsa'] = "\n".join(msa_lines)

    with open(input_json_path, 'w') as f:
        json.dump(af3_input_data, f, indent=2)


    # This is the file you need to call alphafold3.
    # The following code has created it for you.
    script_content = f"""#!/bin/bash
module purge
module load alphafold/3.0.0
python {os.path.abspath(AF3_RUN_SCRIPT)} \\
     --json_path="{os.path.abspath(input_json_path)}" \\
     --output_dir="{os.path.abspath(output_dir)}" \\
     --model_dir="{AF3_MODEL_DIR}" \\
     --db_dir="{AF3_DB_DIR}" \\
     --pdb_database_path="{PDB_DATABASE_PATH}" \\
     --run_data_pipeline=false
"""
    with open(temp_runner_script_path, 'w') as f:
        f.write(script_content)
    os.chmod(temp_runner_script_path, 0o755)

    try:
        print(f"    --> Starting AF3 subprocess for {job_name}. This will take a while...")
        subprocess.run(['bash', temp_runner_script_path], check=True, capture_output=True, text=True)
        print(f"    --> AF3 subprocess for {job_name} completed.")
        
        result_csv_path = None
        for root, _, files in os.walk(output_dir):
            if 'ranking_scores.csv' in files:
                result_csv_path = os.path.join(root, 'ranking_scores.csv')
                print(f"  [Info] Successfully found reward file: {result_csv_path}")
                break
        
        if result_csv_path:
            reward = get_value_as_tensor(result_csv_path)
        else:
            print(f"  [Error] Cannot find 'ranking_scores.csv' in directory {output_dir}. AF3 may not have generated results. Returning score of 0.")
            reward = torch.tensor(0.0)

    except subprocess.CalledProcessError as e:
        print(f"  [Error] AF3 for {job_name} failed! Assigning reward of 0 for this variant.")
        print(f"  Stderr: {e.stderr}")
        reward = torch.tensor(0.0)
    finally:
        if os.path.exists(temp_runner_script_path):
            os.remove(temp_runner_script_path)
            
    return reward

def plot_and_save_graph(losses, rewards, kls, save_path):
    """Plot and save training progress charts based on historical data"""
    plt.figure(figsize=(10, 15))
    
    plt.subplot(3, 1, 1)
    plt.plot(losses)
    plt.title('Total Loss Curve')
    plt.xlabel('Training Steps'); plt.ylabel('Loss Value')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(rewards, color='g')
    plt.title('Average Reward Curve (AF3 Score)')
    plt.xlabel('Training Steps'); plt.ylabel('AF3 Ranking Score')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(kls, color='r')
    plt.title('KL Divergence Curve')
    plt.xlabel('Training Steps'); plt.ylabel('KL Divergence')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

### SECTION 2: Main training loop

def finetune_evodiff():
    """RL fine-tuning of EvoDiff using AF3 scores as rewards"""
    print(f"Using device: {DEVICE}")

    print("Loading pretrained model...")
    model, _, tokenizer, _ = OA_DM_38M()
    model.to(DEVICE)
    model.train()
    ref_model = copy.deepcopy(model)  # Keep reference for KL penalty
    ref_model.to(DEVICE)
    ref_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    seq_len = len(BASE_SEQUENCE)
    losses_history, rewards_history, kls_history = [], [], []

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)

    # Setup reward logging
    if not os.path.exists(REWARD_LOG_FILE):
        with open(REWARD_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "variant_index", "reward_score", "timestamp"])

    # Main training loop
    for step in range(TRAINING_STEPS):
        print(f"\n===== Step {step+1}/{TRAINING_STEPS} =====")
        positions_to_mutate = sorted(random.sample(range(seq_len), NUM_MUTATIONS))
        
        # Generate variants
        generation_order = positions_to_mutate.copy()
        random.shuffle(generation_order)
        variants = [
            generate_variant(model, BASE_SEQUENCE, positions_to_mutate, generation_order, tokenizer, DEVICE)
            for _ in range(NUM_GENERATIONS)
        ]

        # Get AF3 rewards for each variant
        print(f"Computing AF3 rewards for {NUM_GENERATIONS} variants...")
        rewards_list = []
        for i, variant in enumerate(variants):
            print(f"  Processing variant {i+1}/{NUM_GENERATIONS}...")
            start_time = time.time()
            reward = get_af3_ranking_score(variant, step, i)
            end_time = time.time()
            rewards_list.append(reward)

            # Log this reward
            with open(REWARD_LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([step + 1, i + 1, reward.item(), timestamp])
            
            print(f"  Variant {i+1}: reward={reward.item():.4f}, time={end_time - start_time:.1f}s")
        
        rewards = torch.tensor(rewards_list, device=DEVICE)

        # Compute advantages for policy gradient
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        # Policy gradient update with multi-path estimation
        batch_loss = 0
        batch_kl_div = 0
        for i in range(NUM_GENERATIONS):
            for _ in range(NUM_PATHS_PER_VARIANT):  # Multi-path trick
                training_order = positions_to_mutate.copy()
                random.shuffle(training_order)
                
                # Get current and reference log probs
                with torch.set_grad_enabled(True):
                    log_probs = get_log_probs(model, variants[i], training_order, tokenizer, DEVICE)
                with torch.no_grad():
                    ref_log_probs = get_log_probs(ref_model, variants[i], training_order, tokenizer, DEVICE)
                
                # Policy loss + KL penalty
                policy_loss = -(log_probs * advantages[i]).sum()
                kl_div = (torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1).sum()
                batch_loss += policy_loss + BETA * kl_div
                batch_kl_div += kl_div.item()

        total_loss = batch_loss / (NUM_GENERATIONS * NUM_PATHS_PER_VARIANT)
        avg_kl_div = batch_kl_div / (NUM_GENERATIONS * NUM_PATHS_PER_VARIANT)

        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        losses_history.append(total_loss.item())
        rewards_history.append(mean_reward.item())
        kls_history.append(avg_kl_div)

        print(f"Step {step+1}: reward={mean_reward.item():.4f}, loss={total_loss.item():.4f}, kl={avg_kl_div:.4f}")

        # Save checkpoint periodically
        if (step + 1) % SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_step_{step+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save training plot every step
        graph_path = os.path.join(GRAPH_DIR, f"progress_step_{step+1}.png")
        plot_and_save_graph(losses_history, rewards_history, kls_history, graph_path)
        
        # Cleanup old input files to save disk space
        if (step + 1) % CLEANUP_INPUT_EVERY == 0 and step > 0:
            print(f"  Cleaning up old input files...")
            for item_name in os.listdir(ROOT_INPUT_DIR):
                item_path = os.path.join(ROOT_INPUT_DIR, item_name)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"  Warning: couldn't clean {item_path}: {e}")

    print("\nTraining complete!")
    
    # Save final model
    final_model_path = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    print(f"Training plots saved in: {GRAPH_DIR}")


if __name__ == '__main__':
    finetune_evodiff()