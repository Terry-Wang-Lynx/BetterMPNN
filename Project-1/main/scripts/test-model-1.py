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
import logging

## !!Attention!! Configuration Section Here!!

# Update these paths according to your actual environment setup
AF3_MODEL_DIR = "/path/to/your/alphafold3/model"  # You must modify
AF3_DB_DIR = "/path/to/your/alphafold3/dataset"   # You also must modify
PDB_DATABASE_PATH = "/path/to/your/pdb/dataset"   # You still must modify

# Script and config file paths (they are relative paths, so no modification needed)
AF3_RUN_SCRIPT = os.path.join(os.path.dirname(__file__), 'run_alphafold.py')
JSON_TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), '../config/test.json')

# Output directories
ROOT_INPUT_DIR = "test_input"
ROOT_PREDICTION_DIR = "test_prediction"

# Evaluation process files
EVAL_LOG_FILE = "evaluation_results.csv"
AF3_SUMMARY_LOG_FILE = "test_af3_summary_log.txt"

# Evaluation parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GENERATIONS = 8
NUM_MUTATIONS = 10  # Number of positions to mutate each iteration

# Base sequence (protein sequence to be optimized)
# If you haven't modified this place and test.json,
# you are still optimizing our protein! thx!
BASE_SEQUENCE = "SKRIEEQKKNIEKSKKATEELIKNKEELTEEELEGVKEYSKEVEKAEKELEKEK"

# Model checkpoint path (modify to point to your trained model)
MODEL_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), '../checkpoints/model_final.pth')

## End Configuration!! Congratulation!!


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

### SECTION 1: Core Functions

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
        logger.warning(f"Reward file not found: {file_path}. Returning score of 0.")
        return torch.tensor(0.0)
    try:
        df = pd.read_csv(file_path)
        value = df.iloc[0, 2]
        return torch.tensor(float(value))
    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}. Returning score of 0.")
        return torch.tensor(0.0)

def find_summary_file(output_dir):
    """Find AF3 summary file in output directory"""
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file == 'summary_confidences.json':
                return os.path.join(root, file)
    return None

def parse_summary_file(summary_file_path):
    """Parse AF3 summary file to extract confidence metrics"""
    try:
        with open(summary_file_path, 'r') as f:
            summary_data = json.load(f)
        return summary_data
    except Exception as e:
        logger.error(f"Error parsing summary file {summary_file_path}: {e}")
        return None

def log_af3_summary(output_dir, variant_idx, variant_sequence):
    """Log AF3 summary information to log file"""
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

def get_af3_ranking_score(variant_sequence: str, variant_idx: int) -> torch.Tensor:
    """Call AlphaFold3 to compute protein structure ranking score"""
    job_name = f"eval_variant_{variant_idx}"
    input_dir = os.path.join(ROOT_INPUT_DIR, job_name)
    output_dir = os.path.join(ROOT_PREDICTION_DIR, job_name)
    os.makedirs(input_dir, exist_ok=True)

    input_json_path = os.path.join(input_dir, 'input.json')
    temp_runner_script_path = os.path.join(input_dir, '_run_af3.sh')

    if not os.path.exists(JSON_TEMPLATE_FILE):
        logger.error(f"JSON template file '{JSON_TEMPLATE_FILE}' not found! Cannot proceed.")
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
        logger.info(f"Starting AF3 subprocess for {job_name}. This will take a while...")
        subprocess.run(['bash', temp_runner_script_path], check=True, capture_output=True, text=True)
        logger.info(f"AF3 subprocess for {job_name} completed.")
        
        result_csv_path = None
        for root, _, files in os.walk(output_dir):
            if 'ranking_scores.csv' in files:
                result_csv_path = os.path.join(root, 'ranking_scores.csv')
                logger.info(f"Successfully found reward file: {result_csv_path}")
                break
        
        if result_csv_path:
            reward = get_value_as_tensor(result_csv_path)
        else:
            logger.error(f"Cannot find 'ranking_scores.csv' in directory {output_dir}. AF3 may not have generated results. Returning score of 0.")
            reward = torch.tensor(0.0)

    except subprocess.CalledProcessError as e:
        logger.error(f"AF3 for {job_name} failed! Assigning reward of 0 for this variant.")
        logger.error(f"Stderr: {e.stderr}")
        reward = torch.tensor(0.0)
    finally:
        if os.path.exists(temp_runner_script_path):
            os.remove(temp_runner_script_path)
        
        # Record AF3 summary information
        log_af3_summary(output_dir, variant_idx, variant_sequence)
            
    return reward

### SECTION 2: Model Evaluation Function

def evaluate_trained_model():
    """Evaluate the performance of a trained model by generating sequences and assessing with AlphaFold3"""
    logger.info(f"Using device: {DEVICE}")

    # Load the trained model
    logger.info("Loading trained model...")
    model, _, tokenizer, _ = OA_DM_38M()
    model.to(DEVICE)
    
    # Load model checkpoint
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        logger.error(f"Model checkpoint not found at {MODEL_CHECKPOINT_PATH}")
        return
    
    model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()  # Set to evaluation mode
    
    # Create output directories
    os.makedirs(ROOT_INPUT_DIR, exist_ok=True)
    os.makedirs(ROOT_PREDICTION_DIR, exist_ok=True)
    
    # Initialize evaluation log
    if not os.path.exists(EVAL_LOG_FILE):
        with open(EVAL_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["variant_index", "reward_score", "sequence", "timestamp"])
    
    # Initialize AF3 summary log
    if not os.path.exists(AF3_SUMMARY_LOG_FILE):
        with open(AF3_SUMMARY_LOG_FILE, 'w') as f:
            f.write("AF3 Evaluation Summary Log\n")
            f.write("=" * 80 + "\n")

    seq_len = len(BASE_SEQUENCE)
    positions_to_mutate = sorted(random.sample(range(seq_len), NUM_MUTATIONS))
    
    # Generate variants
    logger.info(f"Generating {NUM_GENERATIONS} sequence variants...")
    generation_order = positions_to_mutate.copy()
    random.shuffle(generation_order)
    
    variants = [
        generate_variant(model, BASE_SEQUENCE, positions_to_mutate, generation_order, tokenizer, DEVICE)
        for _ in range(NUM_GENERATIONS)
    ]
    
    # Display generated sequences
    logger.info("\nGenerated sequences:")
    for i, variant in enumerate(variants):
        logger.info(f"Variant {i+1}: {variant}")
    
    # Evaluate each variant with AlphaFold3
    logger.info(f"\nEvaluating {NUM_GENERATIONS} variants with AlphaFold3...")
    rewards_list = []
    
    for i, variant in enumerate(variants):
        logger.info(f"Evaluating variant {i+1}/{NUM_GENERATIONS}...")
        start_time = time.time()
        
        reward = get_af3_ranking_score(variant, i)
        end_time = time.time()
        
        rewards_list.append(reward.item())
        
        # Log this evaluation result
        with open(EVAL_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([i + 1, reward.item(), variant, timestamp])
        
        logger.info(f"Variant {i+1}: reward={reward.item():.4f}, time={end_time - start_time:.1f}s")
    
    # Calculate and display evaluation statistics
    if rewards_list:
        avg_reward = np.mean(rewards_list)
        max_reward = np.max(rewards_list)
        min_reward = np.min(rewards_list)
        std_reward = np.std(rewards_list)
        
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Average Reward: {avg_reward:.4f}")
        logger.info(f"Maximum Reward: {max_reward:.4f}")
        logger.info(f"Minimum Reward: {min_reward:.4f}")
        logger.info(f"Reward Standard Deviation: {std_reward:.4f}")
        logger.info(f"Number of Variants: {NUM_GENERATIONS}")
        
        # Find the best variant
        best_idx = np.argmax(rewards_list)
        logger.info(f"Best Variant (Index {best_idx+1}): {variants[best_idx]}")
        logger.info(f"Best Variant Reward: {max_reward:.4f}")
        
        # Save summary to a separate file
        with open("evaluation_summary.txt", 'w') as f:
            f.write("EvoDiff Model Evaluation Summary\n")
            f.write("="*50 + "\n")
            f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Checkpoint: {MODEL_CHECKPOINT_PATH}\n")
            f.write(f"Number of Variants: {NUM_GENERATIONS}\n")
            f.write(f"Average Reward: {avg_reward:.4f}\n")
            f.write(f"Maximum Reward: {max_reward:.4f}\n")
            f.write(f"Minimum Reward: {min_reward:.4f}\n")
            f.write(f"Reward Standard Deviation: {std_reward:.4f}\n")
            f.write(f"Best Variant (Index {best_idx+1}): {variants[best_idx]}\n")
            f.write(f"Best Variant Reward: {max_reward:.4f}\n")
    
    logger.info("\nEvaluation complete! Results saved to evaluation_results.csv and evaluation_summary.txt")

if __name__ == '__main__':
    evaluate_trained_model()