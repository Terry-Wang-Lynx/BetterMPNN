import numpy as np
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reward function weight configuration
REWARD_WEIGHTS = {
    'pae': 0.4,
    'iptm': 0.5,
    'ptm_designed': 0.1,
    'clash_penalty': 0.1
}

# Maximum PAE value for converting to pAC
PAE_MAX = 31.75

def exponential_decay_reward(value, optimal_value, scale=1.0, direction='maximize'):
    """
    Convert metric to reward value using exponential decay function
    """
    if direction == 'maximize':
        normalized = value / optimal_value
        return np.exp(-(1 - normalized) / scale) if normalized <= 1 else 1.0
    else:
        normalized = optimal_value / value if value > 0 else 0
        return np.exp(-(1 - normalized) / scale) if normalized <= 1 else 1.0


def find_summary_file(directory):
    """
    Find summary_confidences.json file in specified directory
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("protein-protein_complex_") and file.endswith("_summary_confidences.json"):
                return os.path.join(root, file)
    return None


def parse_summary_file(summary_file_path):
    """
    Parse summary_confidences.json file
    """
    try:
        with open(summary_file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error parsing summary file: {e}")
        return None


def calculate_comprehensive_reward(summary_data, reward_weights):
    """
    Calculate comprehensive reward based on AlphaFold3 output
    """
    # Extract basic metrics
    iptm = summary_data.get('iptm', 0.0)
    has_clash = summary_data.get('has_clash', 1.0)
    ranking_score = summary_data.get('ranking_score', 0.0)
    
    # Extract chain PTM information
    chain_ptm = summary_data.get('chain_ptm', [0.0, 0.0])
    if chain_ptm and len(chain_ptm) >= 2:
        ptm_designed = chain_ptm[1]
    else:
        ptm_designed = summary_data.get('ptm', 0.0)

    # Extract inter-chain PAE information
    chain_pair_pae_min = summary_data.get('chain_pair_pae_min', [])
    if chain_pair_pae_min and len(chain_pair_pae_min) >= 2:
        a_to_b = chain_pair_pae_min[0][1] if len(chain_pair_pae_min[0]) > 1 else PAE_MAX
        b_to_a = chain_pair_pae_min[1][0] if len(chain_pair_pae_min) > 1 else PAE_MAX
        mean_pae = (a_to_b + b_to_a) / 2.0
    else:
        mean_pae = PAE_MAX  # Default high PAE value

    # Calculate component rewards
    pae_reward = 1.0 - (mean_pae / PAE_MAX)
    iptm_reward = iptm
    ptm_reward = ptm_designed
    clash_penalty = 0.0 if has_clash == 0.0 else -1.0

    # Calculate comprehensive reward
    total_reward = (
        reward_weights['pae'] * pae_reward +
        reward_weights['iptm'] * iptm_reward +
        reward_weights['ptm_designed'] * ptm_reward +
        reward_weights['clash_penalty'] * clash_penalty
    )
    
    total_reward = max(0.0, min(1.0, total_reward))
    
    # Return reward and detailed information
    return total_reward, {
        'total_reward': total_reward,
        'mean_pae': mean_pae,
        'pae_reward': pae_reward,
        'iptm': iptm,
        'iptm_reward': iptm_reward,
        'ptm_designed': ptm_designed,
        'ptm': ptm_designed,
        'ptm_reward': ptm_reward,
        'has_clash': has_clash,
        'clash_penalty': clash_penalty,
        'chain_ptm': chain_ptm,
        'ranking_score': ranking_score
    }