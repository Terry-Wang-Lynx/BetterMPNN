import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict
import subprocess
import glob
import math
import openpyxl
import time
import zipfile
import openpyxl.utils.exceptions

# 从EvoDiff项目中导入必要的模块
from evodiff.pretrained import OA_DM_38M # 可选择640M进行对比
from evodiff.utils import Tokenizer

GROUP_NAME = "ES_1"  # 定义组

# 全局变量，用于记录训练指标历史
TRAINING_METRICS_HISTORY = []

'''
================================================================================
SECTION 1: 核心函数
================================================================================
'''

def get_log_probs(model, sequence, decode_order, tokenizer, device='cuda'):
    """
    计算给定序列中，由OADM在特定位置和顺序下生成时，每个token的对数概率 (log probability)。
    """
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

def generate_variant(model, sequence: str, mask_positions: list[int], decode_order: list[int],
                     tokenizer, device: str = 'cuda'):
    """
    使用 EvoDiff OADM 模型为一个给定的蛋白质序列生成变体
    """
    all_aas = tokenizer.all_aas
    sequence_list = list(sequence)
    for pos in mask_positions:
        sequence_list[pos] = '#'
    masked_sequence = ''.join(sequence_list)
    sample = torch.tensor(tokenizer.tokenizeMSA(masked_sequence), dtype=torch.long, device=device)
    for i in decode_order:
        timestep = torch.tensor([0], device=device)
        with torch.no_grad(): # 生成过程不需要计算梯度
            prediction = model(sample.unsqueeze(0), timestep)
        logits = prediction[:, i, :len(all_aas) - 6]
        p = torch.nn.functional.softmax(logits, dim=1)
        new_token_id = torch.multinomial(p, num_samples=1)
        sample[i] = new_token_id.squeeze()
    return tokenizer.untokenize(sample)

def parse_all_scores_from_excel(filepath):
    """
    从 summary.xlsx 文件的指定单元格中精确提取所有三个指标。
    """
    try:
        # 使用 openpyxl 直接加载工作簿
        workbook = openpyxl.load_workbook(filepath, data_only=True)
        sheet = workbook.active 

        iptm_str = sheet['F3'].value
        ranking_score_str = sheet['H3'].value
        affinity_str = sheet['U2'].value

        iptm = float(iptm_str) if iptm_str is not None else 0.0
        ranking_score = float(ranking_score_str) if ranking_score_str is not None else 0.0
        affinity = float(affinity_str) if affinity_str is not None else 0.0

        return {
            "ranking_score": ranking_score,
            "iptm": iptm,
            "affinity": affinity
        }

    except (FileNotFoundError, ValueError, KeyError, IndexError, TypeError) as e:
        print(f"[警告] 解析Excel分数文件失败: {filepath}. 错误: {e}")
        return {"ranking_score": 0.0, "iptm": 0.0, "affinity": 0.0}

def custom_sigmoid(x, threshold, k):
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - threshold)))
    except OverflowError:
        return 0.0 if x < threshold else 1.0

def calculate_composite_reward(all_scores):
    global TRAINING_METRICS_HISTORY
    iptm = all_scores.get("iptm", 0.0)
    ranking_score = all_scores.get("ranking_score", 0.0)
    binding_affinity = all_scores.get("affinity", 0.0)
    W_IPTM, W_RANKING, SIGMOID_THRESHOLD, SIGMOID_STEEPNESS = 0.3, 0.7, 0.73, 50
    structure_quality_score = (W_IPTM * iptm) + (W_RANKING * ranking_score)
    sigmoid_factor = custom_sigmoid(structure_quality_score, SIGMOID_THRESHOLD, SIGMOID_STEEPNESS)
    affinity_factor = -binding_affinity if binding_affinity < 0 else 0.0
    final_reward = sigmoid_factor * affinity_factor
    metrics_tuple = (final_reward, ranking_score, iptm, binding_affinity)
    TRAINING_METRICS_HISTORY.append(metrics_tuple)
    print(f"  [Metrics Logged] Reward: {final_reward:.3f}, RankScore: {ranking_score:.3f}, ipTM: {iptm:.3f}, Affinity: {binding_affinity:.2f} kcal/mol")
    return final_reward

def get_reward_from_shell_script(sequence):
    """
    为单个序列调用shell脚本，解析结果，并返回复合奖励。
    """
    shell_script_path = "/data/run01/scz0sfc/af3/get_reward_ES_1.sh"
    base_data_dir = f"/data/run01/scz0sfc/af3/data/{GROUP_NAME}"  # 修改为组特定目录
    job_id = f"py_{int(time.time())}_{random.randint(1000, 9999)}"  # 添加随机数避免冲突
    print(f"  已为Shell脚本生成统一Job ID: {job_id}")
    temp_job_dir = os.path.join(base_data_dir, job_id)

    try:
        print(f"  正在为序列执行shell脚本: {sequence[:10]}...")
        submit_cmd = ["sbatch", "--wait", f"--export=ALL,PARENT_JOB_ID={job_id},GROUP_NAME={GROUP_NAME}", shell_script_path, sequence]
        subprocess.run(submit_cmd, capture_output=True, text=True, check=True)
        print(f"  Shell脚本作业 (统一ID: {job_id}) 已完成。")

        # 查找并解析输出的文件
        pipeline_output_dir = os.path.join(temp_job_dir, "af3_pipeline", "output")
        excel_files = glob.glob(os.path.join(pipeline_output_dir, "*.xlsx"))
        
        print(f"查找Excel文件于: {pipeline_output_dir}")
        if not excel_files:
            raise FileNotFoundError(f"在 {pipeline_output_dir} 中未找到summary.xlsx文件")

        # 添加重试机制处理Excel文件损坏
        max_retries = 3
        retry_delay = 5  # 重试间隔(秒)
        
        for attempt in range(max_retries):
            try:
                # 调用解析函数
                all_scores = parse_all_scores_from_excel(excel_files[0])
                break  # 成功则跳出重试循环
            except (openpyxl.utils.exceptions.InvalidFileException, zipfile.BadZipFile, KeyError, ValueError) as e:
                if attempt < max_retries - 1:
                    print(f"  [重试 {attempt+1}/{max_retries}] Excel文件解析失败: {e}, {retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                    # 重新检查文件是否存在
                    excel_files = glob.glob(os.path.join(pipeline_output_dir, "*.xlsx"))
                    if not excel_files:
                        raise FileNotFoundError(f"在 {pipeline_output_dir} 中未找到summary.xlsx文件")
                else:
                    print(f"  [错误] 经过{max_retries}次重试后仍无法解析Excel文件")
                    return 0.0
        
        # 计算复合奖励
        reward = calculate_composite_reward(all_scores)
        return reward

    except subprocess.CalledProcessError as e:
        print(f"[!!] Shell脚本执行失败! Stderr: {e.stderr}")
        return 0.0
    except Exception as e:
        print(f"[!!] 在get_reward期间发生未知错误: {e}")
        return 0.0
    finally:
        if temp_job_dir:
            print(f"  [调试模式] 中间文件已保留在: {temp_job_dir}")

def get_reward(sequences, device='cuda'):
    """为一批序列获取奖励 (串行执行)。"""
    rewards = []
    print(f"\n开始为 {len(sequences)} 个序列获取奖励 (串行执行)...")
    for i, seq in enumerate(sequences):
        print(f"--- 处理序列 {i+1}/{len(sequences)} ---")
        reward = get_reward_from_shell_script(seq)
        rewards.append(reward)
    print("所有序列奖励计算完成。")
    return torch.tensor(rewards, dtype=torch.float, device=device)

class EvolutionaryStrategySelector:
    """
    基于进化策略的位点选择器
    """
    def __init__(self, seq_len, exploration_noise=0.5, learning_rate=0.2, decay_rate=0.99):
        self.seq_len = seq_len
        # 使用随机初始值
        self.values = np.random.uniform(-1, 1, seq_len)
        self.base_exploration_noise = exploration_noise
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.history = []
        self.step = 0
        
    def select_sites(self, top_k=15):
        """
        自适应探索噪声
        """
        current_noise = self.base_exploration_noise * (self.decay_rate ** self.step)
        
        noisy_values = self.values + np.random.normal(0, current_noise, self.seq_len)
        
        selected_sites = []
        remaining_indices = list(range(self.seq_len))
        
        for _ in range(top_k):
            if not remaining_indices:
                break
                
            max_idx = remaining_indices[np.argmax(noisy_values[remaining_indices])]
            selected_sites.append(max_idx)
            remaining_indices.remove(max_idx)
        
        return sorted(selected_sites)
    
    def update(self, selected_sites, reward):
        """
        更新策略
        """
        normalized_reward = np.tanh(reward / 10)
        
        # 更新
        for site in selected_sites:
            self.values[site] += self.learning_rate * (normalized_reward - self.values[site])
        
        # 惩罚
        all_sites = set(range(self.seq_len))
        unselected_sites = all_sites - set(selected_sites)
        
        for site in unselected_sites:
            self.values[site] -= self.learning_rate * 0.1 * normalized_reward
        
        # 记录
        self.history.append({
            'selected_sites': selected_sites,
            'reward': reward,
            'values': self.values.copy()
        })
        
        self.step += 1
    
    def get_top_sites(self, top_k=15):
        """获取当前价值最高的位点（不添加噪声）"""
        return sorted(np.argsort(self.values)[-top_k:])

'''
================================================================================
SECTION 2: GRPO 强化学习主流程
================================================================================
'''

def grpo_finetune_evodiff():
    """
    使用 GRPO 算法对 EvoDiff 模型进行微调的主函数
    """
    # --- 1. 超参数设置 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 1e-6
    training_steps = 1000
    num_generations = 4
    num_paths_per_variant = 4
    beta = 0.3

    print(f"使用设备: {device}")

    # --- 2. 模型和 Tokenizer 加载 ---
    print("正在加载预训练模型...")
    model, _, tokenizer, _ = OA_DM_38M()
    model.to(device)
    model.train()

    ref_model = copy.deepcopy(model)
    ref_model.to(device)
    ref_model.eval()

    # --- 3. 进化策略选择器初始化 ---
    print("初始化进化策略选择器...")
    base_sequence = "AAAATTATTGATGACGAAGAATTAGCAGAAACCAACAAGAAGATCGAAGAATTGGAGAAGAAGGATGAGAAGGAAGCCCAGAAACTGGCGGAAGAACTGGGCGAGAAAATTAAGGAAAATGAGAAGAAACGCAAGGAAGAGAAGGAAAAGTAA"
    seq_len = len(base_sequence)
    num_mutations = 30
    
    selector = EvolutionaryStrategySelector(
        seq_len=seq_len,
        exploration_noise=1.0,
        learning_rate=0.3,
        decay_rate=0.995
    )

    # --- 4. 优化器设置 ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"原始序列: {base_sequence}")
    print("-" * 50)
    
    losses_history, rewards_history, kls_history = [], [], []
    selected_sites_history = []
    top_sites_history = []
    
    # 记录所有序列及其相关信息
    all_sequences = []

    # 创建目录保存图像
    os.makedirs("training_plots", exist_ok=True)
    os.makedirs("all_sequences_ES_1", exist_ok=True)

    # --- 5. GRPO 训练循环 ---
    for step in range(training_steps):
        # a. 使用进化策略选择器选择突变位点
        positions_to_mutate = selector.select_sites(top_k=num_mutations)
        selected_sites_history.append(positions_to_mutate)
        
        # 记录无噪声下的最高价值位点
        top_sites = selector.get_top_sites(top_k=num_mutations)
        top_sites_history.append(top_sites)
        
        # b. 生成 (Rollout) - 使用单条随机路径
        generation_order = positions_to_mutate.copy()
        random.shuffle(generation_order)
        
        variants = []
        for _ in range(num_generations):
            variant = generate_variant(
                model=model,
                sequence=base_sequence,
                mask_positions=positions_to_mutate,
                decode_order=generation_order,
                tokenizer=tokenizer,
                device=device
            )
            variants.append(variant)

        # c. 奖励计算 (通过 AlphaFold3 和 PRODIGY)
        rewards = get_reward(variants, device=device)

        # 记录所有序列及其相关信息
        for i, variant in enumerate(variants):
            reward_val = rewards[i].item()
            # 确保所有值为Python原生类型
            all_sequences.append({
                'sequence': variant,
                'reward': float(reward_val),
                'step': int(step),
                'positions': [int(pos) for pos in positions_to_mutate]
            })
            
            # 每10步保存一次所有序列
            if step % 10 == 0:
                with open(f"all_sequences_ES_1/all_sequences_step_{step}.json", "w") as f:
                    # 按奖励排序
                    sorted_sequences = sorted(all_sequences, key=lambda x: x['reward'], reverse=True)
                    # 确保所有数据可序列化
                    json_serializable = []
                    for seq_info in sorted_sequences:
                        json_serializable.append({
                            'sequence': seq_info['sequence'],
                            'reward': float(seq_info['reward']),
                            'step': int(seq_info['step']),
                            'positions': [int(pos) for pos in seq_info['positions']]
                        })
                    json.dump(json_serializable, f, indent=2)

        # d. 优势计算 (组内归一化)
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        # --- e. 损失计算 ---
        batch_loss = 0
        batch_kl_div = 0

        # 设置小批量大小
        mini_batch_size = 2  # 可以根据内存调整

        # 遍历每一个生成的变体及其对应的advantage
        for i in range(num_generations):
            variant_sequence = variants[i]
            advantage = advantages[i]

            # 为这一个变体，采样多条路径进行训练
            # 使用小批量处理来减少内存使用
            num_mini_batches = (num_paths_per_variant + mini_batch_size - 1) // mini_batch_size
            
            for batch_idx in range(num_mini_batches):
                start_idx = batch_idx * mini_batch_size
                end_idx = min((batch_idx + 1) * mini_batch_size, num_paths_per_variant)
                
                mini_batch_loss = 0
                mini_batch_kl_div = 0
                
                # 处理当前小批量
                for _ in range(start_idx, end_idx):
                    # 随机采样一条新的训练路径
                    training_order = positions_to_mutate.copy()
                    random.shuffle(training_order)
                    
                    # 使用当前模型计算 logp
                    with torch.set_grad_enabled(True):
                        log_probs = get_log_probs(model, variant_sequence, training_order, tokenizer, device)
                    
                    # 使用参考模型计算 ref_logp
                    with torch.no_grad():
                        ref_log_probs = get_log_probs(ref_model, variant_sequence, training_order, tokenizer, device)

                    policy_loss = -(log_probs * advantage).sum()
                    
                    kl_div_per_token = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
                    kl_div = kl_div_per_token.sum()
                    
                    loss = policy_loss + beta * kl_div
                    mini_batch_loss += loss
                    mini_batch_kl_div += kl_div.item()
                
                # 对小批量损失进行平均并反向传播
                mini_batch_loss = mini_batch_loss / (end_idx - start_idx)
                mini_batch_loss.backward()
                
                # 累积损失
                batch_loss += mini_batch_loss.item() * (end_idx - start_idx)
                batch_kl_div += mini_batch_kl_div
                
                # 清除计算图以释放内存
                del mini_batch_loss
                torch.cuda.empty_cache()

        # 对所有变体和所有路径的损失进行平均
        total_loss = batch_loss / (num_generations * num_paths_per_variant)
        avg_kl_div = batch_kl_div / (num_generations * num_paths_per_variant)

        # f. 更新模型参数
        optimizer.step()
        optimizer.zero_grad()
        
        # g. 更新进化策略选择器
        mean_reward_value = mean_reward.item()
        selector.update(positions_to_mutate, mean_reward_value)
        
        # h. 记录指标
        losses_history.append(total_loss)
        rewards_history.append(mean_reward_value)
        kls_history.append(avg_kl_div)

        # i. 打印日志
        if step % 1 == 0:  # 每步都打印，因为训练步数减少了
            print(f"步骤: {step+1}/{training_steps} | "
                f"平均奖励: {mean_reward_value:.2f} | "
                f"策略损失: {total_loss:.4f} | "
                f"KL散度: {avg_kl_div:.4f}")
                
        # j. 每10步输出图像和位点信息
        if step % 10 == 0:
            print(f"步骤 {step} 选择的突变位点: {positions_to_mutate}")
            print(f"步骤 {step} 最高价值位点: {top_sites}")
            
            # 输出当前序列统计
            if all_sequences:
                sorted_sequences = sorted(all_sequences, key=lambda x: x['reward'], reverse=True)
                top_5_sequences = sorted_sequences[:5]
                print(f"步骤 {step} 前5高奖励序列:")
                for i, seq_info in enumerate(top_5_sequences):
                    print(f"  排名 {i+1}: 奖励={seq_info['reward']:.2f}, 步数={seq_info['step']}")
            
            # 绘制并保存当前训练指标
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'GRPO Evolutionary Strategy (Step {step})', fontsize=16)

            ax1.plot(losses_history, 'b-', label='policy loss')
            ax1.set_xlabel('training steps'); ax1.set_ylabel('loss'); ax1.set_title('Policy Loss'); ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            ax2.plot(rewards_history, 'g-', label='average reward')
            ax2.set_xlabel('training steps'); ax2.set_ylabel('reward'); ax2.set_title('Average Reward'); ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)

            ax3.plot(kls_history, 'r-', label='KL')
            ax3.set_xlabel('training steps'); ax3.set_ylabel('KL'); ax3.set_title('KL Divergence'); ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # 绘制位点选择变化
            site_changes = np.zeros((len(selected_sites_history), seq_len))
            for i, sites in enumerate(selected_sites_history):
                for site in sites:
                    site_changes[i, site] = 1
            
            ax4.imshow(site_changes.T, aspect='auto', cmap='Blues')
            ax4.set_xlabel('Training Step')
            ax4.set_ylabel('Amino Acid Position')
            ax4.set_title('Site Selection Over Time')

            # 添加moving average
            window = 5
            if len(losses_history) >= window:
                loss_ma = np.convolve(losses_history, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(losses_history)), loss_ma, 'b--', label='moving average')
                ax1.legend()
                
                reward_ma = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(rewards_history)), reward_ma, 'g--', label='moving average')
                ax2.legend()
                
                kl_ma = np.convolve(kls_history, np.ones(window)/window, mode='valid')
                ax3.plot(range(window-1, len(kls_history)), kl_ma, 'r--', label='moving average')
                ax3.legend()

            plt.tight_layout()
            plt.savefig(f"training_plots/grpo_evolutionary_strategy_training_metrics_step_{step}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)  # 关闭图像以避免内存泄漏

    # --- 6. 训练结束后绘制最终图表 ---
    print("训练完成，正在生成最终图表...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GRPO with Evolutionary Strategy Selector (Final)', fontsize=16)

    ax1.plot(losses_history, 'b-', label='policy loss')
    ax1.set_xlabel('training steps'); ax1.set_ylabel('loss'); ax1.set_title('Policy Loss'); ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(rewards_history, 'g-', label='average reward')
    ax2.set_xlabel('training steps'); ax2.set_ylabel('reward'); ax2.set_title('Average Reward'); ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    ax3.plot(kls_history, 'r-', label='KL')
    ax3.set_xlabel('training steps'); ax3.set_ylabel('KL'); ax3.set_title('KL Divergence'); ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制位点选择变化
    site_changes = np.zeros((len(selected_sites_history), seq_len))
    for i, sites in enumerate(selected_sites_history):
        for site in sites:
            site_changes[i, site] = 1
    
    ax4.imshow(site_changes.T, aspect='auto', cmap='Blues')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Amino Acid Position')
    ax4.set_title('Site Selection Over Time')
    
    plt.tight_layout()
    plt.savefig("grpo_evolutionary_strategy_training_metrics_final.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制位点选择热图和价值变化
    plot_site_selection_analysis(selector, selected_sites_history, top_sites_history, seq_len)
    
    # 输出最后10步选择的位点
    print("\n最后10步选择的位点:")
    for i in range(max(0, len(selected_sites_history)-10), len(selected_sites_history)):
        print(f"步骤 {i}: {selected_sites_history[i]}")
    
    # 输出所有序列
    print_all_sequences(all_sequences)

def print_all_sequences(all_sequences):
    """
    输出所有序列及其信息
    """
    if not all_sequences:
        print("没有找到任何序列")
        return
        
    # 按奖励排序
    sorted_sequences = sorted(all_sequences, key=lambda x: x['reward'], reverse=True)
    
    print("\n" + "="*80)
    print("所有序列排名")
    print("="*80)
    
    # 输出前50个序列
    for i, seq_info in enumerate(sorted_sequences[:50]):
        print(f"\n排名 {i+1}:")
        print(f"  奖励: {seq_info['reward']:.2f}")
        print(f"  生成步数: {seq_info['step']}")
        print(f"  突变位点: {seq_info['positions']}")
        print(f"  序列: {seq_info['sequence']}")
    
    # 保存到文件
    with open("all_sequences_ES_1/final_all_sequences.json", "w") as f:
        json.dump(sorted_sequences, f, indent=2)
    
    # 绘制奖励分布
    rewards = [s['reward'] for s in sorted_sequences]
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of All Sequence Rewards')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("all_sequences_ES_1/reward_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_site_selection_analysis(selector, selected_sites_history, top_sites_history, seq_len):
    """
    绘制位点选择分析图
    """
    # 创建选择频率矩阵
    selection_freq = np.zeros(seq_len)
    for sites in selected_sites_history:
        for site in sites:
            selection_freq[site] += 1
    
    # 归一化
    selection_freq = selection_freq / len(selected_sites_history)
    
    # 绘制选择频率热图
    plt.figure(figsize=(15, 5))
    plt.bar(range(seq_len), selection_freq)
    plt.xlabel('Amino Acid Position')
    plt.ylabel('Selection Frequency')
    plt.title('Site Selection Frequency During Training')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("site_selection_frequency_improved_es.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制最终价值向量
    plt.figure(figsize=(15, 5))
    plt.bar(range(seq_len), selector.values)
    plt.xlabel('Amino Acid Position')
    plt.ylabel('Final Value')
    plt.title('Final Site Values from Improved Evolutionary Strategy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("final_site_values_improved_es.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制价值变化热图
    if len(selector.history) > 0:
        # 提取历史价值
        value_history = np.array([h['values'] for h in selector.history])
        
        # 绘制热图
        plt.figure(figsize=(15, 8))
        plt.imshow(value_history.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Site Value')
        plt.xlabel('Training Step')
        plt.ylabel('Amino Acid Position')
        plt.title('Site Value Evolution During Training')
        plt.savefig("site_value_evolution_improved_es.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    # 绘制选择位点与最高价值位点的差异
    divergence = []
    for i in range(len(selected_sites_history)):
        selected_set = set(selected_sites_history[i])
        top_set = set(top_sites_history[i])
        # 计算Jaccard距离
        divergence.append(1 - len(selected_set & top_set) / len(selected_set | top_set))
    
    plt.figure(figsize=(10, 5))
    plt.plot(divergence, 'r-')
    plt.xlabel('Training Step')
    plt.ylabel('Selection Divergence')
    plt.title('Divergence Between Selected Sites and Top Value Sites')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("selection_divergence.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    grpo_finetune_evodiff()