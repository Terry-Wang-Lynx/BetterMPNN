import torch
import torch.optim as optim
import numpy as np
import copy
import random
import matplotlib.pyplot as plt

# 从EvoDiff项目中导入必要的模块
from evodiff.pretrained import OA_DM_38M # 使用一个较小的模型进行快速测试
from evodiff.utils import Tokenizer

'''
================================================================================
SECTION 1: 核心函数 (未作任何修改)
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
    使用 EvoDiff OADM 模型为一个给定的蛋白质序列生成变体。
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

def calculate_basic_amino_acid_percentage(sequence: str) -> float:
    """
    计算一个氨基酸序列中碱性氨基酸的百分比。
    """
    basic_amino_acids = {'R', 'K', 'H'}
    if not sequence:
        return 0.0
    sequence_length = len(sequence)
    basic_aa_count = 0
    for amino_acid in sequence.upper():
        if amino_acid in basic_amino_acids:
            basic_aa_count += 1
    percentage = (basic_aa_count / sequence_length) * 100
    return percentage


'''
================================================================================
SECTION 2: GRPO 强化学习主流程 (多路径训练)
================================================================================
'''

def grpo_finetune_evodiff():
    """
    使用 GRPO 算法对 EvoDiff 模型进行微调的主函数。
    """
    # --- 1. 超参数设置 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 1e-5
    training_steps = 300
    num_generations = 4  # 每次为同一个序列生成4个变体
    num_paths_per_variant = 4 # 新增: 为每个变体采样4条路径进行训练
    beta = 0.1           # KL 散度惩罚的权重
    # clip_value = 0.5

    print(f"使用设备: {device}")

    # --- 2. 模型和 Tokenizer 加载 ---
    print("正在加载预训练模型...")
    model, _, tokenizer, _ = OA_DM_38M()
    model.to(device)
    model.train()

    ref_model = copy.deepcopy(model)
    ref_model.to(device)
    ref_model.eval()

    # --- 3. 优化器设置 ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 4. 训练数据设置 ---
    base_sequence = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"
    num_mutations = 15
    seq_len = len(base_sequence)

    print(f"原始序列碱性氨基酸含量: {calculate_basic_amino_acid_percentage(base_sequence):.2f}%")
    print("-" * 50)
    
    losses_history, rewards_history, kls_history = [], [], []

    # --- 5. GRPO 训练循环 ---
    for step in range(training_steps):
        positions_to_mutate = sorted(random.sample(range(seq_len), num_mutations))
        
        # a. 生成 (Rollout) - 使用单条随机路径
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

        # b. 奖励计算
        rewards = torch.tensor([calculate_basic_amino_acid_percentage(v) for v in variants], device=device)

        # c. 优势计算 (组内归一化)
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        # --- d. 损失计算 (核心修改) ---
        batch_loss = 0
        batch_kl_div = 0
        
        # 遍历每一个生成的变体及其对应的advantage
        for i in range(num_generations):
            variant_sequence = variants[i]
            advantage = advantages[i]

            # 为这一个变体，采样多条路径进行训练
            for _ in range(num_paths_per_variant):
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
                batch_loss += loss
                batch_kl_div += kl_div.item()

        # 对所有变体和所有路径的损失进行平均
        total_loss = batch_loss / (num_generations * num_paths_per_variant)
        avg_kl_div = batch_kl_div / (num_generations * num_paths_per_variant)

        # e. 优化
        optimizer.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        losses_history.append(total_loss.item())
        rewards_history.append(mean_reward.item())
        kls_history.append(avg_kl_div)

        # f. 打印日志
        if step % 5 == 0:
            print(f"步骤: {step+1}/{training_steps} | "
                  f"平均奖励: {mean_reward.item():.2f}% | "
                  f"损失: {total_loss.item():.4f} | "
                  f"KL散度: {avg_kl_div:.4f}")

    # --- 6. 训练结束后绘制图表 ---
    print("训练完成，正在生成图表...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('GRPO', fontsize = 16)

    ax1.plot(losses_history, 'b-', label='total loss')
    ax1.set_xlabel('training steps'); ax1.set_ylabel('loss '); ax1.set_title('Training Loss'); ax1.legend()
    ax1.grid(True, linestyle='--', alpha = 0.7)
    
    ax2.plot(rewards_history, 'g-', label='average reward')
    ax2.set_xlabel('training steps'); ax2.set_ylabel('basic amino acid (%)'); ax2.set_title('Average Reward'); ax2.legend()
    ax2.grid(True, linestyle='--', alpha = 0.7)

    ax3.plot(kls_history, 'r-', label='KL')
    ax3.set_xlabel('training steps'); ax3.set_ylabel('KL'); ax3.set_title('KL Divergence'); ax3.legend()
    ax3.grid(True, linestyle='--', alpha = 0.7)

    # moving averages
    window = 3
    if len(losses_history) >= window:
        loss_ma = np.convolve(losses_history, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(losses_history)), loss_ma, 'b--', label='moving averages')
        ax1.legend()
        
        reward_ma = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(rewards_history)), reward_ma, 'g--', label='moving averages')
        ax2.legend()
        
        epochs_ma = np.convolve(kls_history, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(kls_history)), epochs_ma, 'r--', label='moving averages')
        ax3.legend()

    plt.savefig("grpo_multi_path_training_metrics.png", dpi = 300, bbox_inches = 'tight')
    plt.show()


if __name__ == '__main__':
    grpo_finetune_evodiff()