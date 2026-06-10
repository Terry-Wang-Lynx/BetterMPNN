"""GRPO (Group Relative Policy Optimization) — pure RL math, no protein concepts."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GRPOResult:
    loss: torch.Tensor
    kl: torch.Tensor
    policy_loss: torch.Tensor


def compute_advantages(
    rewards: torch.Tensor,
    shaping_alpha: Optional[float] = None,
    scale_factor: float = 5.0,
) -> torch.Tensor:
    """Compute group-relative advantages from a batch of rewards.

    Args:
        rewards: Tensor of shape [batch] with scalar rewards.
        shaping_alpha: If set, apply power-law shaping: sign(r) * |r|^alpha.
        scale_factor: Fallback scaling when std is near zero.

    Returns:
        Tensor of shape [batch] with normalized advantages.
    """
    if len(rewards) <= 1:
        return torch.zeros_like(rewards)

    r = rewards
    if shaping_alpha is not None:
        r = torch.sign(r) * torch.pow(torch.abs(r), shaping_alpha)

    mean = r.mean()
    std = r.std()
    if std > 1e-8:
        return (r - mean) / std
    else:
        return (r - mean) * scale_factor


def compute_grpo_loss(
    current_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.01,
) -> GRPOResult:
    """Compute GRPO loss with KL regularization.

    Args:
        current_logps: [batch, seq_len] log-probs from current policy.
        ref_logps: [batch, seq_len] log-probs from frozen reference policy.
        advantages: [batch] scalar advantages per sequence.
        mask: [batch, seq_len] binary mask for designed positions.
        beta: KL penalty coefficient.

    Returns:
        GRPOResult with total loss, mean KL, and policy loss.
    """
    # Unbiased KL estimator: exp(ref - cur) - (ref - cur) - 1
    kl = torch.exp(ref_logps - current_logps) - (ref_logps - current_logps) - 1

    # Policy gradient term (importance-weighted by detached ratio)
    policy_term = torch.exp(current_logps - current_logps.detach()) * advantages.unsqueeze(1)

    # Per-token loss
    per_token_loss = -(policy_term - beta * kl)

    # Masked mean
    num_valid = torch.clamp(mask.sum(dim=1), min=1.0)
    loss = ((per_token_loss * mask).sum(dim=1) / num_valid).mean()
    kl_mean = ((kl * mask).sum(dim=1) / num_valid).mean()
    policy_loss = -((policy_term * mask).sum(dim=1) / num_valid).mean()

    return GRPOResult(loss=loss, kl=kl_mean, policy_loss=policy_loss)
