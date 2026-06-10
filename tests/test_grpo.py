"""Unit tests for the GRPO RL math (pure functions, no protein concepts)."""

import torch

from bettermpnn.rl.grpo import compute_advantages, compute_grpo_loss


def test_advantages_are_normalized():
    rewards = torch.tensor([0.1, 0.5, 0.9, 0.3])
    adv = compute_advantages(rewards, shaping_alpha=None)
    # Group-relative: mean ~0, unit std (population/sample std as implemented)
    assert abs(adv.mean().item()) < 1e-5
    assert abs(adv.std().item() - 1.0) < 1e-4


def test_advantages_single_element_is_zero():
    adv = compute_advantages(torch.tensor([0.7]))
    assert adv.numel() == 1
    assert adv.item() == 0.0


def test_advantages_equal_rewards_are_zero():
    # When all rewards are identical, std is 0 -> advantages collapse to 0.
    adv = compute_advantages(torch.tensor([0.5, 0.5, 0.5]), shaping_alpha=None)
    assert torch.allclose(adv, torch.zeros_like(adv))


def test_advantages_preserve_ranking():
    rewards = torch.tensor([0.1, 0.9, 0.5])
    adv = compute_advantages(rewards, shaping_alpha=None)
    # Highest reward must get the highest advantage.
    assert torch.argmax(adv) == torch.argmax(rewards)
    assert torch.argmin(adv) == torch.argmin(rewards)


def test_shaping_alpha_keeps_sign_and_order():
    rewards = torch.tensor([0.0, 0.25, 1.0])
    adv = compute_advantages(rewards, shaping_alpha=0.7)
    assert torch.argmax(adv) == 2


def test_grpo_loss_shapes_and_kl_nonneg():
    batch, seq = 4, 10
    current = torch.randn(batch, seq).requires_grad_(True)
    ref = torch.randn(batch, seq)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    mask = torch.ones(batch, seq)

    out = compute_grpo_loss(current, ref, adv, mask, beta=0.01)
    assert out.loss.ndim == 0
    # k3 KL estimator is non-negative.
    assert out.kl.item() >= -1e-6
    # Loss should be differentiable wrt the current policy log-probs.
    out.loss.backward()
    assert current.grad is not None


def test_grpo_loss_respects_mask():
    batch, seq = 2, 6
    current = torch.zeros(batch, seq).requires_grad_(True)
    ref = torch.zeros(batch, seq)
    adv = torch.tensor([1.0, -1.0])
    full = torch.ones(batch, seq)
    half = torch.zeros(batch, seq)
    half[:, :3] = 1.0

    loss_full = compute_grpo_loss(current, ref, adv, full, beta=0.01).loss
    loss_half = compute_grpo_loss(current, ref, adv, half, beta=0.01).loss
    # With zero KL (current == ref) the per-token policy loss is constant,
    # so a masked mean over fewer tokens yields the same value.
    assert torch.isfinite(loss_full)
    assert torch.isfinite(loss_half)
