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


def test_grpo_policy_gradient_sign():
    # High advantage should push log-probs up (negative grad), low advantage down.
    current = torch.zeros(2, 4, requires_grad=True)
    ref = torch.zeros(2, 4)  # KL = 0, isolate the policy-gradient term
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 4)
    out = compute_grpo_loss(current, ref, adv, mask, beta=0.0)
    out.loss.backward()
    assert (current.grad[0] < 0).all()   # +adv -> increase log-prob
    assert (current.grad[1] > 0).all()   # -adv -> decrease log-prob


def test_grpo_loss_respects_mask():
    # Construct per-token KL that differs between the first and second half of
    # each sequence, so masking out one half must change the loss.
    batch, seq = 2, 6
    ref = torch.zeros(batch, seq)
    current = torch.zeros(batch, seq)
    current[:, 3:] = 0.5  # nonzero ref-cur only in the second half -> KL there
    current = current.requires_grad_(True)
    adv = torch.tensor([1.0, -1.0])

    full = torch.ones(batch, seq)
    first_half = torch.zeros(batch, seq)
    first_half[:, :3] = 1.0  # masks out the KL-bearing tokens

    out_full = compute_grpo_loss(current, ref, adv, full, beta=0.5)
    out_first = compute_grpo_loss(current, ref, adv, first_half, beta=0.5)
    # The masked-out KL must move the loss; equality would mean the mask is ignored.
    assert not torch.isclose(out_full.loss, out_first.loss)
    # Masking only KL-free tokens leaves zero KL contribution.
    assert abs(out_first.kl.item()) < 1e-6
    assert out_full.kl.item() > 0
