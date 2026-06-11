"""Unit tests for RMSD / DRMSD geometry utilities."""

import numpy as np

from bettermpnn.utils.rmsd import calculate_rmsd, compute_local_drmsd


def test_rmsd_identical_is_zero():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert calculate_rmsd(coords, coords.copy(), align=True) < 1e-9


def test_rmsd_invariant_to_rigid_motion_when_aligned():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 3))
    # Rotate + translate; Kabsch alignment should recover ~0 RMSD.
    theta = 0.7
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    moved = coords @ R.T + np.array([5.0, -3.0, 2.0])
    assert calculate_rmsd(coords, moved, align=True) < 1e-6
    # Without alignment the translation dominates -> large RMSD.
    assert calculate_rmsd(coords, moved, align=False) > 1.0


def test_rmsd_mismatched_lengths_raises_by_default():
    a = np.zeros((5, 3))
    b = np.zeros((3, 3))
    # Strict by default: positional correspondence requires equal lengths.
    try:
        calculate_rmsd(a, b, align=False)
        assert False, "expected ValueError on length mismatch"
    except ValueError:
        pass
    # Opt-in truncation still available.
    assert calculate_rmsd(a, b, align=False, allow_truncate=True) == 0.0


def test_local_drmsd_zero_for_identical():
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(15, 3))
    assert compute_local_drmsd(coords, coords.copy(), seq_sep=6) < 1e-9


def test_local_drmsd_insensitive_to_global_rotation():
    # A pure hinge/rotation preserves local pairwise distances -> DRMSD ~0,
    # even though global RMSD (unaligned) would be large.
    rng = np.random.default_rng(2)
    coords = rng.normal(size=(30, 3))
    theta = 1.2
    R = np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)],
    ])
    rotated = coords @ R.T
    assert compute_local_drmsd(coords, rotated, seq_sep=6) < 1e-6
