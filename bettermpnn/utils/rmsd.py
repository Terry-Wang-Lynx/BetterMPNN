"""RMSD / DRMSD utilities. Residue correspondence is positional, so inputs must
be equal length; a mismatch is an error unless allow_truncate is set."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calculate_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    align: bool = True,
    allow_truncate: bool = False,
) -> float:
    """RMSD (Å) between two equal-length (N, 3) coordinate sets, with Kabsch alignment."""
    if len(coords1) != len(coords2):
        if not allow_truncate:
            raise ValueError(f"Coordinate length mismatch: {len(coords1)} vs {len(coords2)}.")
        n = min(len(coords1), len(coords2))
        logger.warning(f"Length mismatch {len(coords1)} vs {len(coords2)}; truncating to {n}.")
        coords1, coords2 = coords1[:n], coords2[:n]

    if len(coords1) == 0:
        return 0.0

    if align:
        # Kabsch alignment
        c1 = coords1 - coords1.mean(axis=0)
        c2 = coords2 - coords2.mean(axis=0)

        H = c1.T @ c2
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        c2_aligned = c2 @ R
        diff = c1 - c2_aligned
    else:
        diff = coords1 - coords2

    return float(np.sqrt((diff ** 2).sum() / len(coords1)))


def compute_local_drmsd(
    ref_ca: np.ndarray,
    pred_ca: np.ndarray,
    seq_sep: int = 6,
    allow_truncate: bool = False,
) -> float:
    """Local distance RMSD over residue pairs within `seq_sep`: immune to hinge
    motions (open↔closed) but sensitive to unfolding. Lower = fold preserved."""
    if len(ref_ca) != len(pred_ca) and not allow_truncate:
        raise ValueError(f"Coordinate length mismatch: {len(ref_ca)} vs {len(pred_ca)}.")
    n = min(len(ref_ca), len(pred_ca))
    if n < 2:
        return 0.0

    ref_ca = ref_ca[:n]
    pred_ca = pred_ca[:n]

    diffs_sq = []
    for i in range(n):
        for j in range(i + 1, min(i + seq_sep + 1, n)):
            d_ref = np.linalg.norm(ref_ca[i] - ref_ca[j])
            d_pred = np.linalg.norm(pred_ca[i] - pred_ca[j])
            diffs_sq.append((d_ref - d_pred) ** 2)

    if not diffs_sq:
        return 0.0

    return float(np.sqrt(np.mean(diffs_sq)))
