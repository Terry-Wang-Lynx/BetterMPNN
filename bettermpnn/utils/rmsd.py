"""RMSD / DRMSD utilities for conformational-change assessment.

Residue correspondence is positional (the i-th Cα of each array is paired), so
the two coordinate sets must have equal length. A mismatch usually signals a
wrong chain mapping, missing residues, or mismatched references and is treated
as an error rather than silently truncated.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calculate_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    align: bool = True,
    allow_truncate: bool = False,
) -> float:
    """Calculate RMSD between two equal-length coordinate sets (Kabsch alignment).

    Args:
        coords1: First coordinate set (N, 3)
        coords2: Second coordinate set (N, 3)
        align: Whether to perform Kabsch alignment first
        allow_truncate: If True, truncate to the shorter length on mismatch
            (with a warning) instead of raising. Off by default.

    Returns:
        RMSD in Angstroms.

    Raises:
        ValueError: if the inputs differ in length and ``allow_truncate`` is False.
    """
    if len(coords1) != len(coords2):
        if not allow_truncate:
            raise ValueError(
                f"Coordinate length mismatch: {len(coords1)} vs {len(coords2)}. "
                f"RMSD requires positional residue correspondence (equal lengths). "
                f"Pass allow_truncate=True only if you intend to compare prefixes."
            )
        n = min(len(coords1), len(coords2))
        logger.warning(f"Coordinate length mismatch: {len(coords1)} vs {len(coords2)}. Truncating to {n}.")
        coords1 = coords1[:n]
        coords2 = coords2[:n]

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
    """Compute local distance RMSD (DRMSD) on sequence-local residue pairs.

    Unlike global RMSD, this metric is IMMUNE to domain hinge motions
    (open↔closed conformational change) but sensitive to true unfolding.
    It compares pairwise Cα distances for residues within `seq_sep` positions.

    Args:
        ref_ca: Reference Cα coordinates (N, 3)
        pred_ca: Predicted Cα coordinates (N, 3)
        seq_sep: Maximum sequence separation to consider (default 6)

    Returns:
        local_drmsd: Local distance RMSD in Å (lower = fold preserved)
    """
    if len(ref_ca) != len(pred_ca) and not allow_truncate:
        raise ValueError(
            f"Coordinate length mismatch: {len(ref_ca)} vs {len(pred_ca)}. "
            f"DRMSD requires positional residue correspondence (equal lengths)."
        )
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
