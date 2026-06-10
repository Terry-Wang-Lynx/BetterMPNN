"""Training curve visualization."""

import os
import logging

logger = logging.getLogger(__name__)


def plot_training_curves(results: list[dict], output_dir: str) -> None:
    """Plot and save training progress charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available in this environment. Training curves will NOT be generated.")
        logger.info("To enable plotting, please install matplotlib: pip install matplotlib")
        return
    except Exception as e:
        logger.error(f"Failed to initialize matplotlib: {e}")
        return

    if len(results) < 1:
        return

    steps = [r["step"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Reward with min/max band
    mean_r = [r["mean_reward"] for r in results]
    axes[0, 0].plot(steps, mean_r, "b-o", markersize=2, label="Mean")
    if "max_reward" in results[0]:
        max_r = [r["max_reward"] for r in results]
        min_r = [r["min_reward"] for r in results]
        axes[0, 0].fill_between(steps, min_r, max_r, alpha=0.2, color="b")
        axes[0, 0].plot(steps, max_r, "b--", alpha=0.4, linewidth=0.8, label="Max")
    axes[0, 0].set_title("Reward")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Loss & KL
    ax_loss = axes[0, 1]
    ax_loss.plot(steps, [r["loss"] for r in results], "r-", linewidth=1, label="Loss")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss", color="r")
    ax_loss.tick_params(axis="y", labelcolor="r")
    ax_kl = ax_loss.twinx()
    ax_kl.plot(steps, [r["kl"] for r in results], "g--", linewidth=1, label="KL")
    ax_kl.set_ylabel("KL Divergence", color="g")
    ax_kl.tick_params(axis="y", labelcolor="g")
    axes[0, 1].set_title("Loss & KL")

    # 3. iPTM and pTM (if available)
    if "iptm" in results[0]:
        axes[1, 0].plot(steps, [r.get("iptm", 0) for r in results], "m-o", markersize=2, label="iPTM")
    if "ptm" in results[0]:
        axes[1, 0].plot(steps, [r.get("ptm", 0) for r in results], "c-s", markersize=2, label="pTM")
    if "ranking_score" in results[0]:
        axes[1, 0].plot(steps, [r.get("ranking_score", 0) for r in results], "k-^", markersize=2, label="Ranking")
    axes[1, 0].set_title("Structural Metrics")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. PAE and clash
    if "mean_pae" in results[0]:
        axes[1, 1].plot(steps, [r.get("mean_pae", 0) for r in results], "r-o", markersize=2, label="Mean PAE")
        axes[1, 1].set_ylabel("PAE", color="r")
        axes[1, 1].tick_params(axis="y", labelcolor="r")
    if "has_clash" in results[0]:
        ax_clash = axes[1, 1].twinx()
        ax_clash.plot(steps, [r.get("has_clash", 0) for r in results], "k--", linewidth=0.8, label="Clash")
        ax_clash.set_ylabel("Clash", color="k")
    axes[1, 1].set_title("PAE & Clash")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_plot.png"), dpi=150)
    plt.close()
