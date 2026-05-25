"""CATE distribution plots — histogram + caterpillar (Day 4)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


def plot_cate_histogram(
    cate: np.ndarray,
    ate: float,
    out_path: Path,
    title: str = "CATE distribution — NSW participants",
) -> None:
    """Render a CATE histogram with ATE + zero-effect reference lines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cate, bins=40, edgecolor="black", alpha=0.7)
    ax.axvline(ate, color="red", linestyle="--", linewidth=2, label=f"ATE = ${ate:.0f}")
    ax.axvline(
        0, color="gray", linestyle=":", linewidth=1, label="No effect (CATE = 0)"
    )
    ax.set_xlabel("Predicted treatment effect on 1978 earnings (USD)")
    ax.set_ylabel("Number of individuals")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_cate_with_ci_sorted(
    cate: np.ndarray,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    out_path: Path,
) -> None:
    """Sorted CATE values with 95% bootstrap CIs — a 'caterpillar' plot."""
    order = np.argsort(cate)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    ax.fill_between(
        x, ci_lo[order], ci_hi[order], alpha=0.3, label="95% CI (bootstrap)"
    )
    ax.plot(x, cate[order], color="black", linewidth=1, label="CATE")
    ax.axhline(0, color="red", linestyle="--", linewidth=1, label="Zero effect")
    ax.set_xlabel("NSW individuals (sorted by CATE)")
    ax.set_ylabel("CATE (USD)")
    ax.set_title("Heterogeneity: who benefits, who doesn't?")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
