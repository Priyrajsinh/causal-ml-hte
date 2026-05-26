"""CATE heterogeneity analysis: quartile breakdowns, scatter plots, moderator scores."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.logger import get_logger

plt.switch_backend("Agg")
logger = get_logger(__name__)


def cate_by_quartile(cate_df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Bin a continuous covariate into quartiles and report mean CATE per bin."""
    df = cate_df.copy()
    # Discover actual number of bins after duplicate-edge removal before labelling
    _, edges = pd.qcut(df[by], q=4, retbins=True, duplicates="drop")
    n_bins = len(edges) - 1
    labels = [f"Q{i + 1}" for i in range(n_bins)]
    df["bin"], _ = pd.qcut(df[by], q=4, retbins=True, duplicates="drop", labels=labels)
    summary = (
        df.groupby("bin", observed=False)
        .agg(
            n=("cate", "size"),
            cate_mean=("cate", "mean"),
            cate_std=("cate", "std"),
            cate_p10=("cate", lambda s: float(s.quantile(0.10))),
            cate_p90=("cate", lambda s: float(s.quantile(0.90))),
            mean_x=(by, "mean"),
        )
        .reset_index()
    )
    return summary


def cate_by_category(cate_df: pd.DataFrame, by: str) -> pd.DataFrame:
    """For binary/categorical covariates, report mean CATE per group."""
    return (
        cate_df.groupby(by, observed=False)
        .agg(
            n=("cate", "size"),
            cate_mean=("cate", "mean"),
            cate_std=("cate", "std"),
        )
        .reset_index()
    )


def plot_cate_by_quartile(summary: pd.DataFrame, by: str, out_path: Path) -> None:
    """Bar chart of mean CATE per quartile with p10/p90 error bars."""
    fig, ax = plt.subplots(figsize=(7, 5))
    yerr_rows = summary.apply(
        lambda r: (r["cate_mean"] - r["cate_p10"], r["cate_p90"] - r["cate_mean"]),
        axis=1,
    ).tolist()
    yerr = np.array(yerr_rows).T
    ax.bar(
        summary["bin"].astype(str),
        summary["cate_mean"],
        yerr=yerr,
        capsize=4,
        color="#6366f1",
        edgecolor="black",
        alpha=0.8,
    )
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel(f"{by} quartile")
    ax.set_ylabel("Mean CATE (USD)")
    ax.set_title(f"Treatment effect by {by} quartile")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Saved quartile plot to %s", out_path)


def plot_cate_scatter(cate_df: pd.DataFrame, by: str, out_path: Path) -> None:
    """Scatter plot of CATE vs a covariate with rolling-mean overlay."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(cate_df[by], cate_df["cate"], alpha=0.5, color="#6366f1", s=15)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    order = np.argsort(cate_df[by].values)
    window = max(5, len(cate_df) // 30)
    smoothed = (
        pd.Series(cate_df["cate"].values[order])
        .rolling(window=window, center=True)
        .mean()
    )
    ax.plot(
        cate_df[by].values[order],
        smoothed.values,
        color="red",
        linewidth=2,
        label="rolling mean",
    )
    ax.set_xlabel(by)
    ax.set_ylabel("CATE (USD)")
    ax.set_title(f"Treatment effect vs {by}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Saved scatter plot to %s", out_path)


def moderator_scores(cate_df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Rank covariates by CATE range across quartile bins.

    Bigger range = more heterogeneity along that covariate axis.
    """
    rows = []
    for c in covariates:
        if cate_df[c].nunique() <= 5:
            grp = cate_df.groupby(c, observed=False)["cate"].mean()
        else:
            grp = (
                cate_df.assign(_bin=pd.qcut(cate_df[c], q=4, duplicates="drop"))
                .groupby("_bin", observed=False)["cate"]
                .mean()
            )
        rows.append(
            {
                "covariate": c,
                "cate_range": float(grp.max() - grp.min()),
                "cate_min": float(grp.min()),
                "cate_max": float(grp.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("cate_range", ascending=False)
