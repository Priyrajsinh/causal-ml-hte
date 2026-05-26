"""Policy-targeting analysis: top-N% by CATE, qini-style policy curve."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

_DEFAULT_COVARIATES = [
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]


def top_fraction_targeting(
    cate_df: pd.DataFrame,
    fraction: float = 0.30,
    covariates: list[str] | None = None,
) -> dict[str, object]:
    """Identify the top-`fraction` of individuals by predicted CATE.

    Returns a dict with n_targeted, mean CATEs, total lift, and profile deltas.
    """
    cov = covariates or _DEFAULT_COVARIATES
    n = len(cate_df)
    k = max(1, int(round(n * fraction)))
    sorted_df = cate_df.sort_values("cate", ascending=False).reset_index(drop=True)
    targeted = sorted_df.iloc[:k]
    untargeted = sorted_df.iloc[k:]
    profile: dict[str, float] = targeted[cov].mean().to_dict()
    pop_profile: dict[str, float] = cate_df[cov].mean().to_dict()
    return {
        "fraction": float(fraction),
        "n_targeted": int(k),
        "n_total": int(n),
        "mean_cate_targeted": float(targeted["cate"].mean()),
        "mean_cate_untargeted": float(untargeted["cate"].mean()),
        "estimated_total_lift_usd": float(targeted["cate"].sum()),
        "estimated_avg_lift_usd": float(targeted["cate"].mean()),
        "profile": {key: float(val) for key, val in profile.items()},
        "profile_vs_population": {
            key: float(profile[key] - pop_profile[key]) for key in cov
        },
    }


def plot_policy_curve(cate_df: pd.DataFrame, out_path: Path) -> None:
    """Qini-style curve: total realised CATE vs fraction treated."""
    sorted_cate = cate_df["cate"].sort_values(ascending=False).values
    cumulative = np.cumsum(sorted_cate)
    fractions = np.arange(1, len(sorted_cate) + 1) / len(sorted_cate)
    random_baseline = fractions * float(sorted_cate.sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        fractions,
        cumulative,
        color="#6366f1",
        linewidth=2,
        label="Targeted (rank by predicted CATE)",
    )
    ax.plot(
        fractions,
        random_baseline,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Random selection",
    )
    ax.fill_between(
        fractions,
        cumulative,
        random_baseline,
        color="#6366f1",
        alpha=0.2,
        label="Uplift over random",
    )
    ax.set_xlabel("Fraction of population treated")
    ax.set_ylabel("Total realised earnings lift (USD, summed)")
    ax.set_title("Policy curve — targeting by predicted CATE")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_targeted_profile(profile_vs_pop: dict[str, float], out_path: Path) -> None:
    """Bar chart of how the targeted group differs from the population average."""
    df = pd.DataFrame(
        {
            "covariate": list(profile_vs_pop.keys()),
            "diff": list(profile_vs_pop.values()),
        }
    )
    df["abs"] = df["diff"].abs()
    df = df.sort_values("abs", ascending=True)
    colors = ["#ef4444" if d < 0 else "#22c55e" for d in df["diff"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["covariate"], df["diff"], color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Targeted group mean − Population mean")
    ax.set_title("Who would be targeted? (deviation from average)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
