"""Headline three-way ATE comparison figure (OLS-bias vs DML-recovery vs RCT).

Reads ``reports/results.json`` and draws a bar chart with 95% CIs plus a
horizontal RCT reference line. This is the figure that lands at the top of the
README on Day 8 — keep it legible.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


BAR_SPECS = [
    ("ols_cps_unadjusted", "OLS-CPS\n(unadjusted)", "#ef4444"),
    ("ols_cps_adjusted", "OLS-CPS\n(adjusted)", "#f97316"),
    ("dml_cps", "DML-CPS", "#22c55e"),
    ("dml_nsw", "DML-NSW", "#6366f1"),
]


def plot_three_way_table(results_json: Path, out_png: Path) -> Path:
    """Render the four-bar ATE comparison figure with RCT reference line."""
    data = json.loads(Path(results_json).read_text())
    table = data.get("ate_table") or {}
    truth = data["rcl_ground_truth"]["ate"]
    truth_lo = data["rcl_ground_truth"]["ci_lower"]
    truth_hi = data["rcl_ground_truth"]["ci_upper"]

    labels: list[str] = []
    ates: list[float] = []
    err_lo: list[float] = []
    err_hi: list[float] = []
    colors: list[str] = []
    for key, label, color in BAR_SPECS:
        entry = table.get(key) or data.get(key)
        if entry is None:
            continue
        ate = float(entry["ate"])
        labels.append(label)
        ates.append(ate)
        err_lo.append(ate - float(entry["ci_lower"]))
        err_hi.append(float(entry["ci_upper"]) - ate)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    xpos = np.arange(len(labels))
    ax.bar(xpos, ates, color=colors, edgecolor="black", linewidth=0.6, zorder=2)
    ax.errorbar(
        xpos,
        ates,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.2,
        zorder=3,
    )
    ax.axhline(
        truth,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
        label=f"RCT truth = ${truth:,.0f}",
        zorder=1,
    )
    ax.axhspan(truth_lo, truth_hi, color="#111827", alpha=0.07, zorder=0)
    ax.axhline(0, color="grey", linewidth=0.6, zorder=1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ATE on 1978 earnings (USD)")
    ax.set_title(
        "OLS bias vs DML recovery vs RCT ground truth\n"
        "LaLonde NSW + CPS — 95% CIs as error bars"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    fig.tight_layout()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("Wrote three-way ATE figure to %s", out_png)
    return out_png


if __name__ == "__main__":
    plot_three_way_table(
        Path("reports/results.json"),
        Path("reports/figures/ate_three_way_table.png"),
    )
