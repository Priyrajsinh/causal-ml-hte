"""Naive OLS baselines: NSW (approximately unbiased) vs CPS-observational (biased).

Rule C40: NEVER present naive OLS as the causal estimate. The OLS-on-CPS bias
*is* the scientific narrative — it motivates why DML is needed.
"""

import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from src.logger import get_logger

logger = get_logger(__name__)

COVARIATES = [
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]


def ols_ate(
    df: pd.DataFrame, controls: list[str] | None = None
) -> dict[str, float | int]:
    """Estimate ATE via OLS: re78 ~ treat + controls.

    On NSW (random assignment) this is approximately unbiased.
    On the LaLonde-CPS observational construction this is biased — often
    producing a negative ATE that looks like job training *hurts* earnings.
    Rule C40: this function exists to DEMONSTRATE the bias, not to use OLS causally.
    """
    controls = controls if controls is not None else COVARIATES
    X = sm.add_constant(df[["treat"] + controls].astype(float))
    y = df["re78"].astype(float)
    model = sm.OLS(y, X).fit()
    ci = model.conf_int().loc["treat"]
    return {
        "ate": float(model.params["treat"]),
        "ci_lower": float(ci.iloc[0]),
        "ci_upper": float(ci.iloc[1]),
        "se": float(model.bse["treat"]),
        "p_value": float(model.pvalues["treat"]),
        "r2": float(model.rsquared),
        "n": int(len(df)),
    }


def save_ols_baselines(
    nsw: pd.DataFrame, cps_obs: pd.DataFrame, out_path: Path
) -> dict[str, dict[str, float | int]]:
    """Compute OLS ATE on NSW (unbiased) and CPS-observational (biased).

    Persists results to reports/results.json under keys:
    - ``ols_nsw`` — OLS on the RCT sample (approximately unbiased baseline)
    - ``ols_cps_unadjusted`` — raw OLS on CPS observational (the headline bias)
    - ``ols_cps_adjusted`` — OLS + covariate controls on CPS (still biased)
    """
    ols_nsw = ols_ate(nsw)
    ols_cps_unadjusted = ols_ate(cps_obs, controls=[])
    ols_cps_adjusted = ols_ate(cps_obs)
    result: dict[str, dict[str, float | int]] = {
        "ols_nsw": ols_nsw,
        "ols_cps_unadjusted": ols_cps_unadjusted,
        "ols_cps_adjusted": ols_cps_adjusted,
    }
    existing: dict = (
        json.loads(Path(out_path).read_text()) if Path(out_path).exists() else {}
    )
    existing.update(result)
    with open(str(out_path), "w") as fh:
        json.dump(existing, fh, indent=2)
    logger.info("OLS-NSW ATE: $%.0f", ols_nsw["ate"])
    logger.info(
        "OLS-CPS (unadjusted) ATE: $%.0f  <- the headline bias number",
        ols_cps_unadjusted["ate"],
    )
    logger.info("OLS-CPS (adjusted) ATE: $%.0f", ols_cps_adjusted["ate"])
    return result
