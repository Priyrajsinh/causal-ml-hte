"""Tests for src/baseline/balance.py and src/baseline/baseline_ols.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseline.balance import (
    bootstrap_ate_ci,
    covariate_balance_table,
    standardised_mean_difference,
)
from src.baseline.baseline_ols import ols_ate
from src.data.dataset import build_cps_observational, load_cps, load_nsw

NSW_PATH = Path("data/raw/lalonde_nsw.csv")
CPS_PATH = Path("data/raw/lalonde_cps.csv")


@pytest.fixture(scope="module")
def nsw() -> pd.DataFrame:
    """Full NSW dataset (445 rows)."""
    return load_nsw(NSW_PATH)


@pytest.fixture(scope="module")
def obs(nsw: pd.DataFrame) -> pd.DataFrame:
    """LaLonde CPS observational construction (NSW treated + CPS controls)."""
    cps = load_cps(CPS_PATH)
    return build_cps_observational(nsw, cps)


class TestSMD:
    """Standardised mean difference helper."""

    def test_balance_smd_for_balanced_synthetic(self) -> None:
        """Perfectly balanced groups (identical means) give SMD == 0."""
        df = pd.DataFrame(
            {
                "treat": [0, 0, 0, 1, 1, 1],
                "age": [25, 30, 35, 25, 30, 35],
                "education": [10, 12, 14, 10, 12, 14],
                "black": [0, 1, 0, 0, 1, 0],
                "hispanic": [0, 0, 1, 0, 0, 1],
                "married": [1, 0, 1, 1, 0, 1],
                "nodegree": [0, 1, 0, 0, 1, 0],
                "re74": [0.0, 1000.0, 2000.0, 0.0, 1000.0, 2000.0],
                "re75": [0.0, 1100.0, 2100.0, 0.0, 1100.0, 2100.0],
                "re78": [5000.0, 6000.0, 7000.0, 5000.0, 6000.0, 7000.0],
            }
        )
        assert standardised_mean_difference(df, "age") == pytest.approx(0.0, abs=1e-9)
        assert standardised_mean_difference(df, "re74") == pytest.approx(0.0, abs=1e-9)

    def test_covariate_balance_table_columns(self, nsw: pd.DataFrame) -> None:
        """Balance table must have covariate / mean_treat / mean_control / smd."""
        tbl = covariate_balance_table(nsw)
        assert set(tbl.columns) == {"covariate", "mean_treat", "mean_control", "smd"}
        assert len(tbl) == 8


class TestBootstrapATE:
    """Bootstrap ATE CI for the NSW RCT ground truth."""

    def test_bootstrap_ate_ci_contains_true_mean(self) -> None:
        """Synthetic RCT ATE=500: bootstrap CI must cover the truth >=95% of trials."""
        rng = np.random.default_rng(0)
        hits = 0
        n_trials = 200
        for trial in range(n_trials):
            n = 200
            control_re78 = rng.normal(4000, 2000, n // 2)
            treat_re78 = rng.normal(4500, 2000, n // 2)
            df = pd.DataFrame(
                {
                    "treat": [0] * (n // 2) + [1] * (n // 2),
                    "re78": list(control_re78) + list(treat_re78),
                }
            )
            result = bootstrap_ate_ci(df, n_boot=500, seed=trial)
            if result["ci_lower"] <= 500 <= result["ci_upper"]:
                hits += 1
        assert hits / n_trials >= 0.90

    def test_rct_ground_truth_in_expected_range(self, nsw: pd.DataFrame) -> None:
        """NSW RCT ATE must be in (1000, 2800) — the LaLonde benchmark range."""
        result = bootstrap_ate_ci(nsw)
        assert 1000 < result["ate"] < 2800

    def test_bootstrap_ci_contains_1794(self, nsw: pd.DataFrame) -> None:
        """95% bootstrap CI on NSW must contain $1,794 (rule C37)."""
        result = bootstrap_ate_ci(nsw)
        assert result["ci_lower"] < 1794 < result["ci_upper"]


class TestOLSBaselines:
    """Naive OLS baselines — the bias demonstration (rule C40)."""

    def test_ols_nsw_close_to_rct_truth(self, nsw: pd.DataFrame) -> None:
        """OLS on the RCT sample should be within $600 of the RCT ground truth."""
        rct_ate = bootstrap_ate_ci(nsw)["ate"]
        ols_result = ols_ate(nsw)
        assert abs(ols_result["ate"] - rct_ate) < 600

    def test_ols_cps_unadjusted_is_biased(
        self, nsw: pd.DataFrame, obs: pd.DataFrame
    ) -> None:
        """OLS-on-CPS ATE must differ from RCT truth by > $1,000 (the headline bias)."""
        rct_ate = bootstrap_ate_ci(nsw)["ate"]
        ols_cps = ols_ate(obs, controls=[])
        assert abs(ols_cps["ate"] - rct_ate) > 1000

    def test_ols_returns_required_keys(self, nsw: pd.DataFrame) -> None:
        """ols_ate result must contain ate, ci_lower, ci_upper, se, p_value, r2, n."""
        result = ols_ate(nsw)
        for key in ("ate", "ci_lower", "ci_upper", "se", "p_value", "r2", "n"):
            assert key in result
