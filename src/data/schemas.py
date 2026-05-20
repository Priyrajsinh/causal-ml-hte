"""Pydantic v2 request/response schemas for the CATE prediction API."""

from typing import Literal

from pydantic import BaseModel, Field


class LalondeProfile(BaseModel):
    """A single individual's 8 pre-treatment covariates."""

    age: int = Field(..., ge=17, le=55)
    education: int = Field(..., ge=0, le=18)
    black: Literal[0, 1]
    hispanic: Literal[0, 1]
    married: Literal[0, 1]
    nodegree: Literal[0, 1]
    re74: float = Field(..., ge=0.0)
    re75: float = Field(..., ge=0.0)


class CATEPrediction(BaseModel):
    """Model response for a single CATE prediction."""

    cate: float
    ci_lower: float
    ci_upper: float
    recommendation: Literal["TREAT", "DEFER"]
    nl_summary: str = ""


class ATEReport(BaseModel):
    """ATE comparison table: OLS-CPS bias vs DML vs RCT ground truth."""

    ols_cps_ate: float
    ols_cps_ci: tuple[float, float]
    dml_cps_ate: float
    dml_cps_ci: tuple[float, float]
    dml_nsw_ate: float
    dml_nsw_ci: tuple[float, float]
    rcl_ground_truth_ate: float = 1794.0
    rcl_ground_truth_ci: tuple[float, float] = (550.0, 3038.0)
