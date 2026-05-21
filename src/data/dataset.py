"""LaLonde NSW + CPS data loaders with SHA-256 checksum verification (C34, C41)."""

import hashlib
from pathlib import Path

import pandas as pd

from src.data.validation import validate_lalonde_df
from src.exceptions import ChecksumError, DataLoadError
from src.logger import get_logger

logger = get_logger(__name__)


def verify_checksum(path: Path) -> None:
    """Read .sha256 sidecar next to ``path`` and compare. Rule C41."""
    sidecar = path.parent / f"{path.name}.sha256"
    if not sidecar.exists():
        raise ChecksumError(f"No checksum sidecar for {path}")
    expected = sidecar.read_text().strip()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ChecksumError(
            f"SHA-256 mismatch for {path}: expected {expected[:8]}…, got {actual[:8]}…"
        )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop data_id, cast dtypes, run pandera (rule C34)."""
    if "data_id" in df.columns:
        df = df.drop(columns=["data_id"])
    int_cols = ["treat", "age", "education", "black", "hispanic", "married", "nodegree"]
    float_cols = ["re74", "re75", "re78"]
    for c in int_cols:
        df[c] = df[c].astype(int)
    for c in float_cols:
        df[c] = df[c].astype(float)
    return validate_lalonde_df(df.reset_index(drop=True))


def load_nsw(path: str | Path) -> pd.DataFrame:
    """Load + clean + validate NSW (LaLonde Dehejia–Wahba subset, 445 rows)."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"NSW CSV not found at {path}")
    verify_checksum(path)
    df = pd.read_csv(path)
    df = _clean(df)
    logger.info(
        "Loaded NSW: %d rows, treat=%d controls=%d",
        len(df),
        (df["treat"] == 1).sum(),
        (df["treat"] == 0).sum(),
    )
    return df


def load_cps(path: str | Path) -> pd.DataFrame:
    """Load + clean + validate CPS controls (observational, ~16K rows)."""
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"CPS CSV not found at {path}")
    verify_checksum(path)
    df = pd.read_csv(path)
    if "treat" not in df.columns:
        df["treat"] = 0
    df = _clean(df)
    logger.info("Loaded CPS: %d rows", len(df))
    return df


def build_cps_observational(
    nsw_treated: pd.DataFrame, cps: pd.DataFrame
) -> pd.DataFrame:
    """Concatenate NSW *treated* rows with all CPS controls.

    This is the construction LaLonde (1986) used to demonstrate selection bias:
    replacing the RCT controls with CPS observational controls collapses or
    flips the apparent treatment effect — the bias we recover with DML (Day 3).
    """
    treated = nsw_treated[nsw_treated["treat"] == 1].copy()
    out = pd.concat([treated, cps.assign(treat=0)], ignore_index=True)
    return validate_lalonde_df(out)
