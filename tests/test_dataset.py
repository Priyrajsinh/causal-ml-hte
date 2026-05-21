"""Tests for src/data/dataset.py (rules C34, C41)."""

import hashlib
import textwrap
from pathlib import Path

import pytest

from src.data.dataset import (
    build_cps_observational,
    load_cps,
    load_nsw,
    verify_checksum,
)
from src.exceptions import ChecksumError, DataLoadError

NSW_PATH = Path("data/raw/lalonde_nsw.csv")
CPS_PATH = Path("data/raw/lalonde_cps.csv")


class TestLoadNsw:
    """Load + validate NSW CSV."""

    def test_load_nsw_returns_445_rows(self) -> None:
        """NSW Dehejia–Wahba subset must have exactly 445 rows after cleaning."""
        df = load_nsw(NSW_PATH)
        assert len(df) == 445

    def test_load_nsw_columns(self) -> None:
        """Cleaned NSW must have the 10 schema columns (no data_id)."""
        df = load_nsw(NSW_PATH)
        expected = {
            "treat",
            "age",
            "education",
            "black",
            "hispanic",
            "married",
            "nodegree",
            "re74",
            "re75",
            "re78",
        }
        assert set(df.columns) == expected

    def test_load_nsw_missing_file_raises(self) -> None:
        """DataLoadError on nonexistent path."""
        with pytest.raises(DataLoadError):
            load_nsw(Path("data/raw/does_not_exist.csv"))


class TestChecksum:
    """SHA-256 sidecar verification (rule C41)."""

    def test_checksum_mismatch_raises(self, tmp_path: Path) -> None:
        """Tampered sidecar triggers ChecksumError."""
        csv = tmp_path / "dummy.csv"
        csv.write_text(
            "treat,age,education,black,hispanic,married,nodegree,re74,re75,re78\n"
        )
        sidecar = tmp_path / "dummy.csv.sha256"
        sidecar.write_text("deadbeef" * 8)  # wrong hash
        with pytest.raises(ChecksumError):
            verify_checksum(csv)

    def test_missing_sidecar_raises(self, tmp_path: Path) -> None:
        """Missing sidecar triggers ChecksumError."""
        csv = tmp_path / "dummy.csv"
        csv.write_text("data")
        with pytest.raises(ChecksumError):
            verify_checksum(csv)

    def test_correct_checksum_passes(self, tmp_path: Path) -> None:
        """Matching SHA-256 sidecar passes silently."""
        csv = tmp_path / "dummy.csv"
        csv.write_bytes(b"some content")
        sidecar = tmp_path / "dummy.csv.sha256"
        sidecar.write_text(hashlib.sha256(b"some content").hexdigest())
        verify_checksum(csv)  # must not raise


class TestPanderaRejection:
    """Pandera schema enforcement (rule C34)."""

    def test_pandera_rejects_bad_treat_value(self, tmp_path: Path) -> None:
        """treat=2 must fail pandera validation via load_nsw."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text(
            textwrap.dedent(
                """\
                treat,age,education,black,hispanic,married,nodegree,re74,re75,re78
                2,25,10,1,0,0,1,0.0,0.0,5000.0
            """
            )
        )
        sidecar = tmp_path / "bad.csv.sha256"
        sidecar.write_text(hashlib.sha256(bad_csv.read_bytes()).hexdigest())
        with pytest.raises(Exception):
            load_nsw(bad_csv)


class TestBuildCpsObservational:
    """build_cps_observational: LaLonde (1986) selection-bias construction."""

    def test_has_both_treatment_labels(self) -> None:
        """Combined dataset must contain treated (1) and control (0) rows."""
        nsw = load_nsw(NSW_PATH)
        cps = load_cps(CPS_PATH)
        obs = build_cps_observational(nsw, cps)
        assert set(obs["treat"].unique()) == {0, 1}

    def test_treated_rows_come_from_nsw(self) -> None:
        """All treat==1 rows in the observational dataset originate from NSW."""
        nsw = load_nsw(NSW_PATH)
        cps = load_cps(CPS_PATH)
        obs = build_cps_observational(nsw, cps)
        n_treated = (obs["treat"] == 1).sum()
        assert n_treated == (nsw["treat"] == 1).sum()

    def test_control_count_equals_cps_rows(self) -> None:
        """Control rows must equal the full CPS count (all CPS are controls)."""
        nsw = load_nsw(NSW_PATH)
        cps = load_cps(CPS_PATH)
        obs = build_cps_observational(nsw, cps)
        assert (obs["treat"] == 0).sum() == len(cps)
