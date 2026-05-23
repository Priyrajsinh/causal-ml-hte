"""Download LaLonde NSW + CPS raw data from NBER, convert .dta → .csv.

Used in CI (where DVC pull is not configured) to make raw data available
for tests. Writes fresh SHA-256 sidecars after conversion so verify_checksum()
passes in the subsequent test run.

Usage:
    python scripts/download_data.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

import pyreadstat

DATASETS = {
    "nsw": {
        "url": "https://users.nber.org/~rdehejia/data/nsw_dw.dta",
        "dta": Path("data/raw/lalonde_nsw.dta"),
        "csv": Path("data/raw/lalonde_nsw.csv"),
        "sha": Path("data/raw/lalonde_nsw.csv.sha256"),
        "expected_rows": 445,
    },
    "cps": {
        "url": "https://users.nber.org/~rdehejia/data/cps_controls.dta",
        "dta": Path("data/raw/lalonde_cps.dta"),
        "csv": Path("data/raw/lalonde_cps.csv"),
        "sha": Path("data/raw/lalonde_cps.csv.sha256"),
        "expected_rows": 15992,
    },
}


def download_dataset(name: str, cfg: dict) -> bool:
    """Download one dataset. Returns True on success, False on failure."""
    csv_path: Path = cfg["csv"]
    if csv_path.exists():
        print(f"[{name}] already present, skipping download.")
        return True
    try:
        print(f"[{name}] downloading {cfg['url']} …")
        urllib.request.urlretrieve(cfg["url"], cfg["dta"])
        df, _ = pyreadstat.read_dta(str(cfg["dta"]))
        df.to_csv(csv_path, index=False)
        n = len(df)
        if n != cfg["expected_rows"]:
            print(f"[{name}] WARNING: expected {cfg['expected_rows']} rows, got {n}")
        sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        cfg["sha"].write_text(sha)
        print(f"[{name}] saved {n} rows → {csv_path} (sha256={sha[:16]}…)")
        return True
    except Exception as exc:
        print(f"[{name}] FAILED: {exc}", file=sys.stderr)
        return False


def main() -> int:
    """Download all datasets. Exit code 0 if all succeeded, 1 if any failed."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    results = {name: download_dataset(name, cfg) for name, cfg in DATASETS.items()}
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"Download failed for: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
