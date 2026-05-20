"""Deterministic seed utility. Module-level side-effect-free (rule C11)."""


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) LightGBM."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import lightgbm as lgb

        lgb.early_stopping  # touch the module to confirm import
    except Exception:
        pass
