from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance with variables in columns."""
    return np.cov(returns, rowvar=False, ddof=1)


def ledoit_wolf_covariance(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf covariance estimate and fitted shrinkage intensity."""
    fit = LedoitWolf(assume_centered=False).fit(returns) # dive in 
    return fit.covariance_, float(fit.shrinkage_)


def estimate_all(returns: np.ndarray) -> dict[str, tuple[np.ndarray, float]]:
    """Estimate both matrices from the same draw (paired comparison)."""
    lw_cov, shrinkage = ledoit_wolf_covariance(returns)
    return {
        "sample": (sample_covariance(returns), np.nan),
        "ledoit_wolf": (lw_cov, shrinkage),
    }

