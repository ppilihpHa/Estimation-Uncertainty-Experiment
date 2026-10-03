from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def gmv_weights_unconstrained(sigma: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Closed-form fully-invested GMV weights; short positions are allowed."""
    ones = np.ones(sigma.shape[0])
    inverse = np.linalg.pinv(sigma, rcond=rcond, hermitian=True) 
    numerator = inverse @ ones
    denominator = float(ones @ numerator)
    if not np.isfinite(denominator) or abs(denominator) < 1e-14:
        raise ValueError("GMV denominator is numerically zero.")
    return numerator / denominator


def gmv_weights_long_only(sigma: np.ndarray) -> np.ndarray:
    """Numerically solve fully-invested, long-only GMV."""
    n_assets = sigma.shape[0]
    start = np.full(n_assets, 1.0 / n_assets)

    result = minimize(
        fun=lambda w: float(w @ sigma @ w), 
        x0=start,
        method="SLSQP", # optimizer
        bounds=[(0.0, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"GMV optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()

