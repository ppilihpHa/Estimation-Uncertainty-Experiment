from __future__ import annotations

import numpy as np


def covariance_matrix(
    name: str,
    n_assets: int,
    factor_strength: float = 0.35,
) -> np.ndarray:
    """Construct a positive-definite population covariance matrix."""
    if name == "identity":
        return np.eye(n_assets)

    if name == "one_factor":
        loadings = np.linspace(0.5, 1.5, n_assets) 
        common = factor_strength * np.outer(loadings, loadings)
        sigma = common + (1.0 - factor_strength) * np.eye(n_assets)
        return sigma

    raise ValueError(f"Unknown covariance structure: {name}")


def draw_returns(
    rng: np.random.Generator,
    n_observations: int,
    sigma_true: np.ndarray,
) -> np.ndarray:
    """Draw zero-mean Gaussian returns for one paired replication."""
    return rng.multivariate_normal(
        mean=np.zeros(sigma_true.shape[0]),
        cov=sigma_true,
        size=n_observations,
        method="cholesky",
    )

