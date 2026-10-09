from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ControlledConfig:
    """Design choices for the controlled Monte-Carlo experiment."""

    dimensions: tuple[int, ...] = (20, 50, 100) 
    information_ratios: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90, 1.10) 
    covariance_structures: list[str, ...] = ("one_factor",)
    replications: int = 1000
    seed: int = 20260831 # reproducability

    factor_strength: float = 0.35

    # Pseudoinverse for some cases of N > T.
    pinv_rcond: float = 1e-10


@dataclass(frozen=True)
class RealisticConfig:
    """Design choices for the realistic rolling-window experiment."""

    portfolio_sizes: tuple[int, ...] = (10, 50, 100, 200) 
    lookback_days: int = 504 # two year estimation window
    holding_days: int = 21 # one month holding period
    annualization: int = 252
    selection_metric: str = "market_cap" # TODO(TBD)
    max_missing_fraction: float = 0.05
    transaction_cost_rate = 0.0015 # 15 bps, ref. Palomar that common range is 1 - 30
    seed: int = 20260831
    strategies: tuple[str, ...] = field(
        default=("equal_weight", "sample_gmv", "ledoit_wolf_gmv")
    )

