from __future__ import annotations

import numpy as np
import pandas as pd


def relative_frobenius_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimate - truth, ord="fro") / np.linalg.norm(truth, ord="fro"))


def weight_error(weights: np.ndarray, oracle_weights: np.ndarray) -> float:
    return float(np.linalg.norm(weights - oracle_weights, ord=2) / np.linalg.norm(oracle_weights, ord=2) )


def true_volatility(weights: np.ndarray, sigma_true: np.ndarray) -> float:
    variance = float(weights @ sigma_true @ weights)
    return float(np.sqrt(max(variance, 0.0))) 


def turnover(previous: np.ndarray | None, target: np.ndarray) -> float:
    if previous is None:
        return float("nan")
    return 0.5 * float(np.abs(target - previous).sum()) 

def weight_concentration(weights: np.ndarray):
    return float(np.sum(weights ** 2))

def effective_number_of_assets(weights: np.ndarray):
    concentration = weight_concentration(weights=weights)
    if concentration <= 0.0: return float("nan")
    return float(1.0 / concentration)

def performance_summary(returns: pd.Series, annualization: int = 252) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return {"mean_return": np.nan, "volatility": np.nan, "sharpe": np.nan, "final_wealth": np.nan}
    mean_return = float(clean.mean() * annualization)
    volatility = float(clean.std(ddof=1) * np.sqrt(annualization))
    sharpe = mean_return / volatility if volatility > 0 else np.nan
    return {
        "mean_return": mean_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "final_wealth": float((1.0 + clean).prod()),
    }

