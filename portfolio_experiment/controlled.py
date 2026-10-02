from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ControlledConfig
from .dgp import covariance_matrix, draw_returns
from .estimators import estimate_all
from .metrics import relative_frobenius_error, true_volatility, weight_error
from .optimization import gmv_weights_unconstrained


def run_controlled(config: ControlledConfig, output_dir: Path) -> pd.DataFrame:
    """Initialisation"""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, float | int | str]] = []
    sim_data: dict = {
        "scenarios": {},
        "replications": {}
    }

    for structure in config.covariance_structures: # identity, one_factor
        for n_assets in config.dimensions: # 20, 50, 100
            scenario_key = (
                structure,
                n_assets
            )
            """Determine Oracle"""
            sigma_true = covariance_matrix(structure, n_assets, config.factor_strength) 
            oracle = gmv_weights_unconstrained(sigma_true, config.pinv_rcond)
            oracle_volatility = true_volatility(oracle, sigma_true)
            sim_data["scenarios"][scenario_key] = {
                "sigma_true": sigma_true.copy(),
                "oracle_weights": oracle.copy(),
                "oracle_volatility": oracle_volatility
            }
            for ratio in config.information_ratios:
                n_observations = int(round(n_assets / ratio))
                actual_ratio = n_assets / n_observations # since T is rounded
                """Monte Carlo Replications"""
                for replication in range(config.replications):
                    returns = draw_returns(rng, n_observations, sigma_true)
                    run_key = (
                        structure,
                        n_assets,
                        ratio,
                        replication
                    )
                    sim_data["replications"][run_key] = {
                        "estimators": {}
                    }
                    # Both estimators see the identical X^(r): paired design.
                    for estimator, (sigma_hat, shrinkage) in estimate_all(returns).items():
                        weights = gmv_weights_unconstrained(sigma_hat, config.pinv_rcond) # TODO(lookup) pinv_rcond
                        volatility = true_volatility(weights, sigma_true)
                        risk_ratio = volatility / oracle_volatility
                        sim_data["replications"][run_key]["estimators"][estimator] = {
                            "sigma_hat": sigma_hat.copy(),
                            "weights": weights.copy(),
                            "shrinkage": shrinkage,
                        }
                        rows.append(
                            {
                                "covariance_structure": structure,
                                "n_assets": n_assets,
                                "n_observations": n_observations,
                                "requested_information_ratio": ratio,
                                "information_ratio": actual_ratio,
                                "replication": replication,
                                "estimator": estimator,
                                "shrinkage": shrinkage,
                                "estimation_error": relative_frobenius_error(sigma_hat, sigma_true),
                                "decision_error": weight_error(weights, oracle),
                                "risk_ratio": risk_ratio,
                                "risk_increment": risk_ratio - 1.0,
                            }
                        )
                        print(f"Simulation: Structure {structure}, n_assets {n_assets}, Ratio {ratio} -> Rep {replication} / {config.replications} - no errors") # midrun info

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "replications.csv", index=False)
    _write_aggregates(results, output_dir)
    _write_plots(results, output_dir)
    return results, sim_data


def quick_config(config: ControlledConfig) -> ControlledConfig:
    return replace(config, dimensions=(10, 20), information_ratios=(0.25, 0.75, 1.10), replications=12)


def _write_aggregates(results: pd.DataFrame, output_dir: Path) -> None:
    group = [
        "covariance_structure",
        "n_assets",
        "n_observations",
        "information_ratio",
        "estimator"
        ]
    measures = [
        "estimation_error",
        "decision_error",
        "risk_increment"
        ]
    summary = results.groupby(group)[measures].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(part for part in col if part) for col in summary.columns.to_flat_index()]

    for measure in measures:
        mean_col = f"{measure}_mean"
        std_col = f"{measure}_std"
        count_col = f"{measure}_count"
        ste_col = f"{measure}_ste"
        ci_lower_col = f"{measure}_ci_lower"
        ci_upper_col = f"{measure}_ci_upper"

        summary[ste_col] = summary[std_col] / np.sqrt(summary[count_col])
        summary[ci_lower_col] = summary[mean_col] - 1.96 * summary[ste_col]
        summary[ci_upper_col] = summary[mean_col] + 1.96 * summary[ste_col]

    summary.to_csv(output_dir / "summary.csv", index=False)

    """PRIAL computation -> uses rel frobernius norm which is not the regular case -> currently not used -> tbd"""
    #losses = results.groupby(group)["estimation_error"].mean().unstack("estimator")
    #losses["prial"] = 1.0 - losses["ledoit_wolf"] / losses["sample"]
    #losses.reset_index().to_csv(output_dir / "prial.csv", index=False)


def _write_plots(results: pd.DataFrame, output_dir: Path) -> None:
    thesis_dir = output_dir / "Thesis_sources"
    thesis_dir.mkdir(parents=True, exist_ok=True)

    colors = {"sample": "#F38989", "ledoit_wolf": "#ACBAA1"}
    labels = {
        "estimation_error": "Relative covariance error",
        "decision_error": "Relative weight error",
        "risk_increment": "True risk increment (q - 1)",
    }
    group = [
        "covariance_structure",
        "n_assets",
        "requested_information_ratio",
        "estimator"
    ]
    measures = list(labels.keys())
    grouped = results.groupby(group)[measures].agg(["mean","std","count"]).reset_index()
    grouped.columns = ["_".join(part for part in col if part) for col in grouped.columns.to_flat_index()]

    for measure in measures:
        grouped[f"{measure}_ste"] = (
            grouped[f"{measure}_std"]
            / np.sqrt(grouped[f"{measure}_count"])
        )

        grouped[f"{measure}_ci"] = (
            1.96 * grouped[f"{measure}_ste"]
        )

    y_limits = {}

    for measure in measures:
        lower = grouped[f"{measure}_mean"] - grouped[f"{measure}_ci"]
        upper = grouped[f"{measure}_mean"] + grouped[f"{measure}_ci"]

        y_min = lower.min()
        y_max = upper.max()

        padding = 0.05 * (y_max - y_min)

        y_limits[measure] = (y_min - padding, y_max + padding)

    for n_assets, dimension_data in grouped.groupby("n_assets"):
        for measure, ylabel in labels.items():
            structures = sorted(dimension_data["covariance_structure"].unique())
            fig, axes = plt.subplots(1, len(structures), figsize=(5 * len(structures), 4), squeeze=False)
            for axis, structure in zip(axes[0], structures):
                part = dimension_data[dimension_data["covariance_structure"] == structure]
                for estimator, line in part.groupby("estimator"):
                    line = line.sort_values("requested_information_ratio")
                    x = line["requested_information_ratio"]
                    y = line[f"{measure}_mean"]
                    yerr = line[f"{measure}_ci"]
                    axis.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        marker="o",
                        capsize=3,
                        label=estimator.replace("_"," ").title(), 
                        color=colors[estimator]
                    )
                axis.axvline(1.0, color="0.4", linestyle="--", linewidth=1)
                axis.set(title=structure.replace("_", " ").title(), xlabel="Information ratio N/T", ylabel=ylabel)
                axis.set_ylim(y_limits[measure])
                axis.grid(alpha=0.2)
            axes[0, 0].legend(frameon=False)
            fig.tight_layout()
            fig.savefig(output_dir / f"{measure}_N{n_assets}.png", dpi=160)
            fig.savefig(thesis_dir / f"{measure}_N{n_assets}.pdf", bbox_inches="tight")
            plt.close(fig)
