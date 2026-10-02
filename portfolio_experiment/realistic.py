from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RealisticConfig
from .estimators import ledoit_wolf_covariance, sample_covariance
from .metrics import performance_summary, turnover, weight_concentration, effective_number_of_assets
from .optimization import gmv_weights_long_only


def run_realistic(panel: pd.DataFrame, config: RealisticConfig, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run long-only rolling-window backtests for the configured sizes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = np.sort(panel["Date"].dropna().unique())
    if len(dates) <= config.lookback_days:
        raise ValueError("The panel is shorter than the configured lookback window.")
    return_rows: list[dict[str, object]] = []
    rebalance_rows: list[dict[str, object]] = []
    previous: dict[tuple[int, str], pd.Series] = {}

    diagnostic_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    availability_rows : list[dict[str, object]] = []
    sel_availability_rows : list[dict[str, object]] = []
    skipped_rows : list[dict[str, object]] = []

    for index in range(config.lookback_days, len(dates), config.holding_days):
        print(f"{index} / {len(dates)}")
        estimation_dates = dates[index - config.lookback_days:index]
        holding_dates = dates[index:min(index + config.holding_days, len(dates))]
        estimation = panel[panel["Date"].isin(estimation_dates)]
        holding = panel[panel["Date"].isin(holding_dates)]
        estimation_returns = estimation.pivot(index="Date", columns="DSCD", values="Return")
        availability = estimation_returns.notna().mean()

        availability_rows.append({
            "rebalancing_date": pd.Timestamp(holding_dates[0]),
            "n_assets_total": len(availability),
            "n_avail_100": int((availability >= 1.00).sum()),
            "n_avail_99": int((availability >= 0.99).sum()),
            "n_avail_95": int((availability >= 0.95).sum()),
            "n_avail_90": int((availability >= 0.90).sum()),
            "median_availability": float(availability.median()),
            "min_availability": float(availability.min()),
        })

        eligible = availability[availability >= 1.0 - config.max_missing_fraction].index

        for n_assets in config.portfolio_sizes:
            selected = _select_assets(estimation, eligible, n_assets, config.selection_metric, False)

            top_n_unfiltered = _select_assets(estimation, eligible, n_assets, config.selection_metric, True)
            top_n_availability = availability.reindex(top_n_unfiltered)
            sel_availability_rows.append({
                "rebalancing_date": pd.Timestamp(holding_dates[0]),
                "n_assets": n_assets,
                "top_n_missing_any": int((top_n_availability < 1.0).sum()),
                "top_n_below_99": int((top_n_availability < 0.99).sum()),
                "top_n_below_95": int((top_n_availability < 0.95).sum()),
                "mean_top_n_availability": float(top_n_availability.mean()),
                "min_top_n_availability": float(top_n_availability.min()),
            })

            if len(selected) < n_assets:
                skipped_rows.append({
                    "rebalancing_date": pd.Timestamp(holding_dates[0]),
                    "n_assets": n_assets,
                    "reason": "insufficient_eligible_assets",
                    "n_eligible": len(eligible),
                    "n_required": n_assets,
                })
                continue
            train = estimation_returns.loc[:, selected].dropna()
            test = holding.pivot(index="Date", columns="DSCD", values="Return").reindex(columns=selected)

            """Diagnostics"""
            zero_fraction_by_asset = (train == 0.0).mean()
            asset_vol_daily = train.std(ddof=1)
            asset_vol_annual = asset_vol_daily * np.sqrt(config.annualization)
            pseudo_sharpe = (train.mean() / train.std(ddof=1))
            n_missing_test = int(test.isna().sum().sum())
            missing_faction_test = float(test.isna().mean().mean())
            rebalance_date = pd.Timestamp(holding_dates[0])
            diagnostic_rows.append({
                "rebalance_date": rebalance_date,
                "n_assets": n_assets,
                "n_observations": train.shape[0],
                "n_eligible": len(eligible),
                "n_missing_test": n_missing_test,
                "missing_fraction_test": missing_faction_test,
                "mean_zero_fraction": float(zero_fraction_by_asset.mean()),
                "max_zero_fraction": float(zero_fraction_by_asset.max()),
                "min_asset_vol_annual": float(asset_vol_annual.min()),
                "median_asset_vol_annual": float(asset_vol_annual.median()),
                "max_asset_vol_annual": float(asset_vol_annual.max()),
                "mean_pseudo_sharpe": float(pseudo_sharpe.mean()),
                "max_pseudo_sharpe": float(pseudo_sharpe.max())
            })
            missing = test.isna()
            if missing.any().any():
                missing_locations = missing.stack()
                missing_locations = missing_locations[missing_locations]
                for date, asset in missing_locations.index:
                    missing_rows.append({
                        "rebalancing_date": rebalance_date,
                        "n_assets": n_assets,
                        "date": pd.Timestamp(date),
                        "DSCD": asset
                    })
                #continue
                #test = test.dropna(axis=0, how="any")
                test = test.fillna(0.0)
                print("called test.fillna(0.0)")
            if train.shape[0] < 3 or test.empty:
                continue
            sample = sample_covariance(train.to_numpy())
            lw, _ = ledoit_wolf_covariance(train.to_numpy())
            candidates = {
                "equal_weight": np.full(n_assets, 1.0 / n_assets),
                "sample_gmv": gmv_weights_long_only(sample),
                "ledoit_wolf_gmv": gmv_weights_long_only(lw),
            }
            for strategy in config.strategies:
                target_weights = pd.Series(candidates[strategy], index=selected)
                old = previous.get((n_assets, strategy))
                if old is None:
                    current_turnover = turnover(None, target_weights.to_numpy())
                    transaction_cost = 0 # initial portfolio formation cost are ignored
                else:
                    union = old.index.union(target_weights.index)
                    current_turnover = turnover(
                        old.reindex(union, fill_value=0.0).to_numpy(),
                        target_weights.reindex(union, fill_value=0.0).to_numpy(),
                    )
                    transaction_cost = config.transaction_cost_rate * current_turnover
                """concentration"""
                target_array = target_weights.to_numpy()
                concentration = weight_concentration(target_array)
                effective_n = effective_number_of_assets(target_array)
                max_weight = float(target_array.max())

                """"holding"""
                daily_gross, final_weights = _simulate_holding(target_weights.to_numpy(), test.to_numpy())
                daily_net = daily_gross.copy() 
                daily_net[0] -= transaction_cost

                previous[(n_assets, strategy)] = pd.Series(final_weights, index=selected)
                rebalance_rows.append({
                    "rebalance_date": pd.Timestamp(holding_dates[0]), 
                    "n_assets": n_assets,
                    "n_observations": train.shape[0],
                    "strategy": strategy, 
                    "turnover": current_turnover,
                    "transaction_cost": transaction_cost,
                    "concentration": concentration,
                    "effective_n": effective_n,
                    "max_weight": max_weight
                })
                
                assert len(test.index) == len(daily_net) == len(daily_gross) # so that I notice when something bugs since zip() just stops when lenghts don't match
                for date, value_net, value_gross in zip(test.index, daily_net, daily_gross):
                    return_rows.append({
                        "date": date,
                        "n_assets": n_assets, 
                        "strategy": strategy, 
                        "return": float(value_net),
                        "return_gross": value_gross
                    })

    returns = pd.DataFrame(return_rows)
    rebalances = pd.DataFrame(rebalance_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    missing_oos = pd.DataFrame(missing_rows)
    availability_total = pd.DataFrame(availability_rows)
    availability_selected = pd.DataFrame(sel_availability_rows)
    skipped = pd.DataFrame(skipped_rows)
    returns.to_csv(output_dir / "daily_returns.csv", index=False)
    rebalances.to_csv(output_dir / "rebalances.csv", index=False)
    diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    missing_oos.to_csv(output_dir / "missing_oos_returns.csv", index=False)
    availability_total.to_csv(output_dir / "availability_total.csv", index=False)
    availability_selected.to_csv(output_dir / "availability_selected.csv", index=False)
    skipped.to_csv(output_dir / "skipped.csv", index=False)
    summary = _write_summary(returns, rebalances, config, output_dir)
    _write_plots(returns, rebalances, summary, output_dir)
    return returns, rebalances, diagnostics

def _simulate_holding(initial_weights: np.ndarray, asset_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """introduce drifting weights"""
    weights = initial_weights.copy()
    pf_returns = []

    for daily_asset_returns in asset_returns:
        pf_return = float(weights @ daily_asset_returns)
        pf_returns.append(pf_return)
        weights = (weights * (1.0 + daily_asset_returns) / (1.0 + pf_return)) # actual weight when asset price change
    
    return np.asarray(pf_returns), weights

def _select_assets(estimation: pd.DataFrame, eligible: pd.Index, n_assets: int, metric: str, skip_eligibility: bool) -> list[object]:
    if not skip_eligibility:
        subset = estimation[estimation["DSCD"].isin(eligible)]
    else:
        subset = estimation.copy()

    if metric == "market_cap" and "MarketCAP" in subset.columns:
        scores = subset.sort_values("Date").groupby("DSCD")["MarketCAP"].last()
    elif metric == "pseudo_sharpe":
        matrix = subset.pivot(index="Date", columns="DSCD", values="Return")
        scores = matrix.mean() / matrix.std(ddof=1)
    else:
        raise ValueError(f"Unsupported selection metric: {metric}")
    return scores.nlargest(n_assets).index.tolist()


def _write_summary(returns: pd.DataFrame, rebalances: pd.DataFrame, config: RealisticConfig, output_dir: Path) -> pd.DataFrame:
    
    if returns.empty:
        return pd.DataFrame().to_csv(output_dir / "summary.csv", index=False)
    rows = []
    for (n_assets, strategy), part in returns.groupby(["n_assets", "strategy"]):
        row = {"n_assets": n_assets, "strategy": strategy}
        row.update(performance_summary(part.set_index("date")["return"], config.annualization))
        selected = rebalances[(rebalances["n_assets"] == n_assets) & (rebalances["strategy"] == strategy)]
        row["n_rebalances"] = len(selected)
        row["mean_turnover"] = selected["turnover"].mean()
        row["mean_transaction_cost"] = selected["transaction_cost"].mean()
        row["total_transaction_cost"] = selected["transaction_cost"].sum()
        row["mean_concentration"] = selected["concentration"].mean()
        row["mean_effective_n"] = selected["effective_n"].mean()
        row["mean_max_weight"] = selected["max_weight"].mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary

def _write_plots(returns: pd.DataFrame, rebalances: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    thesis_dir = output_dir / "Thesis_sources"
    thesis_dir.mkdir(parents=True, exist_ok=True)
    
    colors = {
        "equal_weight": "#A2AAB0",
        "sample_gmv": "#F38989", 
        "ledoit_wolf_gmv": "#ACBAA1"
        }
    labels = {
        "equal_weight": "Naive",
        "sample_gmv": "Sample GMV", 
        "ledoit_wolf_gmv": "L&W GMV"
    }
    strategies = list(labels.keys())

    fig, ax = plt.subplots(figsize=(7,4))

    for strategy in strategies:
        part = summary[summary["strategy"] == strategy].sort_values("n_assets")
        ax.plot(
            part["n_assets"],
            part["volatility"],
            marker="o",
            label=labels[strategy],
            color=colors[strategy]
        )
    ax.set(
        xlabel="Number of assets (N)",
        ylabel="Annualized Volatility",
        title="Out-of-sample Volatility"
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "oos_volatility.png", dpi=160)
    fig.savefig(thesis_dir / "oos_volatility.pdf", bbox_inches="tight")
    plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(7,4))

    for strategy in strategies:
        part = summary[summary["strategy"] == strategy].sort_values("n_assets")
        ax.plot(
            part["n_assets"],
            part["sharpe"],
            marker="o",
            label=labels[strategy],
            color=colors[strategy]
        )
    ax.axhline(
        0.0,
        color="0.4",
        linestyle="--",
        linewidth=1
    )
    ax.set(
        xlabel="Number of assets (N)",
        ylabel="Sharpe ratio",
        title="Out-of-Sample Sharpe Ratio"
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "oos_sharpe.png", dpi=160)
    fig.savefig(thesis_dir / "oos_sharpe.pdf", bbox_inches="tight")
    plt.close(fig)
    
    turnover_summary = rebalances.groupby(["n_assets", "strategy"], as_index=False)["turnover"].mean()

    fig, ax = plt.subplots(figsize=(7,4))

    for strategy in strategies:
        part = turnover_summary[turnover_summary["strategy"] == strategy].sort_values("n_assets")
        ax.plot(
            part["n_assets"],
            part["turnover"],
            marker="o",
            label=labels[strategy],
            color=colors[strategy]
        )
    ax.set(
        xlabel="Number of assets (N)",
        ylabel="Mean turnover",
        title="Average Portfolio Turnover"
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_turnover.png", dpi=160)
    fig.savefig(thesis_dir / "mean_turnover.pdf", bbox_inches="tight")
    plt.close(fig)
    
    effective_n_summary = rebalances.groupby(["n_assets", "strategy"], as_index=False)["effective_n"].mean()

    fig, ax = plt.subplots(figsize=(7,4))

    for strategy in strategies:
        part = effective_n_summary[effective_n_summary["strategy"] == strategy].sort_values("n_assets")
        ax.plot(
            part["n_assets"],
            part["effective_n"],
            marker="o",
            label=labels[strategy],
            color=colors[strategy]
        )
    n_values = sorted(effective_n_summary["n_assets"].unique())
    ax.plot(
        n_values,
        n_values,
        linestyle="--",
        linewidth=1,
        color="0.4",
        label="1/N benchmark"
    )
    ax.set(
        xlabel="Number of assets (N)",
        ylabel="Effective number of assets",
        title="Portfolio Concentration"
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "effective_n.png", dpi=160)
    fig.savefig(thesis_dir / "effective_n.pdf", bbox_inches="tight")
    plt.close(fig)

    wealth_min = 1.0
    wealth_max = 1.0

    for _, part in returns.groupby(["n_assets", "strategy"]):
        part = part.sort_values("date")

        wealth = (1.0 + part["return"]).cumprod()
        wealth_gross = (1.0 + part["return_gross"]).cumprod()

        wealth_min = min(
            wealth_min,
            wealth.min(),
            wealth_gross.min()
        )
        wealth_max = max(
            wealth_max,
            wealth.max(),
            wealth_gross.max()
        )

    wealth_padding = 0.05 * (wealth_max - wealth_min)
    wealth_ylim = (
        wealth_min - wealth_padding,
        wealth_max + wealth_padding
    )

    for n_assets, part_n in returns.groupby("n_assets"):
        fig, ax = plt.subplots(figsize=(8, 4))

        for strategy in strategies:
            part = part_n[part_n["strategy"] == strategy].sort_values("date").copy()
            if part.empty:
                continue
            part["wealth"] = (1.0 + part["return"]).cumprod()
            part["wealth_gross"] = (1.0 + part["return_gross"]).cumprod()
            ax.plot(
                part["date"],
                part["wealth"],
                label=labels[strategy],
                color=colors[strategy]
            )
            ax.plot(
                part["date"],
                part["wealth_gross"],
                color=colors[strategy],
                linestyle="--",
                linewidth=1,
                alpha=0.7
            )
        
        ax.axhline(
            1.0,
            color="0.4",
            linestyle="--",
            linewidth=1
        )
        ax.set(
            xlabel="Date",
            ylabel="Cumulative wealth",
            title=f"Cumulative Wealth - N = {n_assets}"
        )
        ax.set_ylim(wealth_ylim)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"cumulative_wealth_N{n_assets}.png", dpi=160)
        fig.savefig(thesis_dir / f"cumulative_wealth_N{n_assets}.pdf", bbox_inches="tight")
        plt.close(fig)