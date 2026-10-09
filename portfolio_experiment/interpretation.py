import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from .config import ControlledConfig, RealisticConfig

def check_results_controlled(config : ControlledConfig, outpath : Path, sim_data: dict) -> None:
    results = pd.read_csv( outpath / "replications.csv")

    """Structural checks"""

    expected_estimators = {"sample", "ledoit_wolf"}
    assert set(results["estimator"].unique()) == expected_estimators 

    expected_rows = (
        len(config.covariance_structures) *
        len(config.dimensions) * 
        len(config.information_ratios) * 
        len(expected_estimators) * 
        config.replications
    )
    assert len(results) == expected_rows 

    paired_counts = (
        results.groupby([
            "covariance_structure",
            "n_assets",
            "n_observations",
            "requested_information_ratio",
            "replication"
        ])["estimator"].nunique()
    )
    assert (paired_counts == len(expected_estimators)).all() 

    key = [
        "covariance_structure",
        "n_assets",
        "n_observations",
        "requested_information_ratio",
        "replication",
        "estimator"
    ]
    assert not results.duplicated(key).any() 

    """Numerical checks"""

    numeric_cols = [
        "estimation_error",
        "decision_error",
        "risk_increment"
    ]
    assert np.isfinite(results[numeric_cols].to_numpy()).all() 

    assert (results["risk_increment"] >= -1e-10).all()
    assert (results["estimation_error"] >= 0).all() 
    assert (results["decision_error"] >= 0).all()

    """Numerical scenario checks"""

    for scenario_key, scenario in sim_data["scenarios"].items():
        sigma_true = scenario["sigma_true"]
        oracle = scenario["oracle_weights"]
        oracle_vol = scenario["oracle_volatility"]

        assert sigma_true.shape[0] == sigma_true.shape[1]
        assert np.allclose(sigma_true, sigma_true.T, atol=1e-10) 
        assert np.isfinite(sigma_true).all() 

        assert np.linalg.eigvalsh(sigma_true).min() > 0.0 

        assert np.isclose(oracle.sum(), 1.0, atol=1e-10) 
        assert np.isfinite(oracle).all()

        assert oracle_vol > 0.0 

    """Replication numerical checks"""

    for rep_key, rep in sim_data["replications"].items():
        for estimator, est in rep["estimators"].items():
            sigma_hat = est["sigma_hat"]
            weights = est["weights"]

            assert sigma_hat.shape[0] == sigma_hat.shape[1]
            assert np.allclose(sigma_hat, sigma_hat.T, atol=1e-10) 
            assert np.isfinite(sigma_hat).all() 

            assert np.linalg.eigvalsh(sigma_hat).min() >= -1e-10 

            assert np.isfinite(weights).all()
            assert np.isclose(weights.sum(), 1.0, atol=1e-8)
    
    print("INFO | sanity checks completed")

def check_results_realistic(returns: pd.DataFrame, rebalances: pd.DataFrame, diagnostics: pd.DataFrame, config: RealisticConfig) -> None:
    """Structural checks"""
    strategies = config.strategies
    for n_assets, part in rebalances.groupby("n_assets"):
        dates_by_strategy = {
            strategy : set(part.loc[part["strategy"] == strategy, "rebalance_date"]) for strategy in strategies
        }
        reference = dates_by_strategy[strategies[0]]
        for strategy in strategies[1:]:
            assert dates_by_strategy[strategy] == reference 

    for n_assets, part in returns.groupby("n_assets"):
        dates_by_strategy = {
            strategy : set(part.loc[part["strategy"] == strategy, "date"]) for strategy in strategies
        }
        reference = dates_by_strategy[strategies[0]]
        for strategy in strategies[1:]:
            assert dates_by_strategy[strategy] == reference  

    assert not returns.duplicated(["date", "n_assets", "strategy"]).any()
    assert not rebalances.duplicated(["rebalance_date", "n_assets", "strategy"]).any()

    """Numerical Checks"""

    assert np.isfinite(returns["return"].to_numpy()).all()  
    for (_, _), part in rebalances.groupby(["n_assets", "strategy"]):
        part = part.sort_values("rebalance_date")
        assert np.isfinite(part.iloc[1:]["turnover"].to_numpy()).all() 

    assert (rebalances["concentration"] <= 1.0 + 1e-10).all() 
    lower_bound = 1.0 / rebalances["n_assets"]
    assert (rebalances["concentration"] >= lower_bound - 1e-10).all() 

    assert (rebalances["effective_n"] >= 1.0 - 1e-10).all() 
    assert (rebalances["effective_n"] <= rebalances["n_assets"] + 1e-10).all() 

    assert (rebalances["max_weight"] >= -1e-10).all() 
    assert (rebalances["max_weight"] <= 1.0 + 1e-10).all() 

    assert (rebalances["transaction_cost"].dropna() >= 0).all()

    """Data Quality Diagnostics"""
    
    warnings = []

    assert np.isfinite(diagnostics["n_observations"].to_numpy()).all()
    
    if config.max_missing_fraction == 0.0:
        fail = diagnostics[diagnostics["n_observations"] != config.lookback_days]
        if not fail.empty:
            warnings.append(
                f"{len(fail)} rebalance dates have fewer than {config.lookback_days} observations"
            )
    
    fail = diagnostics[diagnostics["n_eligible"] < diagnostics["n_assets"]]
    if not fail.empty:
        warnings.append(
            f"{len(fail)} cells had fewer egligible assets than requested"
        )
    
    fail = diagnostics[diagnostics["n_missing_test"] > 0]
    if not fail.empty:
        warnings.append(
            f"{len(fail)} rebalance dates contain missing oos returns"
        )
    
    fail = diagnostics[diagnostics["max_zero_fraction"] > 0.20]
    if not fail.empty:
        warnings.append(
            f"{len(fail)} cells contain at least one asset with >20% zero rturns"
        )
    
    fail = diagnostics[diagnostics["min_asset_vol_annual"] < 0.02]
    if not fail.empty:
        warnings.append(
            f"{len(fail)} cells contain assets with suspiciously low annualized volatilty (<2%)"
        )
    
    data_quality_diagnostics = diagnostics.groupby("n_assets")[
        [
            "n_missing_test",
            "mean_zero_fraction",
            "max_zero_fraction",
            "median_asset_vol_annual"
        ]
    ].mean()

    print("---------")
    print(f"DATA QUALITY DIAGNOSTICS | \n{data_quality_diagnostics}")
    for warning in warnings:
        print(f"WARN | {warning}")
    
    """Portfolio Diagnostics"""

    portfolio_diagnostics = rebalances.groupby(["n_assets", "strategy"]).agg(
        mean_turnover=("turnover", "mean"),
        median_turnover=("turnover", "median"),
        mean_concentration=("concentration", "mean"),
        mean_effective_n=("effective_n", "mean"),
        mean_max_weight=("max_weight", "mean")
    ).reset_index()

    print("---------")
    print(f"PORTFOLIO DIAGNOSTICS | \n{portfolio_diagnostics}")


def analyse_Hypotheses(controlled_config: ControlledConfig, realistic_config: RealisticConfig, outputs_dir: Path):
    result_dir = Path(outputs_dir / "results")
    controlled_dir = Path(outputs_dir / "controlled")
    realistic_dir = Path(outputs_dir / "realistic")
    result_dir.mkdir(parents=True, exist_ok=True)

    H1_report = analyse_H1(controlled_config, controlled_dir)
    H2_report = analyse_H2(ControlledConfig, controlled_dir)
    H3_report = analyse_H3(ControlledConfig, controlled_dir)
    H4_report, H4_rule_report = analyse_H4(ControlledConfig, controlled_dir) 
    H5_report, H5_rule_report = analyse_H5(RealisticConfig, realistic_dir)

    H1_report.to_csv(result_dir / "H1_report.csv", index=False)
    H2_report.to_csv(result_dir / "H2_report.csv", index=False)
    H3_report.to_csv(result_dir / "H3_report.csv", index=False)
    H4_report.to_csv(result_dir / "H4_report.csv", index=False)
    H4_rule_report.to_csv(result_dir / "H4_rule_report.csv", index=False)
    H5_report.to_csv(result_dir / "H5_report.csv", index=False)
    H5_rule_report.to_csv(result_dir / "H5_rule_report.csv", index=False)

def analyse_H1(config: ControlledConfig, output_dir : Path):
    summary = pd.read_csv(output_dir / "summary.csv")
    summary = summary.sort_values(["n_assets","estimator", "information_ratio"])

    report = []

    for (n, estimator), group in summary.groupby(["n_assets", "estimator"]):
        group = group.sort_values("information_ratio")
        ratios = group["information_ratio"].to_numpy()
        errors = group["estimation_error_mean"].to_numpy()
        
        error_min = errors[0]
        error_max = errors[-1]

        absolute_change = error_max - error_min
        relative_change = absolute_change / error_min

        rho, _ = spearmanr(ratios, errors)

        report.append({
            "N": n,
            "estimator": estimator,
            "c_min": ratios[0],
            "c_max": ratios[-1],
            "error_min": error_min,
            "error_max": error_max,
            "absolute_change": absolute_change,
            "relative_change": relative_change,
            "spearman_rho": rho
        })
    
    report = pd.DataFrame(report)
    return report

def analyse_H2(controlled_config: ControlledConfig, output_dir: Path):
    replications = pd.read_csv(output_dir / "replications.csv")
    replications = replications.sort_values(["n_assets", "estimator", "information_ratio"])
    regular = replications[replications["information_ratio"] < 1.0].copy()
    report = []

    for (n, estimator), group in regular.groupby(["n_assets", "estimator"]):
        estimation_errors = group["estimation_error"].to_numpy()
        decision_errors = group["decision_error"].to_numpy()

        rho, _ = spearmanr(estimation_errors, decision_errors)

        report.append({
            "N": n,
            "estimator": estimator,
            "rho": rho,
            "rule": (rho > 0)
        })
    
    report = pd.DataFrame(report)
    return report

def analyse_H3(controlled_config: ControlledConfig, output_dir: Path):
    replications = pd.read_csv(output_dir / "replications.csv")
    replications = replications.sort_values(["n_assets", "estimator", "information_ratio"])
    regular = replications[replications["information_ratio"] < 1.0].copy()
    report = []

    for (n, estimator), group in regular.groupby(["n_assets", "estimator"]):
        decision_errors = group["decision_error"].to_numpy()
        risk_increments = group["risk_increment"].to_numpy()

        rho, _ = spearmanr(decision_errors, risk_increments)

        report.append({
            "N": n,
            "estimator": estimator,
            "rho": rho,
            "rule": (rho > 0)
        })
    
    report = pd.DataFrame(report)
    return report

def analyse_H4(controlled_config: ControlledConfig, output_dir: Path):
    summary = pd.read_csv(output_dir / "summary.csv")
    summary = summary[summary["information_ratio"] < 1.0].copy()

    estimation = summary.pivot(
        index=["n_assets", "information_ratio"],
        columns="estimator",
        values="estimation_error_mean"
    ).reset_index()

    decision = summary.pivot(
        index=["n_assets", "information_ratio"],
        columns="estimator",
        values="decision_error_mean"
    ).reset_index()

    consequence = summary.pivot(
        index=["n_assets", "information_ratio"],
        columns="estimator",
        values="risk_increment_mean"
    ).reset_index()

    report = estimation[["n_assets", "information_ratio"]].copy()

    report["sample_estimation_error"] = estimation["sample"]
    report["lw_estimation_error"] = estimation["ledoit_wolf"]

    report["sample_decision_error"] = decision["sample"]
    report["lw_decision_error"] = decision["ledoit_wolf"]

    report["sample_risk_increment"] = consequence["sample"]
    report["lw_risk_increment"] = consequence["ledoit_wolf"]

    report["RI_sigma"] = (
        (report["sample_estimation_error"] - report["lw_estimation_error"]) / report["sample_estimation_error"]
    )

    report["RI_w"] = (
    (report["sample_decision_error"] - report["lw_decision_error"]) / report["sample_decision_error"]
    )

    report["RI_L"] = (
    (report["sample_risk_increment"] - report["lw_risk_increment"]) / report["sample_risk_increment"]
    )

    rho_results = []

    for n, group in report.groupby("n_assets"):
        rho_w, _ = spearmanr(
            group["information_ratio"],
            group["RI_w"]
        )

        rho_sigma, _ = spearmanr(
            group["information_ratio"],
            group["RI_sigma"]
        )

        rho_L, _ = spearmanr(
            group["information_ratio"],
            group["RI_L"]
        )

        positive_share_sigma = (group["RI_sigma"] > 0).mean()
        positive_share_w = (group["RI_w"] > 0).mean()
        positive_share_L = (group["RI_L"] > 0).mean()

        rho_results.append({
            "N": n,
            "rho_RI_sigma": rho_sigma,
            "rule_RI_sigma": rho_sigma > 0,
            "positive_share_RI_sigma": positive_share_sigma,
            "rho_RI_w": rho_w,
            "rule_RI_w": rho_w > 0,
            "positive_share_RI_w": positive_share_w,
            "rho_L": rho_L,
            "rule_L": rho_L > 0,
            "positive_share_RI_L": positive_share_L
        })
    
    report = pd.DataFrame(report)
    rho_results = pd.DataFrame(rho_results)

    return report, rho_results

def analyse_H5(realistic_config: RealisticConfig, output_dir: Path):
    summary = pd.read_csv(output_dir / "summary.csv")

    turnover = summary.pivot(
        index="n_assets",
        columns="strategy",
        values="mean_turnover"
    ).reset_index()

    concentration = summary.pivot(
        index="n_assets",
        columns="strategy",
        values="mean_effective_n"
    ).reset_index()

    risk = summary.pivot(
        index="n_assets",
        columns="strategy",
        values="volatility"
    ).reset_index()

    report = pd.DataFrame({
        "N": turnover["n_assets"],
        "sample_turnover": turnover["sample_gmv"],
        "lw_turnover": turnover["ledoit_wolf_gmv"],
        "sample_effective_n": concentration["sample_gmv"],
        "lw_effective_n": concentration["ledoit_wolf_gmv"],
        "sample_volatility": risk["sample_gmv"],
        "lw_volatility": risk["ledoit_wolf_gmv"]
    })

    report["RI_TO"] = (report["sample_turnover"] - report["lw_turnover"]) / report["sample_turnover"]
    report["delta_effective_n"] = report["lw_effective_n"] - report["sample_effective_n"] 
    report["RI_risk"] = (report["sample_volatility"] - report["lw_volatility"]) / report["sample_volatility"]

    positive_share_turnover = (report["RI_TO"] > 0).mean()
    positive_share_concentration = (report["delta_effective_n"] > 0).mean()
    positive_share_risk = (report["RI_risk"] > 0).mean()

    rho_turnover, _ = spearmanr(
        report["N"],
        report["RI_TO"]
    )

    rho_concentration, _ = spearmanr(
        report["N"],
        report["delta_effective_n"]
    )

    rho_risk, _ = spearmanr(
        report["N"],
        report["RI_risk"]
    )

    H5_report = report
    H5_rule_report = pd.DataFrame([{
        "positive_share_turnover": positive_share_turnover,
        "positive_share_concentration": positive_share_concentration,
        "positive_share_risk": positive_share_risk,
        "rho_RI_turnover": rho_turnover,
        "rho_RI_concentration": rho_concentration,
        "rho_RI_risk": rho_risk
    }])

    return H5_report, H5_rule_report

