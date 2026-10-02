import pandas as pd
import numpy as np

from pathlib import Path
from .config import RealisticConfig 

REQUIRED_COLUMNS = {"Date", "DSCD", "Return"}
def load_panel(path: Path) -> pd.DataFrame:
    """Load only the panel columns needed for the rolling experiment."""
    columns = ["Date", "DSCD", "Return", "MarketCAP"]
    suffix = path.suffix.lower()
    if suffix == ".feather":
        panel = pd.read_feather(path, columns=columns)
    elif suffix == ".csv":
        panel = pd.read_csv(path, usecols=lambda col: col in columns)
    else:
        raise ValueError("Supported formats: .feather, .csv")
    missing = REQUIRED_COLUMNS.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    panel["Date"] = pd.to_datetime(panel["Date"])
    return panel.sort_values("Date")

"""
This section includes the inital Analysis of the raw Data panel. Parts of it won't be relevant once rules are already at play,
but it documents how the rules regarding the addressing of missing values, zero returns, etc. originated.
"""

def do_analysis_raw(panel: pd.DataFrame, config: RealisticConfig, output_dir: Path):
    diagnostics = analyse_raw(panel, config)
    write_analysis_raw(diagnostics, output_dir)

def analyse_raw(raw: pd.DataFrame, config: RealisticConfig) -> pd.DataFrame:
    annual = config.annualization
    
    grouped = raw.groupby("DSCD")
    asset_diagnostics = grouped.agg(
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        n_rows=("Return", "size"),
        n_returns=("Return", "count"),
        n_market_cap=("MarketCAP", "count")
    )

    asset_diagnostics["missing_return_fraction"] = 1.0 - asset_diagnostics["n_returns"] / asset_diagnostics["n_rows"] 
    asset_diagnostics["zero_return_fraction"] = grouped["Return"].apply(lambda x: (x.dropna() == 0.0).mean())
    asset_diagnostics["annualized_volatility"] = grouped["Return"].std(ddof=1) * np.sqrt(annual)
    asset_diagnostics["mcap_available_fraction"] = asset_diagnostics["n_market_cap"] / asset_diagnostics["n_rows"]

    return asset_diagnostics

def write_analysis_raw(raw_diagnostics: pd.DataFrame, output_dir: Path) -> None:
    """write transparent data-quality report for raw panel"""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "raw_panel_data_quality.txt"

    zero_thresholds = [
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.75,
        0.90
    ]
    missing_thresholds = [
        0.00,
        0.01,
        0.05,
        0.10,
        0.20,
        0.50
    ]

    with out.open("w", encoding="utf-8") as f:
        f.write("=== DATA QUALITY ANALYSIS FOR RAW US PANEL ===\n\n")
        descriptions = {
            "MISSING RETURN DESCRIPTION": raw_diagnostics["missing_return_fraction"].describe(),
            "ZERO RETURN DESCRIPTION": raw_diagnostics["zero_return_fraction"].describe(),
            "ANNUALIZED VOLATILITY DESCRIPTION": raw_diagnostics["annualized_volatility"].describe(),
            "MCAP AVAILABILITY DESCRIPTION": raw_diagnostics["mcap_available_fraction"].describe()
        }
        for title, description in descriptions.items():
            f.write(f"{title}:\n")
            f.write(description.to_string())
            f.write("\n")
        f.write("\n")
        f.write("ZERO RETURN THRESHOLD COUNTS:\n")

        valid_zeros = raw_diagnostics["zero_return_fraction"].dropna()
        total_zeros = len(valid_zeros)
        for th in zero_thresholds:
            count = int((valid_zeros > th).sum())
            share = count / total_zeros if total_zeros > 0 else float("nan")
            f.write(f"> {th:.0%}: {count} assets ({share:.2%})\n")
        f.write("\n")
        f.write("MISSING RETURN THRESHOLD COUNTS:\n")

        valid_missings = raw_diagnostics["missing_return_fraction"].dropna()
        total_missings = len(valid_missings)
        for th in missing_thresholds:
            count = int((valid_missings > th).sum())
            share = count / total_missings if total_missings > 0 else float("nan")
            f.write(f"> {th:.0%}: {count} assets ({share:.2%})\n")
        f.write("\n")
        f.write("GENERAL COUNTS:\n")

        f.write(f"Assets in diagnostics: {len(raw_diagnostics)}\n")
        f.write(f"Assets with valid zero-return metric: {total_zeros}\n")
        f.write(f"Assets with valid missing-return metric: {total_missings}\n")
        assets_without_missing = (raw_diagnostics["missing_return_fraction"] == 0).sum()
        f.write(f"Assets without missing returns: {assets_without_missing}\n")