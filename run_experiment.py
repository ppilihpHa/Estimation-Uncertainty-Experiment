from __future__ import annotations

import argparse
from pathlib import Path

from portfolio_experiment.config import ControlledConfig, RealisticConfig
from portfolio_experiment.controlled import quick_config, run_controlled
from portfolio_experiment.realistic import run_realistic
from portfolio_experiment.interpretation import check_results_controlled, check_results_realistic
from portfolio_experiment.preprocessing import load_panel, do_analysis_raw
from portfolio_experiment.interpretation import analyse_Hypotheses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two-stage portfolio experiment.")
    stages = parser.add_subparsers(dest="stage", required=True)
    controlled = stages.add_parser("controlled", help="Run the controlled Monte-Carlo stage.")
    controlled.add_argument("--quick", action="store_true", help="Use a small test configuration.")
    controlled.add_argument("--output", type=Path, default=Path("outputs/controlled"))
    realistic = stages.add_parser("realistic", help="Run the rolling-window stage.")
    realistic.add_argument("--data", type=Path, required=True)
    realistic.add_argument("--output", type=Path, default=Path("outputs/realistic"))
    evaluation = stages.add_parser("evaluation", help="Evaluation the hypotheses based on the results.")
    evaluation.add_argument("--results", type=Path, default=Path("outputs/"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "controlled":
        config = quick_config(ControlledConfig()) if args.quick else ControlledConfig()
        results, sim_data = run_controlled(config, args.output)
        check_results_controlled(config, args.output, sim_data)
        print(f"Controlled stage complete: {len(results):,} rows -> {args.output.resolve()}")
    elif args.stage == "evaluation":
        c_config = ControlledConfig()
        r_config = RealisticConfig()
        analyse_Hypotheses(c_config, r_config, args.results)
        print("Analysis stage complete")
    else:
        config = RealisticConfig()
        panel = load_panel(args.data)
        do_analysis_raw(panel, config, args.output)
        returns, rebalances, diagnostics = run_realistic(panel, config, args.output)
        check_results_realistic(returns, rebalances, diagnostics, config)
        print(f"Realistic stage complete: {len(returns):,} daily rows -> {args.output.resolve()}")


if __name__ == "__main__":
    main()

