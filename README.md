# The Impact of Parameter Estimation on the Performance of traditional Portfolio Optimization

The project includes the python implementation of the two-layered experiment, as well as its results. 

In this README, there's a quick explanation for running the experiment yourself, as well as a concise overview of the project.

## Installation

Install all needed packages in one go.

```powershell
python -m pip install -r requirements.txt
```

## Controlled Monte Carlo Simulation Analysis

Perform short smoke run (only for testing purposes):

```powershell
python run_experiment.py controlled --quick
```

Perform regular simulation:

```powershell
python run_experiment.py controlled
```

Results are stored in `outputs/controlled/` by default. However, choosing a path is possible via `--output <path>`.
*I recommend using the default paths.*

*Note*: Path logic is relative throughout the project, the used Path libary should prevent pathing issues with Mac.

*Note*: Rerunning the experiments will overwrite results. Either copy them away or referr to the repo for the thesis results.

## Empirical Rolling-Window Analysis

Perform run on a given panel:

```powershell
python run_experiment.py realistic --data ".\panels\US_data_panel_filtered_0.15.feather"
```

Outputs into `outputs\realistic\`.

*Note*: Input data is not included in the repository, since I don't want to publish non-public data in a public repository. Just copy the path of your own data panel into the `--data` parameter.

## Evaluate Results

Peform the evaluation of the results:

```powershell
python run_experiment.py evaluation
```

Evaluates all hypotheses and outputs into `outputs\results\`.

*Note*: Only works if `outputs\controlled\` and `outputs\realistic\` is present.

## Structure

```text
portfolio_experiment/
  config.py         # Run calibration
  dgp.py            # construct synthetic truth  
  estimators.py     # Sample and Ledoit-Wolf
  optimization.py   # GMV-optimization
  metrics.py        # Error and performance metrics
  preprocessing.py  # Data diagnostics
  interpretation.py # Hypotheses operationalization
  controlled.py     # Part A
  realistic.py      # Part B
run_experiment.py   # Application entry point
panels\             # Input files
outputs\            # Output files 
README              # Info
```

**Important:** `config.py` currently represents the exact calibration used in the final runs.

