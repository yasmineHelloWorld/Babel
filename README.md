# Babel

Babel is a small research project for studying how a Speaker and Listener can learn a discrete symbolic language. Phase 1 uses a synthetic world of three shapes and three colors.

## Setup

Use Python 3.10 or newer. Install the project and its NumPy dependency:

```powershell
python -m pip install -e .
```

## Run the synthetic experiment

```powershell
babel train --seed 0 --episodes 30000
```

Each run creates a unique directory under `runs/` containing `config.json`, episode-level `metrics.jsonl`, and `checkpoint.npz`. The CLI prints the final predictive accuracies and learned messages. You can also run without installing the console command using `python -m babel.cli train ...` after setting `PYTHONPATH=src`.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

Phase 2 metric definitions and formulas are documented in [`docs/metric-formulas.md`](docs/metric-formulas.md).

## Reproducibility

The run configuration and seed are saved with each run. Separate NumPy generators are derived for data, environment sampling, Speaker exploration, Listener initialization, and evaluation. The synthetic dataset contains all nine shape-color combinations, and evaluation enumerates those combinations deterministically.
