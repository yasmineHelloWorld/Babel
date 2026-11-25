# Babel

Babel is a small research project for studying how a Speaker and Listener can learn a discrete symbolic language. Phase 1 uses a synthetic world of three shapes and three colors.

## Setup

Use Python 3.10 or newer. Install the project and its NumPy and scikit-learn dependencies:

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

## Tabular benchmarks (Phase 3)

Phase 3 uses UCI Congressional Voting Records as its real-world categorical benchmark and Car Evaluation as a controlled, model-generated attribute-combination benchmark. Congressional Voting Records contains 435 historical House member records, 16 categorical votes, and Democrat/Republican labels. It is a good fit for compact symbolic messages and tests over held-out vote combinations; unknown votes remain a separate category. The source is the [UCI Congressional Voting Records dataset](https://archive.ics.uci.edu/dataset/105/congressional%2Bvoting%2Brecords). Both datasets are downloaded from UCI on first use and cached under `data/`.

```powershell
babel prepare-tabular --config configs/congressional-voting-records.json
```

Use `--dataset car_evaluation` to prepare the controlled benchmark. Split indices are stored in an `.npz` file with their SHA-256 digest and settings in a neighboring `.json`; a summary of the prepared data is written beside them. Existing split files are reused. Categorical one-hot encoding is fit only on training rows, and previously unseen validation or test categories are safely ignored by the encoder. See the data cards in [`docs/data-cards/`](docs/data-cards/).

## Classical baselines (Phase 4)

After preparing a tabular dataset, run the comparisons with the same saved split and a two-position, three-symbol budget (18 possible messages):

```powershell
babel compare-tabular --config configs/congressional-voting-records.json
babel compare-tabular --config configs/car-evaluation.json --symbols 3 --message-length 2
```

This saves `runs/<dataset>-baselines.npz` with train, validation, and test predictions, ground-truth labels, the budget, and the split hash. The full-data logistic-regression oracle sees one-hot features. K-Means is limited to the message capacity and its decoder predicts from cluster IDs only. The decision tree is limited to the same maximum number of leaves. All preprocessing is reused from the Phase 3 train-fitted encoder. Pass `--output` to choose another prediction artifact path.

## Reproducibility

The run configuration and seed are saved with each run. Separate NumPy generators are derived for data, environment sampling, Speaker exploration, Listener initialization, and evaluation. The synthetic dataset contains all nine shape-color combinations, and evaluation enumerates those combinations deterministically.
