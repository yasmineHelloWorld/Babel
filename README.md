# Babel

Babel is a research project for studying emergent discrete symbolic communication between a neural Speaker and Listener agent. The Speaker observes an input and emits a sequence of discrete symbol IDs; the Listener receives *only* those IDs and predicts target properties. Symbol meanings are never assigned in advance — both agents learn a communication code from task reward.

> **Environment note:** C: is nearly full. Use `D:\babel-env\Scripts\python.exe` (Python 3.12 venv on D:) for all commands, or activate it with `D:\babel-env\Scripts\Activate.ps1`.

## Setup

```powershell
# Create venv on D: (C: has limited space)
python -m venv D:\babel-env
D:\babel-env\Scripts\Activate.ps1
pip install -e .
```

## Run tests (35 tests, all phases)

```powershell
D:\babel-env\Scripts\python.exe -m pytest -v
# or via Makefile:
make test
```

Phase 2 metric definitions and formulas are documented in [`docs/metric-formulas.md`](docs/metric-formulas.md).

---

## Phase 1 — Synthetic world

```powershell
babel train --seed 0 --episodes 30000
```

Trains tabular Speaker/Listener on a 3×3 (shape × color) world. Each run creates a unique directory under `runs/` with `config.json`, `metrics.jsonl`, and `checkpoint.npz`.

---

## Phase 3 — Tabular benchmarks

Congressional Voting Records (435 historical 1984 U.S. House records, 16 categorical votes, Democrat/Republican labels) and Car Evaluation (rule-generated, 1728 rows, 6 categorical attributes).

```powershell
babel prepare-tabular --config configs/congressional-voting-records.json
babel prepare-tabular --config configs/car-evaluation.json
```

Split indices are stored in `.npz` files with SHA-256 digest. One-hot encoding is fit on training rows only. See data cards in [`docs/data-cards/`](docs/data-cards/).

---

## Phase 4 — Classical baselines

```powershell
babel compare-tabular --config configs/congressional-voting-records.json
babel compare-tabular --config configs/car-evaluation.json --symbols 3 --message-length 2
```

Saves prediction artifacts for K-Means VQ, size-limited decision tree, and full-data oracle (LogReg + MLP) under the same communication budget `K^L`.

---

## Phase 5 — Neural agents

MLP Speaker (REINFORCE + EMA baseline + entropy bonus) and MLP Listener (cross-entropy). Trained via `run-experiment` manifests.

```powershell
babel run-experiment --config configs/exp2_synthetic_grammar.yaml
babel run-experiment --config configs/exp_ablation_both_reinforce.yaml
```

The Listener still receives *only integer symbol IDs* — `assert_discrete_messages` enforces this at runtime and in tests.

---

## Phase 6 — Fashion-MNIST

IDX loader with gzip caching; PCA fitted on training images only (train-only leakage guard tested).

```powershell
babel run-experiment --config configs/exp1_fashion_mnist_rate_accuracy.yaml
```

(Requires internet access on first run to download Fashion-MNIST IDX files.)

---

## Phase 7 — Experiments and reporting

Seven YAML manifests in `configs/`. Run all experiments and regenerate the report:

```powershell
# Run all manifests (uses make):
make reproduce

# Or run individually:
babel run-experiment --config configs/exp1_congressional_rate_accuracy.yaml
babel run-experiment --config configs/exp2_synthetic_grammar.yaml
babel run-experiment --config configs/exp2_car_evaluation_grammar.yaml
babel run-experiment --config configs/exp3_dialect_matrix.yaml
babel run-experiment --config configs/exp_ablation_both_reinforce.yaml
babel run-experiment --config configs/exp_heldout_congressional.yaml

# Regenerate docs/results.md from saved summaries:
babel generate-report --runs-dir runs --output docs/results.md
```

Results are in [`docs/results.md`](docs/results.md). Key finding: at K=2 the emergent code matches k-Means VQ on the congressional held-out set (0.625 vs 0.616); at higher K, symbol collapse (active vocab = 2) prevents the emergent code from using the full budget.

---

## Phase 8 — Streamlit demo

```powershell
make demo
# or:
D:\babel-env\Scripts\python.exe -m streamlit run src/babel/ui/app.py
```

Four tabs: **Replay Theater** (step through saved episodes), **Live Dictionary** (symbol → object mapping), **Observatory** (accuracy/entropy/MI plots), **Human Arena** (guess the Listener's response).

---

## Reproducibility

```powershell
babel reproduce --config configs/exp1_congressional_rate_accuracy.yaml
```

Every run saves `config.json`, `metrics.jsonl`, `checkpoint.npz`, and a `summary.json`. Split SHA-256 hashes are recorded. Preprocessing (scaling, PCA, one-hot encoding) is always fit on training data only.

---

## Makefile targets

| Target | Action |
|---|---|
| `make test` | Run full test suite (35 tests) |
| `make reproduce` | Run all experiment manifests |
| `make demo` | Launch Streamlit demo |
