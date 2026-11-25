# Babel - Implementation Plan

## Goal

Build a system where two agents learn a discrete symbolic language without being assigned symbol meanings in advance.

- The **Speaker** observes an input and sends a short message.
- The **Listener** receives the message only and predicts the requested properties.
- The learned language is compared with standard methods such as K-Means and a decision tree under the same information budget.

## Rules that apply throughout the project

1. The Listener must never receive the raw input, labels, Speaker hidden states, or Speaker logits. It receives message IDs only.
2. Train, validation, and test splits are created once per dataset and saved.
3. Preprocessing such as scaling, PCA, K-Means, and trees is fit using training data only.
4. The test set is not used to select settings; it is used only for final evaluation.
5. Each run saves its configuration, seed, metrics, and checkpoints.
6. Every new metric needs a test using known examples before it is used in results.
7. Symbol meanings and messages must not be hard-coded. Agents must learn the communication code during training.

---

## Phase 0 - Project Setup

### Goal

Create a stable project structure so every experiment can be reproduced.

### Structure

```text
Babel/
├── README.md
├── pyproject.toml
├── configs/
├── data/
├── docs/
├── runs/
├── src/babel/
│   ├── config.py
│   ├── seeds.py
│   ├── store.py
│   ├── datasets/
│   └── agents/
└── tests/
```

### Deliverables

- [x] `pyproject.toml` defines the project and its `numpy` dependency.
- [x] `README.md` explains how to run the project.
- [x] `config.py` defines `RunConfig`.
- [x] `seeds.py` provides separate random streams for data, environment, Speaker, Listener, and evaluation.
- [x] `store.py` saves configurations, metrics, and checkpoints.

### Exit criterion

A basic experiment creates a directory inside `runs/` containing `config.json`, `metrics.jsonl`, and a checkpoint.

---

## Phase 1 - Synthetic Grammar Lab

### Goal

Build the smallest world that proves the communication game works.

### Dataset

| Property | Values |
|---|---|
| Shape | circle, square, triangle |
| Color | red, green, blue |
| Total objects | 9 combinations |

Each object is represented as a feature vector containing its shape and color.

### Protocol

```text
Object features
      ↓
Speaker
      ↓
Message: [symbol_1, symbol_2]
      ↓
Listener
      ↓
Predicted shape + predicted color
```

### Files

- [x] `datasets/synthetic.py`: generates all 9 objects.
- [x] `env.py`: samples an object from the synthetic world.
- [x] `agents/tabular.py`: contains tabular Speaker and Listener agents.
- [x] `train.py`: contains the training loop.
- [x] `metrics.py`: evaluates the language and predictive accuracy.
- [x] `cli.py`: runs training from the command line.

### Current training design

- The Speaker selects a two-symbol message.
- Each message position has three possible symbols.
- The Listener learns shape and color distributions for each observed message.
- The Speaker reward is the mean probability that the Listener assigns to the correct shape and color.
- The Speaker is trained with REINFORCE, an exponential moving average baseline, and an entropy bonus.

### Exit criterion

- The Listener reaches high shape and color accuracy using messages only.
- Results are saved for every seed.
- The learned messages can be inspected to determine whether they form an organized code.

---

## Phase 2 - Tests and Language Metrics

### Goal

Verify that reported metrics are correct before using them in comparisons or a report.

### Required metrics

| Metric | Question answered |
|---|---|
| Shape accuracy | Did the Listener identify shape? |
| Color accuracy | Did the Listener identify color? |
| Exact accuracy | Did the Listener identify both attributes? |
| Message entropy | Does the Speaker use multiple messages or collapse to one? |
| Mutual information | Is a symbol or message position associated with a property? |
| Active vocabulary | How many symbols are actually used? |
| Ambiguity | Does one message refer to several objects? |
| Synonymy | Does one object receive several messages? |

### Deliverables

- [x] Add tests for entropy and mutual information.
- [x] Create a known compositional oracle language for metric tests.
- [x] Create a random oracle language to confirm metrics do not report false structure.
- [x] Add active vocabulary, ambiguity, and synonymy metrics.
- [x] Define exact formulas for positional disentanglement and topographic similarity before implementing them.

### Exit criterion

Metrics have tests against known and random oracle examples, and the project has a working test command. Runtime test results still need to be confirmed in an environment with Python available.

---

## Phase 3 - Tabular Real Data

### Goal

Apply the same communication game to simple real datasets.

### Datasets

| Dataset | Purpose |
|---|---|
| Wine | First tabular benchmark for rate-accuracy curves |
| Car Evaluation | Grammar and generalization test using held-out attribute combinations |

### New files

```text
src/babel/
├── datasets/tabular.py
├── preprocessing.py
├── split_guard.py
└── data_cards.py
```

### Deliverables

- [ ] Load data reproducibly.
- [ ] Add a data card for each dataset: source, columns, labels, usage, and limitations.
- [ ] Add `SplitGuard` to save split indexes and their SHA-256 hash.
- [ ] Create stratified train, validation, and test splits.
- [ ] Fit `StandardScaler` using training data only.
- [ ] Select hyperparameters using validation data only.

### Car Evaluation generalization test

- The Speaker observes the full row.
- The Listener receives only a message.
- The Listener predicts two selected attributes, such as `buying` and `safety`.
- Specific combinations are hidden during training.
- Evaluate performance on the held-out combinations.

### Exit criterion

The experiment proves that test data was not used during preprocessing or tuning, and held-out-combination results are saved separately.

---

## Phase 4 - Fair Baselines

### Goal

Determine whether learned communication adds value beyond standard methods.

### Baselines

| Baseline | Role |
|---|---|
| Full-data oracle: Logistic Regression or MLP | Upper performance reference when the Listener sees raw data |
| K-Means Vector Quantization | Standard discrete-code comparison |
| Size-limited Decision Tree | Interpretable, limited-size code |

### Deliverables

- [ ] Use the same data split for every baseline.
- [ ] Match the communication budget: number of symbols, message length, or bits.
- [ ] Fit all preprocessing on training data only.
- [ ] Save predictions for every method for later confusion matrices and analysis.

### Exit criterion

Every experiment provides a direct comparison between emergent communication and each baseline under the same settings.

---

## Phase 5 - Neural Agents

### Goal

Move from tabular agents and the small synthetic world to neural agents and more complex data.

### New files

```text
src/babel/agents/
├── neural.py
└── baselines.py
```

### Deliverables

- [ ] Implement MLP Speaker and Listener agents for tabular data.
- [ ] Train the Listener with cross-entropy.
- [ ] Train the Speaker with REINFORCE, an EMA baseline, and an entropy bonus.
- [ ] Add an ablation where both agents use REINFORCE.
- [ ] Log loss, reward, active vocabulary, and accuracy.

### Exit criterion

Neural runs are reproducible across multiple seeds and the Listener still receives no raw input information.

---

## Phase 6 - Fashion-MNIST

### Goal

Test the communication system using features derived from images.

### Deliverables

- [ ] Load Fashion-MNIST.
- [ ] Fit PCA using training images only.
- [ ] Use PCA features as the MLP Speaker input.
- [ ] Start with `L = 1` messages for rate-accuracy curves.
- [ ] Keep a CNN Speaker as a stretch goal after the MLP version works.

### Exit criterion

Results show a clear curve between communication budget in bits and predictive accuracy, compared with the baselines.

---

## Phase 7 - Experiments and Reporting

### Goal

Run experiments consistently and generate reproducible report results.

### Deliverables

- [ ] Add YAML manifests for every experiment.
- [ ] Use five seeds as the standard setting.
- [ ] Report means and confidence intervals.
- [ ] Generate rate-accuracy curves.
- [ ] Run grammar sweeps with small vocabularies, such as `K = 3` or `K = 4`.
- [ ] Report held-out zero-shot results.
- [ ] Generate confusion matrices and ROC AUC where appropriate.
- [ ] Report collapse, memorization, or baseline parity honestly.

### Exit criterion

Every figure in the report can be regenerated using saved configurations and run artifacts.

---

## Phase 8 - Streamlit Demo

### Goal

Create a user interface that replays saved runs without training during a demonstration.

### Deliverables

- [ ] `app.py` reads saved JSONL, NPZ, and checkpoint artifacts only.
- [ ] Show the synthetic world and learned messages.
- [ ] Display a dictionary of learned symbols.
- [ ] Add plots for accuracy, entropy, and mutual information.
- [ ] Add a small human arena for compact vocabularies.
- [ ] Use HTML/SVG to display the communication game stage.

### Exit criterion

`make demo` opens a stable interface using existing run artifacts.

---

## Stretch Goals

These are implemented only after the core phases are complete:

- [ ] Adversarial Spy mode.
- [ ] Evolutionary or genetic agents.
- [ ] Candidate-subset ablation.
- [ ] Iterated learning.
- [ ] CNN Speaker for Fashion-MNIST.
- [ ] Twenty-seed sweeps.

---

## Recommended implementation order

1. Run and inspect Phase 1 locally until each run saves metrics and checkpoints.
2. Add missing metrics and tests in Phase 2.
3. Implement Wine and Car Evaluation with `SplitGuard`.
4. Add the baselines before neural agents, so comparisons exist from the beginning.
5. After tabular experiments are stable, add neural agents and Fashion-MNIST.
