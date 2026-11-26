# Babel – Backend README

Emergent discrete symbolic communication between a neural Speaker and Listener, served over WebSocket to the pixel-art frontend.

## Architecture

```
frontend/babel-whispering-dart.html   ← single-file UI (do not restyle)
backend/
  __init__.py
  env.py        ← world objects, one-hot encoding
  model.py      ← Speaker & Listener MLPs (REINFORCE + Gumbel-softmax)
  lexicon.py    ← sliding-window co-occurrence tracker
  trainer.py    ← BabelTrainer orchestrator, judge session
  main.py       ← FastAPI app + WebSocket /ws endpoint
  train.py      ← standalone CLI training script
tests/
  test_whisperer_env.py
  test_whisperer_lexicon.py
  test_whisperer_trainer.py
```

---

## Quick Start

### 1. Install dependencies

```powershell
# Windows – babel-env on D: to avoid C: space issues
$env:PIP_CACHE_DIR="D:\pip-cache"
D:\babel-env\Scripts\pip install -r requirements_backend.txt
```

### 2. Start the server

```powershell
cd C:\Users\Lenovo\Downloads\Babel\Babel
D:\babel-env\Scripts\uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.  
The small dot in the top-right corner of the canvas turns **green** when connected.  
If the backend is offline the page falls back to local simulation automatically.

### 3. Standalone training (no UI)

```powershell
D:\babel-env\Scripts\python.exe -m backend.train --epochs 2000 --seed 0 --plot
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 2000 | Training steps |
| `--seed` | 0 | Global RNG seed |
| `--gumbel` | off | Use Gumbel-softmax instead of REINFORCE |
| `--rival-lr-mult` | 0.2 | Rival learning-rate multiplier |
| `--plot` | off | Save `accuracy_curve.png` after training |
| `--log-dir` | `runs/` | Directory for `.jsonl` logs |

---

## Running Tests

```powershell
# Fast unit tests only (skip slow acceptance tests)
D:\babel-env\Scripts\python.exe -m pytest tests/test_whisperer_env.py tests/test_whisperer_lexicon.py tests/test_whisperer_trainer.py -v -m "not slow"

# Include slow acceptance tests (~2 min)
D:\babel-env\Scripts\python.exe -m pytest tests/test_whisperer_env.py tests/test_whisperer_lexicon.py tests/test_whisperer_trainer.py -v

# All tests (including original Babel research suite)
D:\babel-env\Scripts\python.exe -m pytest -v
```

---

## WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws`

### Client → Server

| Command | Payload | Description |
|---------|---------|-------------|
| `whisper` | `{"cmd":"whisper"}` | Train one step + sample one round |
| `evolve` | `{"cmd":"evolve","on":true\|false}` | Start/stop continuous training loop |
| `skip` | `{"cmd":"skip","n":25}` | Run n silent training steps, return stats |
| `judge_start` | `{"cmd":"judge_start"}` | Begin 8-round human vs rival judgment |
| `judge_answer` | `{"cmd":"judge_answer","index":0-3}` | Submit answer for current judgment round |
| `reset` | `{"cmd":"reset","seed":123}` | Reset all models with new seed |

### Server → Client

```jsonc
// After whisper / each evolve iteration
{"type":"round","epoch":int,"object":{"s":0-3,"c":0-3},
 "message":[int,int],"listener_guess":{"s":int,"c":int},"rival_guess":{"s":int,"c":int}}

// After every training step
{"type":"stats","epoch":int,"listener_acc":0-1,"rival_acc":0-1,
 "consistency":0-1,"entropy":float,"topsim":float,
 "lexicon":[{"meaning":0-7,"symbol":int,"position":0|1,"stability":0-1},...]}

// After judge_start and after each judge_answer (next round)
{"type":"judge_round","index":1-8,"message":[int,int],
 "options":[{"s":int,"c":int}, ...] /* 4 options */}

// After judge_answer
{"type":"judge_result","correct":bool,"truth":{"s":int,"c":int},
 "rival_guess":{"s":int,"c":int},"rival_correct":bool,
 "you":int,"rival":int,"finished":bool}
```

---

## Model Details

| Component | Architecture |
|-----------|-------------|
| **Speaker** | Linear(8→128)→ReLU→Linear(128→128)→ReLU; head₁: Linear(128→16); sym_embed: Embedding(16,32); head₂: Linear(160→16) |
| **Listener** | Embedding(16,32)×2→flatten(64); Linear(64→128)→ReLU→Linear(128→128)→ReLU; shape_head(128→4); color_head(128→4) |
| **Rival** | Same as Listener, lr × 0.2 |

Training:
- **REINFORCE** with EMA baseline (α=0.05) + entropy bonus (annealed 0.10→0.01 over 1500 epochs)  
- **Gumbel-softmax** alternative (`--gumbel`), temperature annealed 2.0→0.5 over 1000 epochs  
- Partial reward: `(shape_ok + color_ok) / 2 ∈ {0, 0.5, 1}` for smoother gradients  
- Batch size: 64 objects per epoch  
- Base LR: 3e-3 (Adam)

---

## Acceptance Criterion

With `--seed 0`, after 2000 epochs:

- **Listener accuracy ≥ 95%**
- **Rival accuracy clearly lower** (gap > 10%)
- Run `python -m backend.train --epochs 2000 --seed 0 --plot` to verify

---

## Lexicon Stats

The `stats.lexicon` array has 8 entries (meanings 0-3 = shapes, 4-7 = colors).  
Each entry:

```json
{"meaning": 0, "symbol": 5, "position": 0, "stability": 0.92}
```

- `meaning`: 0-3 shape index, 4-7 color index  
- `symbol`: dominant rune ID (0-15) for this meaning in the given position  
- `position`: 0 = first symbol, 1 = second symbol  
- `stability`: P(dominant symbol | meaning) over the sliding window (window=200)
