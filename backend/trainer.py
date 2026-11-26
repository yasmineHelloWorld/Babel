"""Babel training orchestrator.

One *epoch* = one gradient update on a batch of BATCH_SIZE random objects.
One *round* = forward-only sample of one object through the current model,
              suitable for UI animation.

Training objective
------------------
Speaker : REINFORCE with EMA baseline + entropy bonus.
          Reward = (shape_correct + colour_correct) / 2  ∈ {0, 0.5, 1}.
          Partial reward gives smoother gradients than strict AND.
          Optional: Gumbel-softmax straight-through (--gumbel flag).

Listener : Cross-entropy loss on shape and colour heads independently.

Rival    : Same architecture as Listener, same CE loss,
           but its optimizer uses rival_lr_mult × base_lr
           so it learns the Speaker's dialect more slowly.
"""
from __future__ import annotations

import json
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import (
    NUM_COLORS, NUM_SHAPES, WorldObject,
    all_objects, obj_to_tensor, random_object,
)
from .lexicon import LexiconTracker
from .model import HIDDEN, MSG_LEN, SYM_EMBED, VOCAB, Listener, Speaker

log = logging.getLogger(__name__)

# ── Training hyper-parameters ────────────────────────────────────────────────
BATCH_SIZE: int = 64
BASE_LR: float = 3e-3
ENTROPY_START: float = 0.10   # high exploration early
ENTROPY_END: float   = 0.01   # low after anneal
ENTROPY_ANNEAL: int  = 1500   # epochs to reach ENTROPY_END
EMA_ALPHA: float = 0.05       # EMA speed for REINFORCE baseline
GUMBEL_TAU_START: float = 2.0
GUMBEL_TAU_END: float   = 0.5
GUMBEL_ANNEAL: int      = 1000


@dataclass
class TrainStats:
    epoch: int
    listener_acc: float
    rival_acc: float
    entropy: float
    topsim: float
    consistency: float
    lexicon: list[dict]


# ── Main trainer ─────────────────────────────────────────────────────────────

class BabelTrainer:
    """Holds models, optimizers, and all mutable training state."""

    def __init__(
        self,
        seed: int = 42,
        use_gumbel: bool = False,
        rival_lr_mult: float = 0.2,
        log_dir: Path | None = None,
    ) -> None:
        self._seed = seed
        self.use_gumbel = use_gumbel
        self.rival_lr_mult = rival_lr_mult
        self.epoch = 0
        self.baseline = 0.5     # REINFORCE EMA baseline

        self._all_objs: list[WorldObject] = all_objects()
        # Canonical input tensors for the 16 objects – used for topsim
        self._all_X = torch.stack([obj_to_tensor(o) for o in self._all_objs])  # (16, 8)

        self._seed_everything(seed)
        self._build_models()

        self.lexicon = LexiconTracker(vocab=VOCAB, msg_len=MSG_LEN, window=200)
        self._listener_hist: deque[float] = deque(maxlen=50)
        self._rival_hist: deque[float] = deque(maxlen=50)

        # Run log
        if log_dir is None:
            log_dir = Path("runs")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        self._logfile = (log_dir / f"{ts}_seed{seed}.jsonl").open("w", encoding="utf-8")

        # Judge session (set by start_judge)
        self._judge: _JudgeSession | None = None

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self.epoch = 0
        self.baseline = 0.5
        self._seed_everything(self._seed)
        self._build_models()
        self.lexicon.reset()
        self._listener_hist.clear()
        self._rival_hist.clear()
        self._judge = None

    # ── Training steps ───────────────────────────────────────────────────────

    def train_step(self) -> TrainStats:
        """One gradient update on a random batch. Thread-safe (GIL holds)."""
        indices = self._rng.integers(len(self._all_objs), size=BATCH_SIZE)
        objects = [self._all_objs[int(i)] for i in indices]
        x      = torch.stack([obj_to_tensor(o) for o in objects])          # (B, 8)
        shapes = torch.tensor([o.s for o in objects], dtype=torch.long)
        colors = torch.tensor([o.c for o in objects], dtype=torch.long)

        tau = self._gumbel_tau()

        # ── Speaker forward ───────────────────────────────────────────────
        messages, log_probs, entropy, soft_msgs = self.speaker(
            x, gumbel=self.use_gumbel, tau=tau
        )

        # ── Listener forward (for CE update + reward signal) ─────────────
        sl, cl = self.listener(messages)   # messages has no grad → clean separation

        # ── Rival forward (inference only here, separate update below) ────
        with torch.no_grad():
            rsl, rcl = self.rival(messages)

        # ── Listener update (cross-entropy) ──────────────────────────────
        loss_li = F.cross_entropy(sl, shapes) + F.cross_entropy(cl, colors)
        self.opt_li.zero_grad()
        loss_li.backward()
        torch.nn.utils.clip_grad_norm_(self.listener.parameters(), 1.0)
        self.opt_li.step()

        # ── Rival update (same CE, slower LR, separate forward) ──────────
        rsl_r, rcl_r = self.rival(messages.detach())
        loss_rv = F.cross_entropy(rsl_r, shapes) + F.cross_entropy(rcl_r, colors)
        self.opt_rv.zero_grad()
        loss_rv.backward()
        torch.nn.utils.clip_grad_norm_(self.rival.parameters(), 1.0)
        self.opt_rv.step()

        # ── Speaker update ────────────────────────────────────────────────
        ec = self._entropy_coeff()
        if self.use_gumbel:
            # Differentiable path: forward through listener with soft one-hots.
            # Freeze listener weights so only speaker grads accumulate.
            for p in self.listener.parameters():
                p.requires_grad_(False)
            sl_soft, cl_soft = self.listener(messages, soft=soft_msgs)
            loss_sp = (
                F.cross_entropy(sl_soft, shapes) +
                F.cross_entropy(cl_soft, colors) -
                ec * entropy.mean()
            )
            self.opt_sp.zero_grad()
            loss_sp.backward()
            torch.nn.utils.clip_grad_norm_(self.speaker.parameters(), 1.0)
            self.opt_sp.step()
            for p in self.listener.parameters():
                p.requires_grad_(True)
        else:
            # REINFORCE with partial reward and EMA baseline
            with torch.no_grad():
                reward = (
                    (sl.argmax(-1) == shapes).float() +
                    (cl.argmax(-1) == colors).float()
                ) / 2.0                                           # ∈ {0, 0.5, 1}
            self.baseline = (
                (1.0 - EMA_ALPHA) * self.baseline +
                EMA_ALPHA * float(reward.mean())
            )
            advantage = (reward - self.baseline).detach()
            loss_sp = -(log_probs * advantage).mean() - ec * entropy.mean()
            self.opt_sp.zero_grad()
            loss_sp.backward()
            torch.nn.utils.clip_grad_norm_(self.speaker.parameters(), 1.0)
            self.opt_sp.step()

        # ── Metrics ───────────────────────────────────────────────────────
        with torch.no_grad():
            li_acc = float(
                ((sl.argmax(-1) == shapes) & (cl.argmax(-1) == colors)).float().mean()
            )
            rv_acc = float(
                ((rsl.argmax(-1) == shapes) & (rcl.argmax(-1) == colors)).float().mean()
            )
        self._listener_hist.append(li_acc)
        self._rival_hist.append(rv_acc)
        self.epoch += 1

        for obj, msg in zip(objects, messages.tolist()):
            self.lexicon.update(obj.s, obj.c, msg)

        stats = self._make_stats(float(entropy.mean().detach()))
        self._log(stats)
        return stats

    def skip(self, n: int) -> TrainStats:
        """Train n steps silently, return final stats."""
        for _ in range(n):
            self.train_step()
        return self._make_stats()

    # ── Round sampling (UI) ──────────────────────────────────────────────────

    def sample_round(self) -> dict:
        """
        Sample one object, run speaker (stochastic) and both listeners.
        No gradient computation.  Returns the payload for a 'round' WS message.
        """
        obj = random_object(self._rng)
        x = obj_to_tensor(obj).unsqueeze(0)
        with torch.no_grad():
            messages, _, _, _ = self.speaker(x)
            sl, cl = self.listener(messages)
            rsl, rcl = self.rival(messages)
        msg = messages[0].tolist()
        return {
            "epoch":          self.epoch,
            "object":         {"s": obj.s, "c": obj.c},
            "message":        msg,
            "listener_guess": {"s": int(sl[0].argmax()), "c": int(cl[0].argmax())},
            "rival_guess":    {"s": int(rsl[0].argmax()), "c": int(rcl[0].argmax())},
        }

    # ── Judge session ────────────────────────────────────────────────────────

    def start_judge(self) -> dict:
        """Begin an 8-round judgment session.  Returns the first judge_round dict."""
        self._judge = _JudgeSession(self)
        return self._judge.next_round()

    def answer_judge(self, option_index: int) -> dict:
        """Human answers; returns judge_result dict."""
        if self._judge is None:
            raise RuntimeError("No active judge session — call start_judge() first.")
        return self._judge.answer(option_index)

    def judge_next_round(self) -> dict | None:
        """Return the next judge_round dict, or None if 8 rounds exhausted."""
        if self._judge is None or self._judge.finished:
            return None
        return self._judge.next_round()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _seed_everything(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def _build_models(self) -> None:
        self.speaker  = Speaker()
        self.listener = Listener()
        self.rival    = Listener()
        self.opt_sp = torch.optim.Adam(self.speaker.parameters(),  lr=BASE_LR)
        self.opt_li = torch.optim.Adam(self.listener.parameters(), lr=BASE_LR)
        self.opt_rv = torch.optim.Adam(self.rival.parameters(),    lr=BASE_LR * self.rival_lr_mult)

    def _entropy_coeff(self) -> float:
        t = min(1.0, self.epoch / max(1, ENTROPY_ANNEAL))
        return ENTROPY_START + (ENTROPY_END - ENTROPY_START) * t

    def _gumbel_tau(self) -> float:
        t = min(1.0, self.epoch / max(1, GUMBEL_ANNEAL))
        return GUMBEL_TAU_START + (GUMBEL_TAU_END - GUMBEL_TAU_START) * t

    def _make_stats(self, entropy: float | None = None) -> TrainStats:
        li_acc = float(np.mean(self._listener_hist)) if self._listener_hist else 0.0
        rv_acc = float(np.mean(self._rival_hist))    if self._rival_hist    else 0.0
        if entropy is None:
            with torch.no_grad():
                _, _, ent, _ = self.speaker(self._all_X[:8])
                entropy = float(ent.mean())
        return TrainStats(
            epoch=self.epoch,
            listener_acc=li_acc,
            rival_acc=rv_acc,
            entropy=entropy,
            topsim=self._topsim(),
            consistency=self.lexicon.consistency(),
            lexicon=self.lexicon.stats(),
        )

    def _topsim(self) -> float:
        """Topographic similarity: Pearson r between meaning and message Hamming distances."""
        with torch.no_grad():
            msgs = self.speaker.predict(self._all_X)   # (16, 2) int
        n = len(self._all_objs)
        od, md = [], []
        for i in range(n):
            for j in range(i + 1, n):
                oi, oj = self._all_objs[i], self._all_objs[j]
                od.append((oi.s != oj.s) + (oi.c != oj.c))                 # 0-2
                md.append(sum(int(msgs[i, k]) != int(msgs[j, k]) for k in range(MSG_LEN)))
        od_a = np.array(od, dtype=float)
        md_a = np.array(md, dtype=float)
        if od_a.std() < 1e-9 or md_a.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(od_a, md_a)[0, 1])

    def _log(self, stats: TrainStats) -> None:
        try:
            self._logfile.write(json.dumps({
                "epoch":        stats.epoch,
                "listener_acc": stats.listener_acc,
                "rival_acc":    stats.rival_acc,
                "entropy":      stats.entropy,
                "topsim":       stats.topsim,
                "consistency":  stats.consistency,
            }) + "\n")
            self._logfile.flush()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self._logfile.close()
        except Exception:
            pass


# ── Judge session state machine ───────────────────────────────────────────────

class _JudgeSession:
    """State machine for a single 8-round human-vs-rival judgment."""

    NUM_ROUNDS  = 8
    NUM_OPTIONS = 4

    def __init__(self, trainer: BabelTrainer) -> None:
        self._t = trainer
        self.round_index = 0
        self.you   = 0
        self.rival = 0
        self._pending: dict | None = None   # current round data

    @property
    def finished(self) -> bool:
        return self.round_index >= self.NUM_ROUNDS

    def next_round(self) -> dict:
        """Sample a new judgment round; return the judge_round payload."""
        self.round_index += 1
        obj = random_object(self._t._rng)
        x   = obj_to_tensor(obj).unsqueeze(0)
        with torch.no_grad():
            messages, _, _, _ = self._t.speaker(x)
        msg = messages[0].tolist()

        # 3 distinct distractors
        seen = {(obj.s, obj.c)}
        distractors: list[WorldObject] = []
        while len(distractors) < self.NUM_OPTIONS - 1:
            d = random_object(self._t._rng)
            if (d.s, d.c) not in seen:
                distractors.append(d)
                seen.add((d.s, d.c))

        candidates = [obj] + distractors          # target always at index 0
        indices = np.arange(self.NUM_OPTIONS)
        self._t._rng.shuffle(indices)
        options = [candidates[i] for i in indices]
        correct_idx = int(np.where(indices == 0)[0][0])

        self._pending = {
            "obj":         obj,
            "options":     options,
            "correct_idx": correct_idx,
            "message":     msg,
        }
        return {
            "type":    "judge_round",
            "index":   self.round_index,
            "message": msg,
            "options": [{"s": o.s, "c": o.c} for o in options],
        }

    def answer(self, option_index: int) -> dict:
        """Human answers; compute rival response; return judge_result."""
        if self._pending is None:
            raise RuntimeError("Call next_round() before answer().")
        p = self._pending
        self._pending = None
        obj         = p["obj"]
        options     = p["options"]
        correct_idx = p["correct_idx"]

        human_correct = (option_index == correct_idx)

        # Rival listener inference
        x = obj_to_tensor(obj).unsqueeze(0)
        with torch.no_grad():
            messages, _, _, _ = self._t.speaker(x)
            rsl, rcl = self._t.rival(messages)
        rival_shape = int(rsl[0].argmax())
        rival_color = int(rcl[0].argmax())
        rival_correct = (rival_shape == obj.s and rival_color == obj.c)

        if human_correct:
            self.you   += 1
        if rival_correct:
            self.rival += 1

        return {
            "type":          "judge_result",
            "index":         self.round_index,
            "correct":       human_correct,
            "truth":         {"s": obj.s, "c": obj.c},
            "rival_guess":   {"s": rival_shape, "c": rival_color},
            "rival_correct": rival_correct,
            "you":           self.you,
            "rival":         self.rival,
            "finished":      self.finished,
        }
