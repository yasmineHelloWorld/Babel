"""Sliding-window co-occurrence tracker for the emergent lexicon.

Meanings
--------
  0-3  : shapes  0-3
  4-7  : colours 0-3

For each training round (shape=s, colour=c, message=[sym0, sym1]):
  - meaning s   co-occurs with (position=0, sym0) and (position=1, sym1)
  - meaning 4+c co-occurs with (position=0, sym0) and (position=1, sym1)

The dominant (position, symbol) pair is the most frequent one within
the sliding window.  Stability = P(dominant | meaning) in the window.
"""
from __future__ import annotations

from collections import deque

import numpy as np

NUM_MEANINGS: int = 8   # 4 shapes + 4 colours


class LexiconTracker:
    """Thread-safe (GIL) sliding-window co-occurrence tracker."""

    def __init__(
        self,
        vocab: int = 16,
        msg_len: int = 2,
        window: int = 200,
    ) -> None:
        self.vocab = vocab
        self.msg_len = msg_len
        self.window = window
        # counts[meaning, position, symbol]
        self.counts: np.ndarray = np.zeros(
            (NUM_MEANINGS, msg_len, vocab), dtype=np.float32
        )
        # Ring buffer of event-lists for window expiry
        self._buf: deque[list[tuple[int, int, int]]] = deque()

    # ── Public API ──────────────────────────────────────────────────────

    def update(self, shape: int, colour: int, message: list[int]) -> None:
        """Record one communication round."""
        events: list[tuple[int, int, int]] = []
        for pos, sym in enumerate(message):
            events.append((shape,     pos, sym))
            events.append((4 + colour, pos, sym))
        for m, pos, sym in events:
            self.counts[m, pos, sym] += 1.0
        self._buf.append(events)
        # Expire oldest batch if over window size
        while len(self._buf) > self.window:
            for m, pos, sym in self._buf.popleft():
                self.counts[m, pos, sym] = max(0.0, self.counts[m, pos, sym] - 1.0)

    def stats(self) -> list[dict]:
        """
        Return one entry per meaning.

        Each entry:
          meaning   int  0-7
          symbol    int  dominant symbol ID
          position  int  dominant position (0 or 1)
          stability float  P(dominant | meaning) in the window
        """
        result = []
        for m in range(NUM_MEANINGS):
            flat = self.counts[m]           # (msg_len, vocab)
            total = float(flat.sum())
            if total == 0.0:
                result.append({"meaning": m, "symbol": 0, "position": 0, "stability": 0.0})
                continue
            pos_i, sym_i = np.unravel_index(flat.argmax(), flat.shape)
            result.append({
                "meaning":   m,
                "symbol":    int(sym_i),
                "position":  int(pos_i),
                "stability": float(flat[pos_i, sym_i] / total),
            })
        return result

    def consistency(self) -> float:
        """Mean stability across all meanings (0 = random, 1 = perfectly settled)."""
        s = self.stats()
        return float(np.mean([e["stability"] for e in s]))

    def reset(self) -> None:
        self.counts[:] = 0.0
        self._buf.clear()
