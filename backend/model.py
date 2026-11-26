"""Speaker and Listener neural network models.

Discrete channel constraint (Plan Rule 1):
  The Listener receives ONLY integer symbol IDs.  During the Gumbel-softmax
  training path, gradients flow through one-hot vectors in the Speaker's
  output — the Listener's sym_embed treats these as differentiable look-ups —
  but the *values* that cross the channel are always discrete integers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ── Hyper-parameters (kept as module-level constants so both sides agree) ──
VOCAB: int = 16    # K – number of distinct rune symbols
MSG_LEN: int = 2   # L – runes per message
SYM_EMBED: int = 32
HIDDEN: int = 128


class Speaker(nn.Module):
    """
    Sees an 8-dim one-hot object vector.  Autoregressively emits MSG_LEN
    symbols.  sym₁ is emitted unconditionally; sym₂ conditions on sym₁.

    Returns
    -------
    messages   (B, MSG_LEN)        int64  – discrete symbol IDs
    log_probs  (B,)                float  – Σ log p(sym_i | …)  [None for Gumbel]
    entropy    (B,)                float  – mean per-step entropy
    soft_msgs  (B, MSG_LEN, VOCAB) float  – straight-through one-hots [None for REINFORCE]
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden: int = HIDDEN,
        vocab: int = VOCAB,
        embed: int = SYM_EMBED,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(hidden, vocab)
        self.sym_embed = nn.Embedding(vocab, embed)
        self.head2 = nn.Linear(hidden + embed, vocab)

    def forward(
        self,
        x: torch.Tensor,
        gumbel: bool = False,
        tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        h = self.trunk(x)                       # (B, hidden)
        logits1 = self.head1(h)                 # (B, vocab)

        if gumbel:
            oh1 = F.gumbel_softmax(logits1, tau=tau, hard=True)   # (B, vocab)
            sym1 = oh1.argmax(-1)                                  # (B,) int
            sym1_emb = oh1 @ self.sym_embed.weight                 # differentiable
        else:
            dist1 = Categorical(logits=logits1)
            sym1 = dist1.sample()                                  # (B,)
            sym1_emb = self.sym_embed(sym1)                        # (B, embed)

        logits2 = self.head2(torch.cat([h, sym1_emb], dim=-1))    # (B, vocab)

        if gumbel:
            oh2 = F.gumbel_softmax(logits2, tau=tau, hard=True)
            sym2 = oh2.argmax(-1)
            soft_msgs = torch.stack([oh1, oh2], dim=1)            # (B, MSG_LEN, vocab)
            log_probs = None
        else:
            dist2 = Categorical(logits=logits2)
            sym2 = dist2.sample()
            log_probs = dist1.log_prob(sym1) + dist2.log_prob(sym2)   # (B,)
            soft_msgs = None

        entropy = (
            Categorical(logits=logits1).entropy() +
            Categorical(logits=logits2).entropy()
        ) / 2.0                                                    # (B,)
        messages = torch.stack([sym1, sym2], dim=1)               # (B, MSG_LEN)
        return messages, log_probs, entropy, soft_msgs

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic argmax message – used for topographic similarity."""
        h = self.trunk(x)
        sym1 = self.head1(h).argmax(-1)
        sym1_emb = self.sym_embed(sym1)
        sym2 = self.head2(torch.cat([h, sym1_emb], dim=-1)).argmax(-1)
        return torch.stack([sym1, sym2], dim=1)


class Listener(nn.Module):
    """
    Receives MSG_LEN integer symbol IDs only (discrete channel).
    Predicts shape (4 classes) and colour (4 classes) independently.

    During the Gumbel-softmax training path the caller may pass `soft`
    one-hots so that gradients flow back to the Speaker.
    """

    def __init__(
        self,
        vocab: int = VOCAB,
        msg_len: int = MSG_LEN,
        embed: int = SYM_EMBED,
        hidden: int = HIDDEN,
        num_shapes: int = 4,
        num_colors: int = 4,
    ) -> None:
        super().__init__()
        self.sym_embed = nn.Embedding(vocab, embed)
        self.trunk = nn.Sequential(
            nn.Linear(msg_len * embed, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.shape_head = nn.Linear(hidden, num_shapes)
        self.color_head = nn.Linear(hidden, num_colors)

    def forward(
        self,
        messages: torch.Tensor,
        soft: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        messages : (B, MSG_LEN) int64
        soft     : (B, MSG_LEN, vocab) float one-hots for Gumbel path
        Returns  : (shape_logits (B,4), color_logits (B,4))
        """
        if soft is not None:
            # Differentiable: matmul with embedding weight matrix
            emb = (soft @ self.sym_embed.weight).flatten(1)   # (B, MSG_LEN*embed)
        else:
            emb = self.sym_embed(messages).flatten(1)          # (B, MSG_LEN*embed)
        h = self.trunk(emb)
        return self.shape_head(h), self.color_head(h)
