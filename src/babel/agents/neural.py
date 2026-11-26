"""PyTorch neural Speaker and Listener agents with strict discrete channel boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from babel.types import assert_discrete_messages


def _set_torch_seed(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


class NeuralSpeaker(nn.Module):
    """MLP Speaker observing input features x and emitting discrete symbols in {0..K-1}^L."""

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        message_length: int,
        *,
        hidden_dim: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or vocab_size < 2 or message_length < 1:
            raise ValueError("input_dim >= 1, vocab_size >= 2, and message_length >= 1 required")
        self.input_dim = int(input_dim)
        self.vocab_size = int(vocab_size)
        self.message_length = int(message_length)
        self.hidden_dim = int(hidden_dim)
        self._rng = _set_torch_seed(seed)

        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.vocab_size) for _ in range(self.message_length)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return message logits of shape (batch, message_length, vocab_size)."""
        hidden = self.trunk(features)
        return torch.stack([head(hidden) for head in self.heads], dim=1)

    def sample_messages(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample discrete messages and return (messages_int, sum_log_prob, mean_entropy)."""
        logits = self.forward(features)
        messages = []
        log_probs = []
        entropies = []
        for pos in range(self.message_length):
            dist = Categorical(logits=logits[:, pos, :])
            action = dist.sample()
            messages.append(action)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
        msg_tensor = torch.stack(messages, dim=1).to(torch.int64)
        total_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)
        mean_entropy = torch.stack(entropies, dim=1).mean(dim=1)
        return msg_tensor, total_log_prob, mean_entropy

    @torch.no_grad()
    def act(self, features: np.ndarray, *, deterministic: bool = False) -> np.ndarray:
        """SpeakerAgent.act: return discrete integer message IDs in {0..K-1}^L."""
        arr = np.asarray(features, dtype=np.float32)
        single = arr.ndim == 1
        if single:
            arr = arr[np.newaxis, :]
        was_training = self.training
        self.eval()
        try:
            x_tensor = torch.from_numpy(arr)
            logits = self.forward(x_tensor)
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = torch.stack(
                    [Categorical(logits=logits[:, pos, :]).sample() for pos in range(self.message_length)],
                    dim=1,
                )
            result = actions.cpu().numpy().astype(np.int64)
        finally:
            self.train(was_training)
        return result[0] if single else result


class NeuralListener(nn.Module):
    """Listener receiving ONLY discrete integer message IDs m — never raw features or logits.

    Supports two architectures:
    - ``holistic``: concatenates per-position symbol embeddings into a shared MLP.
    - ``structured``: factored additive per-position projections with no cross-position
      nonlinear interaction, enabling zero-shot compositional generalization.
    """

    def __init__(
        self,
        vocab_size: int,
        message_length: int,
        target_cardinalities: Sequence[int],
        *,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        mode: str = "holistic",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or message_length < 1 or not target_cardinalities:
            raise ValueError("Invalid vocab_size, message_length, or target_cardinalities")
        if mode not in {"holistic", "structured"}:
            raise ValueError("mode must be 'holistic' or 'structured'")
        self.vocab_size = int(vocab_size)
        self.message_length = int(message_length)
        self.target_cardinalities = tuple(int(c) for c in target_cardinalities)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.mode = mode
        self._rng = _set_torch_seed(seed)

        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.vocab_size, self.embed_dim) for _ in range(self.message_length)]
        )
        if self.mode == "holistic":
            self.decoder_trunk = nn.Sequential(
                nn.Linear(self.message_length * self.embed_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
            self.heads = nn.ModuleList(
                [nn.Linear(self.hidden_dim, card) for card in self.target_cardinalities]
            )
        else:
            # Factored per-position linear maps summed per attribute head: zero cross-term
            # memorization so compositional messages generalize to unseen combinations.
            self.position_projections = nn.ModuleList(
                [
                    nn.ModuleList([nn.Linear(self.embed_dim, card) for card in self.target_cardinalities])
                    for _ in range(self.message_length)
                ]
            )

    def forward_logits(self, messages: torch.Tensor) -> list[torch.Tensor]:
        """Accept integer tensor (batch, L) ONLY and return per-attribute logit tensors."""
        if messages.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(f"Listener requires integer message IDs, got {messages.dtype}")
        if messages.ndim != 2 or messages.shape[1] != self.message_length:
            raise ValueError(f"Expected message shape (batch, {self.message_length}), got {tuple(messages.shape)}")
        if torch.any(messages < 0) or torch.any(messages >= self.vocab_size):
            raise ValueError(f"Message IDs must be in [0, {self.vocab_size - 1}]")

        pos_embeds = [self.embeddings[pos](messages[:, pos].long()) for pos in range(self.message_length)]
        if self.mode == "holistic":
            joined = torch.cat(pos_embeds, dim=-1)
            hidden = self.decoder_trunk(joined)
            return [head(hidden) for head in self.heads]

        outputs: list[torch.Tensor] = []
        for attr_idx in range(len(self.target_cardinalities)):
            attr_logits = sum(
                self.position_projections[pos][attr_idx](pos_embeds[pos])
                for pos in range(self.message_length)
            )
            outputs.append(attr_logits)
        return outputs

    @torch.no_grad()
    def predict(self, messages: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """ListenerAgent.predict: predict probability distribution(s) from discrete messages only."""
        arr = assert_discrete_messages(
            messages, vocab_size=self.vocab_size, message_length=self.message_length
        )
        single = arr.ndim == 1
        if single:
            arr = arr[np.newaxis, :]
        was_training = self.training
        self.eval()
        try:
            msg_tensor = torch.from_numpy(arr.astype(np.int64))
            logits_list = self.forward_logits(msg_tensor)
            probs = tuple(F.softmax(logits, dim=-1).cpu().numpy() for logits in logits_list)
        finally:
            self.train(was_training)
        if single:
            probs = tuple(p[0] for p in probs)
        return probs[0] if len(probs) == 1 else probs


@dataclass(frozen=True)
class NeuralTrainingConfig:
    vocab_size: int = 4
    message_length: int = 1
    hidden_dim: int = 64
    embed_dim: int = 32
    listener_mode: str = "holistic"
    training_mode: str = "hybrid_ce"
    epochs: int = 120
    batch_size: int = 64
    speaker_lr: float = 3e-3
    listener_lr: float = 5e-3
    entropy_bonus: float = 0.02
    listener_entropy_bonus: float = 0.01
    baseline_ema: float = 0.05
    log_every: int = 10


def _normalize_targets(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.int64)
    return arr[:, np.newaxis] if arr.ndim == 1 else arr


def _evaluate_neural_split(
    speaker: NeuralSpeaker,
    listener: NeuralListener,
    features: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Evaluate Speaker -> discrete messages -> Listener strictly across the information boundary."""
    messages = speaker.act(features, deterministic=True)
    probs = listener.predict(messages)
    if not isinstance(probs, tuple):
        probs = (probs,)
    preds = np.column_stack([np.argmax(p, axis=-1) for p in probs])
    exact_acc = float(np.mean(np.all(preds == targets, axis=1)))
    per_attr_acc = [float(np.mean(preds[:, i] == targets[:, i])) for i in range(targets.shape[1])]
    active_vocab = len(set(int(s) for s in messages.flat))
    return {
        "exact_accuracy": exact_acc,
        "per_attribute_accuracy": per_attr_acc,
        "messages": messages,
        "preds": preds,
        "probs": probs,
        "active_vocabulary": active_vocab,
    }


def train_neural_agents(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    val_features: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    target_cardinalities: Sequence[int] | None = None,
    config: NeuralTrainingConfig = NeuralTrainingConfig(),
    seed: int = 0,
    metrics_callback=None,
) -> dict:
    """Train NeuralSpeaker and NeuralListener with hybrid_ce or both_reinforce."""
    if config.training_mode not in {"hybrid_ce", "both_reinforce"}:
        raise ValueError("training_mode must be 'hybrid_ce' or 'both_reinforce'")
    x_train = np.asarray(train_features, dtype=np.float32)
    y_train = _normalize_targets(train_labels)
    if target_cardinalities is None:
        target_cardinalities = tuple(int(y_train[:, col].max() + 1) for col in range(y_train.shape[1]))

    streams = np.random.SeedSequence(seed).spawn(3)
    env_rng = np.random.default_rng(streams[0])
    speaker_seed = int(np.random.default_rng(streams[1]).integers(0, 2**31 - 1))
    listener_seed = int(np.random.default_rng(streams[2]).integers(0, 2**31 - 1))

    speaker = NeuralSpeaker(
        x_train.shape[1],
        config.vocab_size,
        config.message_length,
        hidden_dim=config.hidden_dim,
        seed=speaker_seed,
    )
    listener = NeuralListener(
        config.vocab_size,
        config.message_length,
        target_cardinalities,
        hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim,
        mode=config.listener_mode,
        seed=listener_seed,
    )

    speaker_opt = torch.optim.Adam(speaker.parameters(), lr=config.speaker_lr)
    listener_opt = torch.optim.Adam(listener.parameters(), lr=config.listener_lr)
    baseline = 0.0
    n_samples = len(x_train)
    history: list[dict] = []

    y_val = _normalize_targets(val_labels) if val_labels is not None else None
    x_val = np.asarray(val_features, dtype=np.float32) if val_features is not None else None

    for epoch in range(1, config.epochs + 1):
        speaker.train()
        listener.train()
        # On tiny datasets (e.g., 3x3 world with 8-9 objects), repeating rows within
        # the epoch gives REINFORCE multiple Monte-Carlo draws per gradient step.
        repeat_factor = max(1, (32 + n_samples - 1) // n_samples) if n_samples < 32 else 1
        order = np.concatenate([env_rng.permutation(n_samples) for _ in range(repeat_factor)])
        epoch_rewards, epoch_spk_losses, epoch_lst_losses = [], [], []
        progress = (epoch - 1) / max(1, config.epochs - 1)
        entropy_weight = config.entropy_bonus * (1.0 - 0.75 * progress)

        for start in range(0, len(order), config.batch_size):
            batch_idx = order[start : start + config.batch_size]
            xb = torch.from_numpy(x_train[batch_idx])
            yb = torch.from_numpy(y_train[batch_idx])

            # Speaker generates discrete messages (non-differentiable channel).
            messages, spk_log_prob, spk_entropy = speaker.sample_messages(xb)
            discrete_messages = messages.detach()

            # Listener receives ONLY discrete integer messages.
            logits_list = listener.forward_logits(discrete_messages)

            if config.training_mode == "hybrid_ce":
                listener_loss = sum(
                    F.cross_entropy(logits, yb[:, attr_idx])
                    for attr_idx, logits in enumerate(logits_list)
                ) / len(logits_list)
                listener_opt.zero_grad()
                listener_loss.backward()
                listener_opt.step()

                with torch.no_grad():
                    per_attr_prob = [
                        F.softmax(logits, dim=-1).gather(1, yb[:, attr_idx : attr_idx + 1]).squeeze(1)
                        for attr_idx, logits in enumerate(logits_list)
                    ]
                    reward = torch.stack(per_attr_prob, dim=1).mean(dim=1)
            else:
                # Both-REINFORCE ablation: Listener also samples discrete class predictions.
                lst_hits = []
                lst_terms = []
                for attr_idx, logits in enumerate(logits_list):
                    dist = Categorical(logits=logits)
                    pred_action = dist.sample()
                    hit = (pred_action == yb[:, attr_idx]).to(torch.float32)
                    lst_hits.append(hit)
                    attr_adv = hit - hit.mean().detach()
                    lst_terms.append(attr_adv * dist.log_prob(pred_action) + config.listener_entropy_bonus * (1.0 - 0.75 * progress) * dist.entropy())
                reward = torch.stack(lst_hits, dim=1).mean(dim=1)
                listener_loss = -torch.stack(lst_terms, dim=1).mean()
                listener_opt.zero_grad()
                listener_loss.backward()
                listener_opt.step()

            mean_reward = float(reward.mean().item())
            advantage = reward - baseline
            baseline = (1.0 - config.baseline_ema) * baseline + config.baseline_ema * mean_reward

            speaker_loss = -(advantage * spk_log_prob + entropy_weight * spk_entropy).mean()
            speaker_opt.zero_grad()
            speaker_loss.backward()
            speaker_opt.step()

            epoch_rewards.append(mean_reward)
            epoch_spk_losses.append(float(speaker_loss.item()))
            epoch_lst_losses.append(float(listener_loss.item()))

        if epoch % config.log_every == 0 or epoch == config.epochs:
            train_eval = _evaluate_neural_split(speaker, listener, x_train, y_train)
            row = {
                "epoch": epoch,
                "reward": float(np.mean(epoch_rewards)),
                "baseline": float(baseline),
                "speaker_loss": float(np.mean(epoch_spk_losses)),
                "listener_loss": float(np.mean(epoch_lst_losses)),
                "train_accuracy": train_eval["exact_accuracy"],
                "active_vocabulary": train_eval["active_vocabulary"],
            }
            if x_val is not None and y_val is not None:
                val_eval = _evaluate_neural_split(speaker, listener, x_val, y_val)
                row["val_accuracy"] = val_eval["exact_accuracy"]
            history.append(row)
            if metrics_callback is not None:
                metrics_callback(row)

    final_train = _evaluate_neural_split(speaker, listener, x_train, y_train)
    result = {
        "speaker": speaker,
        "listener": listener,
        "history": history,
        "train_eval": final_train,
    }
    if x_val is not None and y_val is not None:
        result["val_eval"] = _evaluate_neural_split(speaker, listener, x_val, y_val)
    return result


def save_neural_checkpoint(path: str | Path, speaker: NeuralSpeaker, listener: NeuralListener) -> Path:
    """Save neural agent state dicts and architecture metadata."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "speaker_state": speaker.state_dict(),
            "listener_state": listener.state_dict(),
            "speaker_meta": {
                "input_dim": speaker.input_dim,
                "vocab_size": speaker.vocab_size,
                "message_length": speaker.message_length,
                "hidden_dim": speaker.hidden_dim,
            },
            "listener_meta": {
                "vocab_size": listener.vocab_size,
                "message_length": listener.message_length,
                "target_cardinalities": listener.target_cardinalities,
                "hidden_dim": listener.hidden_dim,
                "embed_dim": listener.embed_dim,
                "mode": listener.mode,
            },
        },
        target,
    )
    return target
