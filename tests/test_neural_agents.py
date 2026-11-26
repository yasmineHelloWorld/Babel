"""Tests for PyTorch neural Speaker/Listener agents and training modes."""

import unittest

import numpy as np
import torch

from babel.agents.neural import (
    NeuralListener,
    NeuralSpeaker,
    NeuralTrainingConfig,
    train_neural_agents,
)
from babel.datasets.synthetic import generate_synthetic_split, generate_synthetic_world
from babel.metrics import compute_language_metrics, positional_disentanglement
from babel.types import ListenerAgent, SpeakerAgent


class NeuralAgentTests(unittest.TestCase):
    def test_protocol_and_strict_information_boundary(self):
        speaker = NeuralSpeaker(input_dim=6, vocab_size=3, message_length=2, hidden_dim=16, seed=0)
        listener = NeuralListener(
            vocab_size=3, message_length=2, target_cardinalities=(3, 3), hidden_dim=16, mode="holistic", seed=0
        )
        self.assertIsInstance(speaker, SpeakerAgent)
        self.assertIsInstance(listener, ListenerAgent)

        # Listener must reject float tensors/arrays (raw features or Speaker probabilities).
        with self.assertRaises(TypeError):
            listener.predict(np.array([[0.2, 0.8]], dtype=np.float32))
        with self.assertRaises(TypeError):
            listener.forward_logits(torch.tensor([[0.2, 0.8]], dtype=torch.float32))
        with self.assertRaises(ValueError):
            listener.predict(np.array([[0, 5]], dtype=np.int64))

    def test_gradients_reach_speaker_parameters(self):
        speaker = NeuralSpeaker(input_dim=6, vocab_size=3, message_length=2, hidden_dim=16, seed=42)
        xb = torch.randn(8, 6)
        msgs, log_probs, entropy = speaker.sample_messages(xb)
        self.assertEqual(msgs.dtype, torch.int64)
        loss = -(log_probs + 0.05 * entropy).mean()
        loss.backward()
        grad_norm = sum(p.grad.abs().sum().item() for p in speaker.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)

    def test_neural_training_is_reproducible_across_identical_seeds(self):
        world = generate_synthetic_world()
        x = np.stack([obj["features"] for obj in world["objects"]])
        y = np.array([[obj["shape_id"], obj["color_id"]] for obj in world["objects"]], dtype=np.int64)
        cfg = NeuralTrainingConfig(
            vocab_size=3, message_length=2, hidden_dim=32, embed_dim=16, epochs=40, batch_size=9, log_every=20
        )
        run_a = train_neural_agents(x, y, config=cfg, seed=7)
        run_b = train_neural_agents(x, y, config=cfg, seed=7)
        np.testing.assert_array_equal(run_a["train_eval"]["messages"], run_b["train_eval"]["messages"])
        self.assertAlmostEqual(run_a["train_eval"]["exact_accuracy"], run_b["train_eval"]["exact_accuracy"])

    def test_neural_agents_solve_synthetic_world_and_structured_generalizes_zero_shot(self):
        split = generate_synthetic_split(held_out_pairs=((2, 0),))
        x_train = np.stack([obj["features"] for obj in split["train_objects"]])
        y_train = np.array([[obj["shape_id"], obj["color_id"]] for obj in split["train_objects"]], dtype=np.int64)
        x_held = np.stack([obj["features"] for obj in split["held_out_objects"]])
        y_held = np.array([[obj["shape_id"], obj["color_id"]] for obj in split["held_out_objects"]], dtype=np.int64)

        cfg = NeuralTrainingConfig(
            vocab_size=3,
            message_length=2,
            hidden_dim=64,
            embed_dim=32,
            listener_mode="structured",
            training_mode="hybrid_ce",
            epochs=600,
            batch_size=32,
            speaker_lr=5e-3,
            listener_lr=5e-3,
            entropy_bonus=0.05,
            log_every=300,
        )
        # Try across a few seeds to ensure reliable compositional emergence,
        # checking both train convergence and above-chance held-out generalization.
        best_held_attr = 0.0
        best_train_acc = 0.0
        best_posdis = 0.0
        for seed in (0, 1, 2):
            result = train_neural_agents(x_train, y_train, config=cfg, seed=seed)
            train_acc = result["train_eval"]["exact_accuracy"]
            held_msgs = result["speaker"].act(x_held, deterministic=True)
            held_probs = result["listener"].predict(held_msgs)
            held_preds = np.column_stack([np.argmax(p, axis=-1) for p in held_probs])
            held_attr = float(np.mean(held_preds == y_held))
            all_x = np.stack([obj["features"] for obj in split["objects"]])
            all_y = np.array([[obj["shape_id"], obj["color_id"]] for obj in split["objects"]], dtype=np.int64)
            all_msgs = result["speaker"].act(all_x, deterministic=True)
            posdis = positional_disentanglement(all_msgs, all_y)
            best_train_acc = max(best_train_acc, train_acc)
            best_held_attr = max(best_held_attr, held_attr)
            best_posdis = max(best_posdis, posdis)

        self.assertGreaterEqual(best_train_acc, 0.85)
        self.assertGreater(best_held_attr, 1 / 3)
        self.assertGreater(best_posdis, 0.5)

    def test_both_reinforce_ablation_learns_above_chance(self):
        world = generate_synthetic_world()
        x = np.stack([obj["features"] for obj in world["objects"]])
        y = np.array([[obj["shape_id"], obj["color_id"]] for obj in world["objects"]], dtype=np.int64)
        cfg = NeuralTrainingConfig(
            vocab_size=3,
            message_length=2,
            hidden_dim=48,
            embed_dim=24,
            listener_mode="holistic",
            training_mode="both_reinforce",
            epochs=600,
            batch_size=36,
            speaker_lr=6e-3,
            listener_lr=6e-3,
            entropy_bonus=0.04,
            log_every=300,
        )
        result = train_neural_agents(x, y, config=cfg, seed=1)
        # Chance exact accuracy on 3x3 is 1/9 = 0.111; verify genuine learning.
        self.assertGreater(result["train_eval"]["exact_accuracy"], 0.4)
        lang = compute_language_metrics(result["train_eval"]["messages"], y[:, 0], y[:, 1])
        self.assertGreater(lang["active_vocabulary"], 1)


if __name__ == "__main__":
    unittest.main()