import unittest

import numpy as np

from babel.datasets.synthetic import generate_synthetic_split, generate_synthetic_world
from babel.metrics import (
    compute_classification_metrics,
    compute_dialect_matrix,
    compute_language_metrics,
    compute_multi_attribute_metrics,
    positional_disentanglement,
    topographic_similarity,
    topographic_similarity_with_pvalue,
)
from babel.types import Batch, assert_discrete_messages


class LanguageMetricTests(unittest.TestCase):
    def setUp(self):
        self.objects = generate_synthetic_world()["objects"]
        self.shapes = [obj["shape_id"] for obj in self.objects]
        self.colors = [obj["color_id"] for obj in self.objects]

    def test_compositional_oracle_has_known_information_structure(self):
        messages = np.array([[obj["shape_id"], obj["color_id"]] for obj in self.objects])
        metrics = compute_language_metrics(messages, self.shapes, self.colors)

        self.assertAlmostEqual(metrics["message_entropy_bits"], np.log2(9))
        self.assertAlmostEqual(metrics["shape_mutual_information_bits"], np.log2(3))
        self.assertAlmostEqual(metrics["color_mutual_information_bits"], np.log2(3))
        self.assertAlmostEqual(metrics["object_mutual_information_bits"], np.log2(9))
        self.assertAlmostEqual(metrics["position_property_mutual_information_bits"]["0:shape"], np.log2(3))
        self.assertAlmostEqual(metrics["position_property_mutual_information_bits"]["0:color"], 0.0)
        self.assertAlmostEqual(metrics["position_property_mutual_information_bits"]["1:shape"], 0.0)
        self.assertAlmostEqual(metrics["position_property_mutual_information_bits"]["1:color"], np.log2(3))
        self.assertAlmostEqual(metrics["posdis"], 1.0)
        self.assertEqual(metrics["active_vocabulary"], 3)
        self.assertAlmostEqual(metrics["ambiguity_bits"], 0.0)
        self.assertAlmostEqual(metrics["synonymy_bits"], 0.0)

    def test_independent_random_oracle_has_near_zero_label_information(self):
        rng = np.random.default_rng(9182)
        # Independent random symbol draws for every object provide a finite-sample null.
        messages = rng.integers(0, 3, size=(len(self.objects), 2000, 2))
        metrics = compute_language_metrics(messages, self.shapes, self.colors)

        self.assertLess(metrics["shape_mutual_information_bits"], 0.02)
        self.assertLess(metrics["color_mutual_information_bits"], 0.02)
        self.assertLess(metrics["posdis"], 0.02)
        for value in metrics["position_property_mutual_information_bits"].values():
            self.assertLess(value, 0.02)

    def test_stochastic_synonymy_is_positive(self):
        messages = np.zeros((len(self.objects), 2, 2), dtype=int)
        messages[:, 1, 0] = 1
        metrics = compute_language_metrics(messages, self.shapes, self.colors)

        self.assertAlmostEqual(metrics["synonymy_bits"], 1.0)
        self.assertAlmostEqual(metrics["ambiguity_bits"], np.log2(9))
        self.assertAlmostEqual(metrics["object_mutual_information_bits"], 0.0)

    def test_compositional_oracle_has_perfect_topographic_similarity(self):
        messages = np.array([[obj["shape_id"], obj["color_id"]] for obj in self.objects])
        features = np.array([obj["features"] for obj in self.objects])

        self.assertAlmostEqual(topographic_similarity(messages, features), 1.0)
        rho, pval = topographic_similarity_with_pvalue(messages, features, n_permutations=200, seed=0)
        self.assertAlmostEqual(rho, 1.0)
        self.assertLess(pval, 0.02)

    def test_posdis_handles_collapsed_position_and_exact_compositional_code(self):
        attrs = np.column_stack([self.shapes, self.colors])
        compositional = np.column_stack([self.shapes, self.colors])
        self.assertAlmostEqual(positional_disentanglement(compositional, attrs), 1.0)
        collapsed = np.zeros_like(compositional)
        self.assertAlmostEqual(positional_disentanglement(collapsed, attrs), 0.0)

    def test_classification_and_multi_attribute_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2])
        y_proba = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.1, 0.85],
            [0.85, 0.1, 0.05],
            [0.2, 0.3, 0.5],
            [0.05, 0.1, 0.85],
        ])
        clf = compute_classification_metrics(y_true, y_pred, y_proba)
        self.assertAlmostEqual(clf["accuracy"], 5 / 6)
        self.assertGreater(clf["roc_auc_ovr"], 0.9)
        self.assertEqual(len(clf["confusion_matrix"]), 3)

        multi = compute_multi_attribute_metrics(
            np.array([[0, 1], [1, 2], [2, 0]]),
            np.array([[0, 1], [1, 0], [2, 0]]),
        )
        self.assertAlmostEqual(multi["exact_accuracy"], 2 / 3)
        self.assertAlmostEqual(multi["per_attribute_accuracy"][0], 1.0)
        self.assertAlmostEqual(multi["per_attribute_accuracy"][1], 2 / 3)

    def test_dialect_matrix_hungarian_aligns_permuted_isomorphic_codes(self):
        targets = np.column_stack([self.shapes, self.colors])
        msgs_a = np.column_stack([self.shapes, self.colors])
        # Dialect B permutes symbol IDs AND swaps positions: (color + 1 mod 3, shape + 2 mod 3)
        msgs_b = np.column_stack([(np.array(self.colors) + 1) % 3, (np.array(self.shapes) + 2) % 3])

        def listener_a(msgs: np.ndarray) -> np.ndarray:
            return msgs.copy()

        def listener_b(msgs: np.ndarray) -> np.ndarray:
            # L_B expects pos 1 -> shape via inverse (+1 mod 3) and pos 0 -> color via inverse (+2 mod 3)
            return np.column_stack([(msgs[:, 1] + 1) % 3, (msgs[:, 0] + 2) % 3])

        res = compute_dialect_matrix([msgs_a, msgs_b], [listener_a, listener_b], targets, vocab_size=3)
        self.assertAlmostEqual(res["diagonal_mean"], 1.0)
        self.assertLess(res["raw_off_diagonal_mean"], 0.4)
        self.assertGreater(res["hungarian_off_diagonal_mean"], res["raw_off_diagonal_mean"])
        self.assertAlmostEqual(res["hungarian_off_diagonal_mean"], 1.0)

    def test_synthetic_split_and_protocol_helpers(self):
        split = generate_synthetic_split(held_out_pairs=((2, 0),))
        self.assertEqual(len(split["train_objects"]), 8)
        self.assertEqual(len(split["held_out_objects"]), 1)
        msgs = assert_discrete_messages(np.array([[0, 1], [2, 0]]), vocab_size=3, message_length=2)
        batch = Batch(features=np.zeros((2, 6)), labels=msgs)
        self.assertEqual(len(batch.features), 2)
        with self.assertRaises(TypeError):
            assert_discrete_messages(np.array([[0.0, 1.0]]))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_language_metrics([[0, 0], [1, 1]], [0], [0])
        with self.assertRaises(ValueError):
            topographic_similarity([[0, 0], [1, 1]], [[0], [1]])


if __name__ == "__main__":
    unittest.main()
