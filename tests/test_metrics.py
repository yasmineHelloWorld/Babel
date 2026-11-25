import unittest

import numpy as np

from babel.datasets.synthetic import generate_synthetic_world
from babel.metrics import compute_language_metrics, topographic_similarity


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
        self.assertAlmostEqual(metrics["position_property_mutual_information_bits"]["0:shape"], np.log2(3))
        self.assertAlmostEqual(metrics["position_property_mutual_information_bits"]["0:color"], 0.0)
        self.assertAlmostEqual(metrics["position_property_mutual_information_bits"]["1:shape"], 0.0)
        self.assertAlmostEqual(metrics["position_property_mutual_information_bits"]["1:color"], np.log2(3))
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
        for value in metrics["position_property_mutual_information_bits"].values():
            self.assertLess(value, 0.02)

    def test_stochastic_synonymy_is_positive(self):
        messages = np.zeros((len(self.objects), 2, 2), dtype=int)
        messages[:, 1, 0] = 1
        metrics = compute_language_metrics(messages, self.shapes, self.colors)

        self.assertAlmostEqual(metrics["synonymy_bits"], 1.0)
        self.assertAlmostEqual(metrics["ambiguity_bits"], np.log2(9))

    def test_compositional_oracle_has_perfect_topographic_similarity(self):
        messages = np.array([[obj["shape_id"], obj["color_id"]] for obj in self.objects])
        features = np.array([obj["features"] for obj in self.objects])

        self.assertAlmostEqual(topographic_similarity(messages, features), 1.0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_language_metrics([[0, 0], [1, 1]], [0], [0])
        with self.assertRaises(ValueError):
            topographic_similarity([[0, 0], [1, 1]], [[0], [1]])


if __name__ == "__main__":
    unittest.main()
