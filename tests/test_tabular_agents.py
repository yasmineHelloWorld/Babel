"""Behavioral tests for the tabular Speaker and Listener agents."""

import unittest

import numpy as np

from babel.agents.tabular import TabularListener, TabularSpeaker


class TabularSpeakerTests(unittest.TestCase):
    def make_speaker(self, seed: int = 7) -> TabularSpeaker:
        return TabularSpeaker(4, 2, 3, np.random.default_rng(seed))

    def test_probabilities_are_normalized_for_each_position(self) -> None:
        speaker = self.make_speaker()
        probabilities = speaker.probabilities(2)

        self.assertEqual(probabilities.shape, (2, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
        self.assertTrue(np.all(probabilities > 0))

    def test_speak_returns_valid_symbols_and_corresponding_probabilities(self) -> None:
        speaker = self.make_speaker()
        message, probabilities = speaker.speak(1)

        self.assertEqual(message.shape, (2,))
        self.assertEqual(probabilities.shape, (2, 3))
        self.assertTrue(np.issubdtype(message.dtype, np.integer))
        self.assertTrue(np.all((0 <= message) & (message < 3)))
        for position, symbol in enumerate(message):
            self.assertGreater(probabilities[position, symbol], 0)

    def test_speak_sampling_approximately_matches_policy(self) -> None:
        speaker = self.make_speaker(seed=17)
        speaker.logits[0, 0] = np.array([-1.2, 0.3, 1.4])
        expected = speaker.probabilities(0)[0]
        draws = np.fromiter((speaker.speak(0)[0][0] for _ in range(8000)), dtype=np.int64)
        observed = np.bincount(draws, minlength=3) / len(draws)

        np.testing.assert_allclose(observed, expected, atol=0.025)

    def test_reinforce_update_increases_selected_symbol_probability(self) -> None:
        speaker = self.make_speaker()
        message = np.array([2, 0])
        probabilities = speaker.probabilities(0)
        before = probabilities.copy()

        speaker.update(0, message, probabilities, advantage=1.0,
                       learning_rate=0.2, entropy_bonus=0.0)
        after = speaker.probabilities(0)

        for position, symbol in enumerate(message):
            self.assertGreater(after[position, symbol], before[position, symbol])
        np.testing.assert_array_equal(speaker.logits[1:], self.make_speaker().logits[1:])

    def test_entropy_bonus_moves_a_skewed_policy_toward_uniform(self) -> None:
        speaker = self.make_speaker()
        speaker.logits[0, 0] = np.array([5.0, -2.0, 0.0])
        before = speaker.probabilities(0)[0]
        message = np.array([0, 0])
        probabilities = speaker.probabilities(0)

        speaker.update(0, message, probabilities, advantage=0.0,
                       learning_rate=0.2, entropy_bonus=1.0)
        after = speaker.probabilities(0)[0]

        self.assertLess(after.max() - after.min(), before.max() - before.min())


class TabularListenerTests(unittest.TestCase):
    def make_listener(self, seed: int = 11) -> TabularListener:
        return TabularListener(2, 3, 3, 4, np.random.default_rng(seed))

    def test_initial_predictions_are_valid_distributions(self) -> None:
        listener = self.make_listener()
        shape, color = listener.predict(np.array([1, 2]))

        self.assertEqual(shape.shape, (3,))
        self.assertEqual(color.shape, (4,))
        self.assertTrue(np.all(shape >= 0))
        self.assertTrue(np.all(color >= 0))
        self.assertAlmostEqual(float(shape.sum()), 1.0)
        self.assertAlmostEqual(float(color.sum()), 1.0)

    def test_observation_updates_only_the_matching_message_and_labels(self) -> None:
        listener = self.make_listener()
        message = np.array([1, 2])
        other_message = np.array([0, 0])
        shape_before, color_before = listener.predict(message)
        other_before = listener.predict(other_message)
        shape_counts_before = listener.shape_counts.copy()
        color_counts_before = listener.color_counts.copy()

        listener.observe(message, shape_id=2, color_id=3)
        shape_after, color_after = listener.predict(message)
        other_after = listener.predict(other_message)

        self.assertGreater(shape_after[2], shape_before[2])
        self.assertGreater(color_after[3], color_before[3])
        np.testing.assert_allclose(other_after[0], other_before[0])
        np.testing.assert_allclose(other_after[1], other_before[1])
        expected_shape_counts = shape_counts_before.copy()
        expected_color_counts = color_counts_before.copy()
        expected_shape_counts[1, 2, 2] += 1
        expected_color_counts[1, 2, 3] += 1
        np.testing.assert_array_equal(listener.shape_counts, expected_shape_counts)
        np.testing.assert_array_equal(listener.color_counts, expected_color_counts)

    def test_shape_and_color_observations_update_independent_counts(self) -> None:
        listener = self.make_listener()
        message = np.array([2, 1])
        shape_before = listener.shape_counts.copy()
        color_before = listener.color_counts.copy()

        listener.observe(message, shape_id=1, color_id=2)

        self.assertEqual(listener.shape_counts[2, 1, 1], shape_before[2, 1, 1] + 1)
        self.assertEqual(listener.color_counts[2, 1, 2], color_before[2, 1, 2] + 1)
        self.assertEqual(np.count_nonzero(listener.shape_counts - shape_before), 1)
        self.assertEqual(np.count_nonzero(listener.color_counts - color_before), 1)

    def test_seeded_initialization_is_reproducible(self) -> None:
        left = self.make_listener(seed=23)
        right = self.make_listener(seed=23)

        np.testing.assert_array_equal(left.shape_counts, right.shape_counts)
        np.testing.assert_array_equal(left.color_counts, right.color_counts)


if __name__ == "__main__":
    unittest.main()
