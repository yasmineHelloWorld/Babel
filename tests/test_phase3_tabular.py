import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from babel.datasets.tabular import load_car_evaluation, load_congressional_voting_records
from babel.config import TabularConfig
from babel.preprocessing import prepare_tabular_data
from babel.split_guard import DataSplit, SplitGuard
from babel.tabular_experiment import prepare_tabular_experiment


class PhaseThreeTabularTests(unittest.TestCase):
    def test_congressional_loader_reads_cached_uci_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "congressional-voting-records.zip"
            rows = (
                "democrat,y,n,?,y,n,y,n,y,n,?,y,n,y,n,y,n\n"
                "republican,n,y,y,n,y,n,y,n,y,n,n,y,n,y,n,y\n"
            )
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("house-votes-84.data", rows)

            dataset = load_congressional_voting_records(tmp, download=False)

        self.assertEqual(dataset.features.shape, (2, 16))
        self.assertEqual(dataset.labels.tolist(), [0, 1])
        self.assertEqual(dataset.features[0, 2], 2)
        self.assertEqual(dataset.label_names, ("democrat", "republican"))
        self.assertEqual(dataset.feature_values[0], ("y", "n", "?"))

    def test_car_loader_reads_cached_uci_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "car-evaluation.zip"
            rows = (
                "vhigh,vhigh,2,2,small,low,unacc\n"
                "low,med,4,4,big,high,good\n"
                "med,low,3,more,med,med,acc\n"
            )
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("car.data", rows)

            dataset = load_car_evaluation(tmp, download=False)

        self.assertEqual(dataset.features.shape, (3, 6))
        self.assertEqual(dataset.labels.tolist(), [0, 2, 1])
        self.assertEqual(dataset.feature_names[0], "buying")

    def test_stratified_split_is_persisted_and_reused(self):
        labels = np.repeat(np.arange(3), 20)
        with tempfile.TemporaryDirectory() as tmp:
            guard = SplitGuard(Path(tmp) / "congressional_voting_records.npz")
            first = guard.create(labels, seed=27)
            reused = guard.create(labels, seed=999)
            loaded = guard.load(labels=labels)

        self.assertEqual(first.sha256, reused.sha256)
        self.assertEqual(first.sha256, loaded.sha256)
        self.assertEqual(len(np.unique(np.concatenate((loaded.train, loaded.validation, loaded.test)))), len(labels))
        for indices in (loaded.train, loaded.validation, loaded.test):
            counts = np.bincount(labels[indices], minlength=3)
            self.assertGreater(np.min(counts), 0)

    def test_split_detects_changed_row_order(self):
        labels = np.repeat(np.arange(2), 12)
        with tempfile.TemporaryDirectory() as tmp:
            guard = SplitGuard(Path(tmp) / "split.npz")
            guard.create(labels, seed=3)
            changed = labels[::-1].copy()
            with self.assertRaisesRegex(ValueError, "different labels or row ordering"):
                guard.load(labels=changed)

    def test_tabular_config_round_trips_held_out_values(self):
        config = TabularConfig.from_dict({
            "dataset": "congressional_voting_records",
            "held_out_combinations": [[0, 2], [1, 1]],
            "held_out_columns": [3, 8],
        })
        config.validate()
        self.assertEqual(config.held_out_combinations, ((0, 2), (1, 1)))
        self.assertEqual(config.held_out_columns, (3, 8))
        self.assertTrue(config.split_path.name.endswith("-held-out.npz"))

    def test_tabular_experiment_writes_reproducibility_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            rows = []
            for party in ("democrat", "republican"):
                for index in range(12):
                    votes = [("y", "n", "?")[(index + column) % 3] for column in range(16)]
                    rows.append(",".join((party, *votes)))
            with ZipFile(data_dir / "congressional-voting-records.zip", "w") as archive:
                archive.writestr("house-votes-84.data", "\n".join(rows) + "\n")

            result = prepare_tabular_experiment(TabularConfig(
                data_dir=str(data_dir), splits_dir=str(root / "splits"),
            ))
            self.assertTrue(Path(result["summary_path"]).is_file())

        self.assertEqual(result["dataset"], "congressional_voting_records")
        self.assertEqual(result["n_samples"], 24)
        self.assertEqual(result["split_type"], "stratified")

    def test_categorical_encoder_is_fit_on_training_rows_only(self):
        features = np.array([["yea"], ["nay"], ["absent-in-training"], ["yea"]])
        labels = np.array([0, 0, 1, 1])
        split = DataSplit(np.array([0, 1]), np.array([2]), np.array([3]), "test")
        prepared = prepare_tabular_data(features, labels, split)

        self.assertEqual(prepared.encoder.categories_[0].tolist(), ["nay", "yea"])
        self.assertEqual(prepared.test_features.shape, (1, 2))
        self.assertEqual(prepared.test_features.tolist(), [[0.0, 1.0]])
        self.assertEqual(prepared.validation_features.tolist(), [[0.0, 0.0]])

    def test_held_out_combinations_are_test_only(self):
        pairs = [(buying, safety) for buying in range(4) for safety in range(3) for _ in range(4)]
        features = np.zeros((len(pairs), 6), dtype=np.float64)
        features[:, 0] = [pair[0] for pair in pairs]
        features[:, 5] = [pair[1] for pair in pairs]
        held_out = {(0, 0), (3, 2)}
        with tempfile.TemporaryDirectory() as tmp:
            split = SplitGuard(Path(tmp) / "car.npz").create_held_out_combinations(features, held_out=held_out, seed=8)
            guard = SplitGuard(Path(tmp) / "car.npz")
            reloaded = guard.create_held_out_combinations(features, held_out=held_out, seed=8)

        self.assertEqual(split.sha256, reloaded.sha256)
        train_pairs = {(int(features[i, 0]), int(features[i, 5])) for i in split.train}
        validation_pairs = {(int(features[i, 0]), int(features[i, 5])) for i in split.validation}
        test_pairs = {(int(features[i, 0]), int(features[i, 5])) for i in split.test}
        self.assertTrue(held_out.isdisjoint(train_pairs | validation_pairs))
        self.assertEqual(test_pairs, held_out)


if __name__ == "__main__":
    unittest.main()
