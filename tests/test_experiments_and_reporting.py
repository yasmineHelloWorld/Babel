"""Tests for YAML experiment manifests and programmatic docs/results.md generation."""

import tempfile
import unittest
from pathlib import Path

from babel.config import ExperimentConfig
from babel.experiments import run_manifest
from babel.reporting import generate_results_markdown


class ExperimentAndReportingTests(unittest.TestCase):
    def test_yaml_manifest_round_trip_and_synthetic_grammar_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "exp.yaml"
            manifest.write_text(
                "\n".join([
                    "name: test_synth",
                    "dataset: synthetic",
                    "message_length: 2",
                    "vocab_sizes: [3]",
                    "seeds: [0, 1]",
                    "epochs: 40",
                    "batch_size: 32",
                    "hidden_dim: 32",
                    "held_out_combinations: [[2, 0]]",
                    f"data_dir: { (root / 'data').as_posix() }",
                    f"splits_dir: { (root / 'splits').as_posix() }",
                    f"runs_dir: { (root / 'runs').as_posix() }",
                    "extra:",
                    "  kind: grammar",
                ]),
                encoding="utf-8",
            )
            cfg = ExperimentConfig.from_yaml(manifest)
            self.assertEqual(cfg.vocab_sizes, (3,))
            res = run_manifest(manifest)
            self.assertTrue(Path(res["summary_path"]).is_file())

            report = generate_results_markdown(root / "runs", root / "results.md")
            text = report.read_text(encoding="utf-8")
            self.assertIn("test_synth", text)
            self.assertIn("PosDis", text)


if __name__ == "__main__":
    unittest.main()