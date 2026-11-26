import argparse
import json
from pathlib import Path

from babel.baselines import run_tabular_baselines
from babel.config import RunConfig, TabularConfig
from babel.experiments import run_manifest
from babel.reporting import generate_results_markdown
from babel.tabular_experiment import prepare_tabular_experiment
from babel.train import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="babel", description="Train a symbolic communication experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="train on the synthetic grammar world")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--episodes", type=int, default=30_000)
    train_parser.add_argument("--vocab-size", type=int, default=3)
    train_parser.add_argument("--message-length", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=0.08)
    train_parser.add_argument("--entropy-bonus", type=float, default=0.01)
    train_parser.add_argument("--baseline-rate", type=float, default=0.05)
    train_parser.add_argument("--log-every", type=int, default=500)
    train_parser.add_argument("--runs-dir", default="runs")
    tabular_parser = subparsers.add_parser("prepare-tabular", help="prepare a reproducible Phase 3 tabular dataset")
    tabular_parser.add_argument("--config", dest="config_path", help="JSON tabular configuration file")
    tabular_parser.add_argument("--dataset", choices=("congressional_voting_records", "car_evaluation"))
    tabular_parser.add_argument("--seed", type=int)
    tabular_parser.add_argument("--data-dir")
    tabular_parser.add_argument("--splits-dir")
    tabular_parser.add_argument("--held-out", type=int, nargs=2, action="append", metavar=("VALUE_A", "VALUE_B"),
                                help="integer-coded pair to reserve for zero-shot evaluation; repeat as needed")
    tabular_parser.add_argument("--held-out-columns", type=int, nargs=2, metavar=("COLUMN_A", "COLUMN_B"))
    baseline_parser = subparsers.add_parser("compare-tabular", help="run Phase 4 classical baselines on a prepared split")
    baseline_parser.add_argument("--config", dest="config_path", required=True, help="JSON tabular configuration file")
    baseline_parser.add_argument("--symbols", type=int, default=3, help="symbols available per message position")
    baseline_parser.add_argument("--message-length", type=int, default=2)
    baseline_parser.add_argument("--output", help="NPZ output path (default: runs/<dataset>-baselines.npz)")

    exp_parser = subparsers.add_parser("run-experiment", help="run a Phase 7 YAML experiment manifest")
    exp_parser.add_argument("--config", dest="config_path", required=True, help="YAML experiment manifest")

    rep_parser = subparsers.add_parser("reproduce", help="run all YAML experiment manifests and generate docs/results.md")
    rep_parser.add_argument("--configs-dir", default="configs")
    rep_parser.add_argument("--runs-dir", default="runs")
    rep_parser.add_argument("--output", default="docs/results.md")

    report_parser = subparsers.add_parser("generate-report", help="compile docs/results.md from saved runs/summaries")
    report_parser.add_argument("--runs-dir", default="runs")
    report_parser.add_argument("--output", default="docs/results.md")

    args = parser.parse_args()
    if args.command == "prepare-tabular":
        values = json.loads(Path(args.config_path).read_text(encoding="utf-8")) if args.config_path else {}
        config = TabularConfig.from_dict(values)
        overrides = {
            "dataset": args.dataset,
            "seed": args.seed,
            "data_dir": args.data_dir,
            "splits_dir": args.splits_dir,
            "held_out_combinations": tuple(tuple(pair) for pair in args.held_out) if args.held_out else None,
            "held_out_columns": tuple(args.held_out_columns) if args.held_out_columns else None,
        }
        config = TabularConfig.from_dict({**config.to_dict(), **{key: value for key, value in overrides.items() if value is not None}})
        result = prepare_tabular_experiment(config)
    elif args.command == "compare-tabular":
        values = json.loads(Path(args.config_path).read_text(encoding="utf-8"))
        config = TabularConfig.from_dict(values)
        output = args.output or f"runs/{config.dataset}-baselines.npz"
        predictions = run_tabular_baselines(config, message_symbols=args.symbols,
                                            message_length=args.message_length, output_path=output)
        result = {"output_path": output, "budget": predictions.budget,
                  "methods": list(predictions.predictions)}
    elif args.command == "run-experiment":
        result = run_manifest(args.config_path)
    elif args.command == "reproduce":
        train(RunConfig(seed=0, runs_dir=args.runs_dir))
        manifest_paths = sorted(Path(args.configs_dir).glob("*.yaml"))
        outputs = []
        for path in manifest_paths:
            res = run_manifest(path)
            outputs.append({"manifest": str(path), "summary_path": res["summary_path"]})
        report_path = generate_results_markdown(args.runs_dir, args.output)
        result = {"manifests_run": outputs, "report_path": str(report_path)}
    elif args.command == "generate-report":
        report_path = generate_results_markdown(args.runs_dir, args.output)
        result = {"report_path": str(report_path)}
    else:
        config = RunConfig(seed=args.seed, episodes=args.episodes, vocab_size=args.vocab_size,
                           message_length=args.message_length, learning_rate=args.learning_rate,
                           entropy_bonus=args.entropy_bonus, baseline_rate=args.baseline_rate,
                           log_every=args.log_every, runs_dir=args.runs_dir)
        result = train(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
