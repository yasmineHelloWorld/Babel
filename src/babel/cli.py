import argparse
import json

from babel.config import RunConfig
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
    args = parser.parse_args()
    config = RunConfig(seed=args.seed, episodes=args.episodes, vocab_size=args.vocab_size,
                       message_length=args.message_length, learning_rate=args.learning_rate,
                       entropy_bonus=args.entropy_bonus, baseline_rate=args.baseline_rate,
                       log_every=args.log_every, runs_dir=args.runs_dir)
    result = train(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
