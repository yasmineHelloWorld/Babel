from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    episodes: int = 30_000
    vocab_size: int = 3
    message_length: int = 2
    learning_rate: float = 0.08
    entropy_bonus: float = 0.01
    baseline_rate: float = 0.05
    log_every: int = 500
    runs_dir: str = "runs"

    def validate(self) -> None:
        if self.episodes < 1 or self.log_every < 1:
            raise ValueError("episodes and log_every must be positive")
        if self.vocab_size < 2 or self.message_length < 1:
            raise ValueError("vocab_size must be >= 2 and message_length >= 1")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= self.entropy_bonus or not 0 < self.baseline_rate <= 1:
            raise ValueError("entropy_bonus must be nonnegative and baseline_rate in (0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "RunConfig":
        return cls(**values)

    @property
    def output_root(self) -> Path:
        return Path(self.runs_dir)


@dataclass(frozen=True)
class TabularConfig:
    dataset: str = "congressional_voting_records"
    seed: int = 0
    data_dir: str = "data"
    splits_dir: str = "data/splits"
    held_out_combinations: tuple[tuple[int, int], ...] = ()
    held_out_columns: tuple[int, int] = (0, 5)

    def validate(self) -> None:
        if self.dataset not in {"congressional_voting_records", "car_evaluation", "wine"}:
            raise ValueError("dataset must be 'congressional_voting_records', 'car_evaluation', or 'wine'")
        if any(len(pair) != 2 for pair in self.held_out_combinations):
            raise ValueError("each held-out combination must contain two integer-coded values")
        if len(self.held_out_columns) != 2 or min(self.held_out_columns) < 0:
            raise ValueError("held_out_columns must contain two nonnegative column indexes")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "TabularConfig":
        normalized = dict(values)
        if "held_out_combinations" in normalized:
            normalized["held_out_combinations"] = tuple(tuple(pair) for pair in normalized["held_out_combinations"])
        if "held_out_columns" in normalized:
            normalized["held_out_columns"] = tuple(normalized["held_out_columns"])
        return cls(**normalized)

    @property
    def split_path(self) -> Path:
        suffix = "-held-out" if self.held_out_combinations else ""
        return Path(self.splits_dir) / f"{self.dataset}{suffix}.npz"


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed YAML manifest for neural/tabular multi-seed sweeps and real-data benchmarks."""

    name: str
    dataset: str
    message_length: int = 1
    vocab_sizes: tuple[int, ...] = (2, 3, 4, 6, 8, 16)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 120
    batch_size: int = 64
    hidden_dim: int = 64
    speaker_lr: float = 3e-3
    listener_lr: float = 5e-3
    entropy_bonus: float = 0.02
    baseline_ema: float = 0.05
    listener_mode: str = "holistic"
    training_mode: str = "hybrid_ce"
    pca_components: int = 32
    target_columns: tuple[int, ...] = ()
    held_out_combinations: tuple[tuple[int, int], ...] = ()
    held_out_columns: tuple[int, int] = (0, 5)
    data_dir: str = "data"
    splits_dir: str = "data/splits"
    runs_dir: str = "runs"
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("experiment name must be non-empty")
        if self.message_length < 1:
            raise ValueError("message_length must be >= 1")
        if not self.vocab_sizes or any(k < 2 for k in self.vocab_sizes):
            raise ValueError("all vocab_sizes must be >= 2")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.epochs < 1 or self.batch_size < 1 or self.hidden_dim < 4:
            raise ValueError("epochs, batch_size, and hidden_dim must be positive")
        if self.listener_mode not in {"holistic", "structured"}:
            raise ValueError("listener_mode must be 'holistic' or 'structured'")
        if self.training_mode not in {"hybrid_ce", "both_reinforce"}:
            raise ValueError("training_mode must be 'hybrid_ce' or 'both_reinforce'")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ExperimentConfig":
        normalized = dict(values)
        for key in ("vocab_sizes", "seeds", "target_columns"):
            if key in normalized:
                normalized[key] = tuple(int(v) for v in normalized[key])
        if "held_out_combinations" in normalized:
            normalized["held_out_combinations"] = tuple(tuple(int(x) for x in pair) for pair in normalized["held_out_combinations"])
        if "held_out_columns" in normalized:
            normalized["held_out_columns"] = tuple(int(v) for v in normalized["held_out_columns"])
        return cls(**normalized)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"YAML config must be a mapping: {path}")
        cfg = cls.from_dict(payload)
        cfg.validate()
        return cfg
