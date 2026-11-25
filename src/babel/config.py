from dataclasses import asdict, dataclass
from pathlib import Path


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
        if self.dataset not in {"congressional_voting_records", "car_evaluation"}:
            raise ValueError("dataset must be 'congressional_voting_records' or 'car_evaluation'")
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
