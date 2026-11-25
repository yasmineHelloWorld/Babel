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
