from dataclasses import dataclass


@dataclass
class TrainingArgs:
    """Run knobs. Architecture lives in Config; these are separate on purpose."""

    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2

    # metrics go to runs/<run_name>/, one JSON object per line
    run_dir: str = "runs"
    run_name: str | None = None  # defaults to a timestamp

    eval_prompt: str = "Once upon a time"
