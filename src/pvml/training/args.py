from dataclasses import dataclass


@dataclass
class TrainingArgs:
    """Run knobs. Architecture lives in Config; these are separate on purpose."""

    name: str = "run"  # the run directory; experiments pass their own filename

    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 0

    run_dir: str = "runs"
    eval_prompt: str = "Once upon a time"

