"""23x the capacity of the baseline, same number of steps.

    python experiments/tinystories-d128-l6-h4-ctx128-5k.py
"""

from pathlib import Path

from _run import run
from pvml.config import Config
from pvml.training.args import TrainingArgs

MODEL = Config(
    d_model=128, n_heads=4, d_head=32, d_mlp=512,
    n_layers=6, n_ctx=128, debug=False,
)

ARGS = TrainingArgs(
    name=Path(__file__).stem,
    batch_size=32,
    epochs=10,
    max_steps_per_epoch=500,
    lr=1e-3,
    seed=0,
)

if __name__ == "__main__":
    run(MODEL, ARGS)
