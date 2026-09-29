"""Baseline. 51K parameters outside the vocabulary tables.

    python experiments/tinystories-d32-l4-h16-ctx128-5k.py
"""

from pathlib import Path

from _run import run
from pvml.config import Config
from pvml.training.args import TrainingArgs

MODEL = Config(
    d_model=32, n_heads=16, d_head=2, d_mlp=128,
    n_layers=4, n_ctx=128, debug=False,
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
