"""Does more capacity still pay at equal steps? n_ctx held at 128.

    python experiments/tinystories-d256-l8-h8-ctx128-5k.py
"""

from pathlib import Path

from _run import run
from pvml.config import Config
from pvml.training.args import TrainingArgs

MODEL = Config(
    d_model=256, n_heads=8, d_head=32, d_mlp=1024,
    n_layers=8, n_ctx=128, debug=False,
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
