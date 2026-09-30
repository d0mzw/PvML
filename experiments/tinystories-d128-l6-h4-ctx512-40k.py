"""Was the missing coherence a context limit? 4x the window, 2x the steps.

At n_ctx=128 only 5.2% of TinyStories fit whole, so every earlier run was
trained on windows that start and end mid-story. At 512 it is 96.8%, and the
model sees a beginning, a middle and an end. Going further to 1024 buys three
more points of coverage for roughly 2.9x the step cost, so 512 is the jump.

40,000 steps at this window is 655M tokens, 1.37 epochs of the corpus and 8x
the budget of the best ctx128 run. Everything else is held at the values the
other five used, so the window and the step count are the only changes.

max_steps_per_epoch is 1000 rather than 500: same total, half as many evals.

    python experiments/tinystories-d128-l6-h4-ctx512-40k.py
"""

from pathlib import Path

from _run import run
from pvml.config import Config
from pvml.training.args import TrainingArgs

MODEL = Config(
    d_model=128, n_heads=4, d_head=32, d_mlp=512,
    n_layers=6, n_ctx=512, debug=False,
)

ARGS = TrainingArgs(
    name=Path(__file__).stem,
    batch_size=32,
    epochs=40,
    max_steps_per_epoch=1000,
    lr=1e-3,
    seed=0,
)

if __name__ == "__main__":
    run(MODEL, ARGS)
