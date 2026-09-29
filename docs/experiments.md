# Experiments

One file per run, named for the run directory it writes.

```
experiments/tinystories-d128-l6-h4-ctx128-5k.py
        ->  runs/tinystories-d128-l6-h4-ctx128-5k/
```

Each file is a `MODEL`, an `ARGS`, and a call to `run`. No arguments to parse:
changing the experiment means editing the file, and the filename records what
changed. `config.json` in the run directory records every knob either way.

```python
MODEL = Config(d_model=128, n_heads=4, d_head=32, d_mlp=512, n_layers=6, n_ctx=128)

ARGS = TrainingArgs(name=Path(__file__).stem, batch_size=32, epochs=10,
                    max_steps_per_epoch=500, lr=1e-3, seed=0)
```

The name carries what varies between experiments, so a smaller model cannot
overwrite a bigger one's results.

## _run.py

Shared machinery, so a fix lands once rather than in every experiment file.
Holds the wiring and `make_sampler`.

`make_sampler` loads the trained weights into a `HookedTransformer` and calls
its `generate`. Our parameter names match TransformerLens's, so the weights
load straight across and the logits come out identical. Sampling properly is
top-k, top-p, frequency penalty and KV caching, which is its own exercise; this
borrows a tested loop so the samples say something about the model.

## sample.py

A REPL over a finished run.

```
python experiments/sample.py runs/tinystories-d128-l6-h4-ctx128-5k
```

```
:greedy           always take the top token
:temp 0.7         0 means greedy
:topk 40          0 disables
:topp 0.95        0 disables
:len 60
:settings
:q
```

It rebuilds the `Config` from `config.json`, loads `model.pt`, and hands the
weights to a `HookedTransformer`.

Two things TransformerLens expects that are easy to get wrong: `None` rather
than `0` disables `top_k` and `top_p`, and it checks `top_k` before `top_p`, so
setting one clears the other here rather than silently ignoring your input.

## run_all.sh

Runs the planned experiments in order. The sequencing is the shell's job so a
stalled watcher cannot stall the queue; a failing run stops the chain rather
than continuing silently. Markers go to `/tmp/pvml-runs.log`, full training
output stays on the terminal.
