# Training

Everything needed to fit weights rather than borrow them.

## losses.py

```python
log_probs = logits.log_softmax(dim=-1)
return log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
```

`get_log_probs` returns the log-probability the model gave to each token that
actually came next, one per graded prediction rather than a scalar.

- `[:, :-1]` drops the last position, which predicts a token that is not in the
  sequence. `tokens[:, 1:]` drops the first token, which nothing predicts.
  What is left lines up: prediction `i` against token `i+1`
- `gather` uses each answer token as an index into that position's `d_vocab`
  scores. Not a max: it returns what the model gave the right answer, however
  it ranked it
- Swap the two slices and it still runs, still returns a number, and trains
  towards nonsense
- An untrained model scores `log(50257) = 10.83`, which is the check that the
  loss is wired up correctly

## args.py

`TrainingArgs` holds the run knobs, separate from `Config`, which holds the
architecture. These change between runs; those change between models.

`name` is the run directory. Experiment files pass their own filename, so
`experiments/` and `runs/` cannot drift apart.

## trainer.py

Three departures from ARENA's version, each removing a coupling:

- **The loaders are constructor arguments**, not a module-level `dataset_dict`
  the trainer reaches into. It trains on any dataset
- **`training_step` returns the loss and logs nothing.** `train` owns the run,
  so it owns the logging, and a single batch can be run without a run directory
- **Text sampling is a `sample_fn` hook** called after each epoch, rather than
  replacing `train` by monkeypatch

`evaluate` reports next-token accuracy on the held-out chunks: how often argmax
was right, using the same offset as the loss. More legible than nats.

### What a run leaves behind

```
runs/<name>/
├── config.json     every knob: training args, model config, anything extra
├── metrics.jsonl   one object per step, with elapsed seconds
├── model.pt        weights, rewritten after every epoch
└── summary.json    device, wall time, steps per second, final accuracy
```

Written locally, one JSON object per line. No account, nothing leaves the
machine, and a run is reproducible from its own directory.

`model.pt` is rewritten each epoch rather than versioned, so an interrupted run
still leaves usable weights. There is no history of checkpoints.

## data/tinystories.py

```python
tokenized = tokenize_and_concatenate(dataset, tokenizer, max_length=cfg.n_ctx, ...)
split = tokenized.train_test_split(test_size=test_size, seed=seed)
```

- The corpus is glued into one token stream and chopped every `n_ctx`, so every
  row is the same length and no padding is needed. A chunk can begin and end
  mid-story
- 2.1M stories become 3.7M chunks of 128 tokens, about 477M tokens
- `train_test_split` shuffles first, so the held-out chunks are not all from the
  tail. `seed=` keeps the split stable across runs
- The loaders are returned rather than left global

## device.py

`get_device()` is for scripts. Modules take their device from their own
parameters, which is why no module imports this.

ROCm reports as `cuda`, so `t.cuda.is_available()` is true on an AMD GPU and
`torch.version.hip` is how you tell which backend you actually have.
