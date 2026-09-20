# LayerNorm

`src/pvml/modules/normalization.py` — ported from ARENA 1.1, `exercise.py:153-165`.

## What it does

Standardises each position's residual-stream vector to mean 0 and standard
deviation 1, then applies a learned scale and shift.

The normalisation runs along `d_model` **only** — independently for every
`(batch, position)` pair. Position 3 of sequence 0 is normalised using its own
768 numbers and nothing else. No statistics are shared across positions or
across the batch, which is what distinguishes LayerNorm from BatchNorm and why
it works with variable sequence lengths and batch size 1.

## Parameters

| name | shape | init | role |
|------|-------|------|------|
| `w` | `(d_model,)` | ones | learned scale, per channel |
| `b` | `(d_model,)` | zeros | learned shift, per channel |

Initialised to the identity transform, so at step 0 the layer is pure
standardisation and *learns* whether to deviate. Zeros for `w` would kill the
signal; ones for `b` would inject a constant offset.

Note `w` and `b` are shared across all positions — one scale and shift for the
whole layer — while the mean and std are per-position. Normalise locally,
transform globally.

## Forward

```
in          (batch, posn, d_model)
mean        (batch, posn, 1)
std         (batch, posn, 1)
normalised  (batch, posn, d_model)
out         (batch, posn, d_model)
```

## Details worth remembering

**`keepdim=True`** leaves the reduced axis as a size-1 slot so it broadcasts
back. Broadcasting aligns shapes from the right:

```
(2, 10, 768) - (2, 10, 1)   ->  1 stretches to 768   ok
(2, 10, 768) - (2, 10)      ->  compares 768 vs 10   error
```

The danger case is when the mismatched sizes happen to be compatible — if batch
size equalled `d_model`, the second form would broadcast silently against the
wrong axis and produce plausible garbage with no error.

**`unbiased=False`** divides by N rather than N-1. Bessel's correction exists
for *estimating* a population variance from a sample; here we have all 768
numbers and want exactly these standardised. It is also what GPT-2 was trained
with, so weight parity depends on it. At `d_model=768` the difference is a
factor of about 1.00065 — small enough to train fine, large enough to fail a
parity check.

**`layer_norm_eps` goes inside the sqrt**, added to the variance, not to the
std afterwards. Order matters for matching reference implementations.

**Scale is discarded.** `[1,2,3,4]` and `[10,20,30,40]` normalise to the same
vector. Anything downstream that needs magnitude must get it elsewhere.

## Run

```
python -m pvml.modules.normalization
```

```
LayerNorm in: (2, 10, 768)
        mean: (2, 10, 1)
         std: (2, 10, 1)
  normalised: (2, 10, 768)    # (batch, posn, d_model) - (batch, posn, 1)
        w, b: (768,)          # padded to (1, 1, d_model), stretched to out
         out: (2, 10, 768)
```
