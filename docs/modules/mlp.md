# MLP

`src/pvml/modules/mlp.py`, ported from ARENA 1.1 *Transformer from Scratch*.

Reproduces GPT-2's `blocks.0.hook_mlp_out` exactly.

## What it does

Expand the residual stream to four times its width, apply the only
nonlinearity in the block, project back.

Every position goes through the same two matrices independently. Nothing here
mixes positions. That only happens in attention.

## Parameters

| name | shape | note |
|------|-------|------|
| `W_in` | `(d_model, d_mlp)` = `(768, 3072)` | 2-D, no head axis |
| `W_out` | `(d_mlp, d_model)` = `(3072, 768)` | |
| `b_in` | `(d_mlp,)` | |
| `b_out` | `(d_model,)` | |

**4,722,432 parameters**, twice what attention costs. Two thirds of a
transformer block is the MLP, which is the opposite of where most
explanations spend their time.

## Forward

```
normalized_resid_mid   (batch, posn, 768)
  @ W_in + b_in    ->  pre       (batch, posn, 3072)    contracts d_model
  gelu_new         ->  post      (batch, posn, 3072)
  @ W_out + b_out  ->  mlp_out   (batch, posn, 768)     contracts d_mlp
```

`gelu_new` is the **only nonlinearity in the block**. Attention, LayerNorm and
every projection are linear, so without it the whole model would collapse into
a single matrix multiply.

`d_mlp = 4 * d_model` is convention, not a requirement.

## Input

`normalized_resid_mid` is the residual stream *after* attention has been added,
through the block's `ln2`. Same relationship attention has with `ln1`.

Don't feed it `cache["blocks.0.ln2.hook_normalized"]` , because that hook fires before
`ln2` applies its `w` and `b`. Use `ref.blocks[0].ln2(resid_mid)`.

## Run

```
python -m pvml.modules.mlp
```
