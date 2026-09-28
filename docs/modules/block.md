# TransformerBlock

`src/pvml/modules/block.py` — ported from ARENA 1.1 *Transformer from Scratch*.

Matches GPT-2's `blocks.0.hook_resid_post` to `1.1e-05` on values reaching 120,
which is float32 accumulation over four layers rather than an error. Every
module up to here matched exactly; this is the first one that composes several.

## What it does

```python
resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
```

Attention and the MLP each read the stream and **add a delta back**. Neither
replaces it. That additivity is what keeps every component's contribution
separable, and it is why a single head's effect can be isolated.

## Children

| name | class |
|------|-------|
| `ln1` | `LayerNorm` — normalises what attention reads |
| `attn` | `Attention` |
| `ln2` | `LayerNorm` — normalises what the MLP reads |
| `mlp` | `MLP` |

**7,087,872 parameters.** Twelve blocks plus the 38.6M embedding is roughly
GPT-2 small's 124M.

## The residual stream

The stream is `(batch, posn, d_model)` at every point, in every block. That
constant width is why blocks stack.

**LayerNorm sits on the branches, not the trunk.** `ln1` normalises what
attention *reads*; what gets added back is the unnormalised `attn_out`. So the
stream itself never passes through a LayerNorm, and accumulates raw
contributions across all twelve blocks.

The wide shapes inside the sublayers — `(1, 7, 3072)` in the MLP,
`(1, 7, 12, 64)` in attention — are private scratch space, projected back to
`d_model` before anything is added.

## Tagging

`__init__` calls `tag_tree(self)`, so the two LayerNorms print as `[ln1]` and
`[ln2]` rather than two indistinguishable `[ln]`. A `Transformer` calls it
again later and overwrites these with `[b0.ln1]`, `[b0.attn]` and so on.

## Run

```
python -m pvml.modules.block
```

Prints the whole forward pass, 36 lines from `resid_pre` to `resid_post`, with
the two `[block]` lines marking the residual additions.
