# Docs

Notes on each piece of the transformer, written while porting it from the
ARENA 1.1 *Transformer from Scratch* exercises into `src/pvml/`.

Each page covers what the layer does, its parameter shapes, and the details
that are easy to get wrong — the kind that still train if you get them wrong.

## Modules

Listed in porting order, which is also the order they run in a forward pass.

| module | file | docs |
|--------|------|------|
| `Config` | `pvml/config.py` | — |
| `LayerNorm` | `pvml/modules/normalization.py` | [normalization.md](modules/normalization.md) |
| `Embed`, `PosEmbed` | `pvml/modules/embedding.py` | [embedding.md](modules/embedding.md) |
| `Attention` | `pvml/modules/attention.py` | [attention.md](modules/attention.md) |
| `MLP` | `pvml/modules/mlp.py` | [mlp.md](modules/mlp.md) |
| `TransformerBlock` | `pvml/modules/block.py` | [block.md](modules/block.md) |
| `Unembed` | `pvml/modules/unembedding.py` | [unembedding.md](modules/unembedding.md) |
| `Transformer` | `pvml/modules/transformer.py` | _not started_ |

## Debug tracing

Every module prints its shapes through `pvml.debug.trace`, gated on
`Config.debug` (off by default; each module's `__main__` turns it on).

```
[b0.attn]                      q: (1, 7, 12, 64)  # (batch, posn, n_heads, d_head)
```

`tag_tree(model)` stamps each submodule with its position, so a full trace can
be filtered with `grep 'b3.attn'`. Weight shapes pass `once=True` and print for
the first instance of a class only.

## Layout

Everything that is an `nn.Module` lives in `pvml/modules/`. `config.py` and
`debug.py` sit at the package root, since neither is a layer. `pvml/reference/`
holds the loaders for other people's implementations, which is where the
`transformer_lens` dependency is quarantined.

## Reference

`pvml/reference/` is the only place `transformer_lens` is imported. It
loads GPT-2 small with `fold_ln`, `center_unembed` and `center_writing_weights`
all set to `False` — they default to `True` and algebraically rearrange the
weights, so leaving them on means nothing we load will match.

**A hook is not a module boundary.** `ln1.hook_normalized` fires on `x / scale`,
*before* the layer applies its `w` and `b`, so it is not what attention
receives. Feed modules `ref.blocks[0].ln1(resid_pre)` instead. The same applies
to `ln2` and `ln_final`.
