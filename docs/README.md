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
| `Attention` | `pvml/modules/attention.py` | [attention.md](modules/attention.md) — mask only |
| `MLP` | `pvml/modules/mlp.py` | _not started_ |
| `TransformerBlock` | `pvml/modules/block.py` | _not started_ |
| `Transformer`, `Unembed` | `pvml/models/transformer.py` | _not started_ |

## Reference

`pvml/models/loading.py` is the only place `transformer_lens` is imported. It
loads GPT-2 small with `fold_ln`, `center_unembed` and `center_writing_weights`
all set to `False` — they default to `True` and algebraically rearrange the
weights, so leaving them on means nothing we load will match.
