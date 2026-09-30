# Transformer

`src/pvml/modules/transformer.py`, ported from ARENA 1.1 *Transformer from Scratch*.

The first check where nothing is borrowed. Every other module was handed its
input by the reference; this one takes tokens and produces logits through its
own embedding, twelve of its own blocks, and its own unembedding.

```
max abs diff vs gpt-2: 9.155e-05
same argmax everywhere: True
missing keys: []
```

`missing_keys: []` is the part that matters as much as the diff. Every GPT-2
weight found a home, so nothing is silently untrained or misnamed. That is the
check `strict=False` would otherwise hide.

## Forward

```python
residual = self.embed(tokens) + self.pos_embed(tokens)

for block in self.blocks:
    residual = block(residual)

logits = self.unembed(self.ln_final(residual))
```

- `residual = block(residual)` only works because the shape never changes. That
  is why blocks stack
- Nothing normalises the stream itself along the way, so `ln_final` is needed
  before the unembedding can read it
- `nn.ModuleList` gives the children paths like `blocks.3.attn`, which is what
  TransformerLens calls its hooks. One vocabulary for the trace and the
  reference activations

## tag_tree

`__init__` calls `tag_tree(self)`, which stamps every submodule with its
position. A trace then prints `[b3.attn]` rather than twelve indistinguishable
`[attn]` lines, and `grep 'b3.'` gets one block.

## Parameters

```
embed        38,597,376
pos_embed       786,432
blocks       85,054,464
ln_final          1,536
unembed      38,647,633
total       163,087,441
```

GPT-2 small is usually quoted as 124M, which is this total minus the
unembedding: the original ties `W_U` to `W_E` and counts them once, while
TransformerLens keeps them separate. Same model, two honest numbers.

## Run

```
python -m pvml.modules.transformer
```
