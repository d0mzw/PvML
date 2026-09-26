# Attention

`src/pvml/modules/attention.py` — ported from ARENA 1.1 *Transformer from Scratch*.

**Status: partial.** `apply_causal_mask` and the `IGNORE` buffer are done.
`W_Q`/`W_K`/`W_V`/`W_O` and `forward` are not started.

## Why attention exists

A language model predicts the next token, and a position often needs
information from somewhere earlier in the sequence:

```
"When Mary and John went to the store, John gave a drink to ___"
```

Answering `Mary` means reaching back and finding the name that isn't John.
Nothing else in the transformer can do that — LayerNorm and the MLP each see a
single position and cannot look sideways. **Attention is the only mechanism
that moves information between positions.**

The rest is implementation of a soft lookup:

| piece | question it answers |
|-------|---------------------|
| query | what is this position looking for? |
| key | what does each earlier position advertise? |
| dot product | how well do they match? |
| softmax | turn matches into percentages |
| value | what gets copied back |

## Planned shapes

For a 7-token sequence with `n_heads=12`, `d_head=64`:

```
resid (normalised)        (1, 7, 768)
   ├── @ W_Q + b_Q  ->  q  (1, 7, 12, 64)
   ├── @ W_K + b_K  ->  k  (1, 7, 12, 64)
   └── @ W_V + b_V  ->  v  (1, 7, 12, 64)

   q . k             ->  scores   (1, 12, 7, 7)
        / sqrt(d_head)             scale
        apply_causal_mask          mask the future
        softmax(-1)   ->  pattern  (1, 12, 7, 7)

   pattern . v       ->  z        (1, 7, 12, 64)
   z @ W_O + b_O     ->  out      (1, 7, 768)
```

`n_heads` is the leading axis of every weight, so all 12 heads live in one
tensor and run as a single batched matmul. `d_head = d_model / n_heads`, so the
heads collectively occupy exactly `d_model` dimensions of working space.

`scores` is the only tensor in the model that grows as sequence length squared,
which is where context length gets expensive.

## IGNORE

```python
self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32))
```

A 0-dimensional tensor holding `-inf` — the fill value for masked positions.

Registered as a **buffer**, not a parameter: it is module state that is never
learned, so it gets no gradients and does not appear in `.parameters()`. The
reason to register it at all is that buffers follow `.to(device)`, while a
plain `self.x = t.tensor(...)` would be left behind on CPU.

ARENA passes `device=device` here, reading a module-level global that this repo
does not have. Dropped — registration already handles placement.

`-inf` gives exact zeros after softmax. The failure mode to know about: a row
that is *entirely* masked becomes `0/0 = NaN`. Causal masking never does that,
since every position can see itself, but padding masks can — which is why many
implementations use a large finite number instead.

## apply_causal_mask

Takes the `(batch, n_heads, query_pos, key_pos)` scores and sets everything
above the diagonal to `-inf`, so softmax turns those entries into zero.

**`diagonal=1` is the detail that matters.** `t.triu` keeps the upper triangle;
starting one above the main diagonal leaves the diagonal itself unmasked, so a
position may attend to itself:

```
diagonal=0  (wrong)         diagonal=1  (correct)
[[1, 1, 1, 1],              [[0, 1, 1, 1],
 [0, 1, 1, 1],               [0, 0, 1, 1],
 [0, 0, 1, 1],               [0, 0, 0, 1],
 [0, 0, 0, 1]]               [0, 0, 0, 0]]
```

With `diagonal=0`, position 0 would have a fully masked row and produce `NaN`.

The mask is 2-D and broadcasts over batch and heads — causality depends only on
position, identically for every sequence and every head.

`all_ones` is rebuilt on every call because sequence length varies, and it
takes an explicit `device=attn_scores.device`: unlike a registered buffer, a
tensor created inside a method does not follow `.to()`. TransformerLens instead
precomputes a `(n_ctx, n_ctx)` mask buffer once, avoiding the per-call
allocation.

`masked_fill_` is **in place** — it mutates the tensor passed in and returns
the same object. Fine inside `forward`, but it would corrupt a cached reference
activation passed in from `run_with_cache`.

## Why mask at all

Training predicts the next token at every position in one forward pass. Without
the mask, position 3 could attend to position 4 and read the answer it is being
asked to predict — loss collapses, nothing is learned, and the model is useless
at inference where future tokens do not exist.

Masking *before* softmax is also required. Softmax normalises across the row,
so zeroing entries afterwards would leave rows summing to less than 1. Setting
`-inf` first makes `exp(-inf) = 0` participate as a true zero in the
denominator, and the surviving probabilities still sum to exactly 1.

## Run

```
python -m pvml.modules.attention
```

A zeros tensor stands in for real scores, so the output is pure structure:

```
      scores: (1, 1, 4, 4)    # (batch, n_heads, query_pos, key_pos)
    triangle: (4, 4)          # True above the diagonal = cannot attend

masked scores (query_pos down, key_pos across):
tensor([[0., -inf, -inf, -inf],
        [0., 0., -inf, -inf],
        [0., 0., 0., -inf],
        [0., 0., 0., 0.]])

after softmax — each row is a probability distribution:
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000, 0.0000],
        [0.3333, 0.3333, 0.3333, 0.0000],
        [0.2500, 0.2500, 0.2500, 0.2500]])
```

Each row is one query position's access rights: position 0 sees only itself,
position 3 sees everything up to and including itself. With scores all zero,
softmax spreads weight uniformly over whatever is visible.
