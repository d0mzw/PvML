# Attention

`src/pvml/modules/attention.py`, ported from ARENA 1.1 *Transformer from Scratch*.

Verified against GPT-2: loading `ref.blocks[0].attn.state_dict()` and feeding
the real `ln1` output reproduces `blocks.0.hook_attn_out` exactly, along with
every intermediate (`q`, `k`, `v`, `pattern`).

## Why attention exists

A language model predicts the next token, and a position often needs
information from somewhere earlier in the sequence:

```
"When Mary and John went to the store, John gave a drink to ___"
```

Answering `Mary` means reaching back and finding the name that isn't John.
LayerNorm and the MLP each see a single position and cannot look sideways.
**Attention is the only mechanism that moves information between positions.**

## Parameters

| name | shape | note |
|------|-------|------|
| `W_Q`, `W_K`, `W_V` | `(n_heads, d_model, d_head)` | project the residual stream down into each head |
| `W_O` | `(n_heads, d_head, d_model)` | reversed: projects back up |
| `b_Q`, `b_K`, `b_V` | `(n_heads, d_head)` | one per head |
| `b_O` | `(d_model,)` | no head axis, added once after the heads are summed |
| `IGNORE` | `()` | registered buffer holding `-inf`, the mask fill value |

2,362,368 parameters per layer, 28.3M across GPT-2 small's twelve.

The leading `n_heads` axis is why all twelve heads run as one batched matmul
rather than a `ModuleList` of twelve modules.

## Forward

```
normalized_resid_pre   (batch, posn, d_model)
   ├── @ W_Q + b_Q  ->  q             (batch, posn, n_heads, d_head)
   ├── @ W_K + b_K  ->  k             (batch, posn, n_heads, d_head)
   └── @ W_V + b_V  ->  v             (batch, posn, n_heads, d_head)

   q . k            ->  attn_scores   (batch, n_heads, posn_Q, posn_K)
        / sqrt(d_head)
        apply_causal_mask
        softmax(-1) ->  attn_pattern  (batch, n_heads, posn_Q, posn_K)

   pattern . v      ->  z             (batch, posn, n_heads, d_head)
   z @ W_O + b_O    ->  attn_out      (batch, posn, d_model)
```

Three contractions, and each one is the operation's meaning:

| step | contracts | what it does |
|------|-----------|--------------|
| `attn_scores` | `d_head` | compares every query to every key |
| `z` | `posn_K` | mixes across positions, the only such step |
| `attn_out` | `n_heads` and `d_head` | sums the heads into one residual update |

`attn_scores` is the only tensor that grows as sequence length squared.

## apply_causal_mask

Sets everything above the diagonal to `-inf` so no position attends to a later
one. `diagonal=1` leaves the diagonal itself unmasked, so a position can attend
to itself; with `diagonal=0`, position 0 would have a fully masked row and
softmax would return `NaN`.

The mask is 2-D and broadcasts over batch and heads, since causality depends
only on position.

**Mask before softmax.** Softmax normalises across the row, so zeroing
afterwards would leave rows summing to less than 1. Setting `-inf` first makes
`exp(-inf) = 0` participate as a true zero in the denominator.

**`masked_fill_` is in place.** It mutates the tensor passed in, which would
corrupt a cached activation handed in from `run_with_cache`.

## Gotchas found while porting

- **`hook_normalized` is not attention's input.** TransformerLens fires that
  hook on `x / scale`, *before* `ln1` applies its `w` and `b`. The real input is
  `ref.blocks[0].ln1(resid_pre)`. Using the hook directly gives a max diff of
  162 with no other symptom.
- **`load_state_dict` needs `strict=False`**, because TransformerLens carries a
  precomputed `mask` buffer that this implementation builds on the fly.
- **ARENA's `register_buffer` passes `device=device`**, reading a module-level
  global this repo does not have. Dropped: registration already makes the
  buffer follow `.to()`.

## Run

```
python -m pvml.modules.attention
```

Prints the parameter shapes, one full trace, and head 0's real attention
pattern on a tokenized sentence:

```
   query / key   <BOS>     The     cat     sat      on     the     mat
         <BOS>    1.00    0.00    0.00    0.00    0.00    0.00    0.00
           The    0.93    0.07    0.00    0.00    0.00    0.00    0.00
           cat    0.71    0.10    0.18    0.00    0.00    0.00    0.00
           sat    0.64    0.14    0.04    0.18    0.00    0.00    0.00
            on    0.48    0.15    0.12    0.23    0.03    0.00    0.00
           the    0.60    0.12    0.09    0.16    0.02    0.02    0.00
           mat    0.37    0.09    0.06    0.03    0.08    0.10    0.26
```

The zeros above the diagonal are the causal mask. The heavy `<BOS>` column is
the attention-sink behaviour: softmax forces each row to sum to 1, so a head
with nothing useful to do parks its weight on a token carrying little
information.
