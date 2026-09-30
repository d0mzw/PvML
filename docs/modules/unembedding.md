# Unembed

`src/pvml/modules/unembedding.py`, ported from ARENA 1.1 *Transformer from Scratch*.

Reproduces GPT-2's logits exactly.

## What it does

One score per vocabulary entry, per position. This is where `d_model` becomes
`d_vocab` and the model finally commits to predictions.

## Parameters

| name | shape | note |
|------|-------|------|
| `W_U` | `(d_model, d_vocab)` = `(768, 50257)` | the mirror of `W_E` |
| `b_U` | `(d_vocab,)` | `requires_grad=False` |

**38,647,633 parameters.** `W_E` and `W_U` together are about 62% of GPT-2
small, two matrices that only exist because the vocabulary is large.

`b_U` is frozen because GPT-2 has no unembedding bias. It exists so the shapes
line up and stays at zero.

## Forward

```
"batch posn d_model, d_model d_vocab -> batch posn d_vocab"
```

`d_model` contracts. `d_vocab` appears in only two places in the whole model:
here and at `W_E`. Everything between them is `d_model`.

## Input

`normalized_resid_final` is the residual stream after all twelve blocks,
through `ln_final`. Use `ref.ln_final(cache["blocks.11.hook_resid_post"])`, not
`cache["ln_final.hook_normalized"]`, which fires before the layer applies `w`
and `b`.

## Run

```
python -m pvml.modules.unembedding
```

Prints the trace, the parity result, and what GPT-2 predicts at every position:

```
     <BOS>  ->  '\n'
       The  ->  ' first'
       cat  ->  ' was'
       sat  ->  ' on'
        on  ->  ' the'
       the  ->  ' floor'
       mat  ->  ','
```

Every position predicting its own next token from one forward pass, which is
what the causal mask exists to make possible. `' the'` predicts `' floor'`
because position 5 cannot see `' mat'`.
