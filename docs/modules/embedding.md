# Embed and PosEmbed

`src/pvml/modules/embedding.py` — ported from ARENA 1.1, `exercise.py:168-198`.

These are the two layers that turn integers into vectors. Their outputs are
summed to form the initial residual stream:

```
residual = Embed(tokens) + PosEmbed(tokens)
```

Both take `Int[Tensor, "batch position"]` and return
`Float[Tensor, "batch position d_model"]`. That type change — ints in, floats
out — is the signature of an embedding layer.

## Embed

| name | shape | role |
|------|-------|------|
| `W_E` | `(d_vocab, d_model)` = `(50257, 768)` | one row per vocabulary entry |

`forward` is a single lookup: `self.W_E[tokens]`. No matrix multiply.

The shape jump is worth understanding. Indexing a 2-D table with a 2-D index
tensor gives 3-D:

```
(d_vocab, d_model)[(batch, posn)]  ->  (batch, posn, d_model)
```

Each token id is replaced by its entire row, so the index shape is preserved
and the row axis is appended.

At 50257 x 768 that is 38.6M parameters — roughly 30% of GPT-2 small's total,
spent purely on mapping ids to vectors.

## PosEmbed

| name | shape | role |
|------|-------|------|
| `W_pos` | `(n_ctx, d_model)` = `(1024, 768)` | one row per position |

`forward` reads only `tokens.shape`; the token *values* are ignored entirely.
Position embeddings encode where a token sits, not which token it is.

```python
batch, seq_len = tokens.shape
sliced = self.W_pos[:seq_len]
out = einops.repeat(sliced, "seq d_model -> batch seq d_model", batch=batch)
```

`W_pos` always has `n_ctx` rows regardless of input length, so it is sliced to
`seq_len` and then repeated across the batch — every sequence gets identical
position vectors, because position 3 means the same thing everywhere.

`n_ctx = 1024` is a hard architectural ceiling, not a configuration
preference: there is no row 1025, so GPT-2 cannot attend beyond 1024
positions.

## Tokenization

GPT-2 uses byte-pair encoding. `d_vocab = 50257` is 256 byte tokens + 50,000
learned merges + 1 special token (`<|endoftext|>`).

```
'The cat sat on the mat'
  -> ['<|endoftext|>', 'The', ' cat', ' sat', ' on', ' the', ' mat']
```

Two things to notice: a BOS token is prepended, and leading spaces belong to
the token (`' cat'` and `'cat'` are different ids with different embedding
rows). Attaching the space to the following word makes encoding losslessly
reversible.

## Loading GPT-2 weights

`__main__` loads real weights through `pvml.models.loading.load_reference_gpt2`
and runs the layers on a tokenized sentence. `load_state_dict` accepts them
because the parameter names match TransformerLens's (`W_E`, `W_pos`) — a
reason to keep ARENA's naming verbatim.

The reference model lands on GPU, so the modules are moved to match:

```python
device = next(ref.parameters()).device
```

derived from the model rather than read from a global. This is the first place
that mattered: `ref.to_tokens()` returns tokens on the reference model's
device, and CPU modules cannot index them.

## Run

```
python -m pvml.modules.embedding
```

```
    Embed in: (1, 7)          # token ids, not activations
         W_E: (50257, 768)    # one row per vocab entry
         out: (1, 7, 768)
 PosEmbed in: (1, 7)          # values ignored, only the shape is read
       W_pos: (1024, 768)     # one row per position, up to n_ctx
      sliced: (7, 768)        # W_pos[:seq_len]
         out: (1, 7, 768)     # same table repeated for each sequence
```
