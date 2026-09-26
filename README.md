# PvML
Exploration of AI/ML internals, interpretability, and safety using AMD Strix Halo

## Disclaimer

The code here comes from working through the ARENA 3.0 curriculum. I claim no credit for the original material, and this repository is not affiliated with or endorsed by ARENA.

## Quickstart

First install the ROCm build of PyTorch, then install the package itself. The
second step pulls in the remaining dependencies from `pyproject.toml` and
installs `pvml` in editable mode, so source edits take effect immediately:
```
pip install --pre -r requirements-torch.txt
pip install -e .
```

## Usage

Each module carries its own `__main__` block that runs the layer and prints a
shape trace, gated on `Config.debug`. Run them with `-m` — not by file path,
which breaks the package-relative imports:

```
python -m pvml.config
python -m pvml.modules.normalization
python -m pvml.modules.embedding
python -m pvml.modules.attention
```

`pvml.modules.embedding` loads real GPT-2 weights through
`pvml.models.loading`, so the first run downloads ~500MB from the HuggingFace
Hub into `~/.cache/huggingface`.

Each module is documented in [`docs/modules/`](docs/modules/) — what the layer
does, its parameter shapes, and the details that are easy to get wrong.

### Example output

`python -m pvml.modules.embedding` — GPT-2's own `W_E` and `W_pos` loaded into
`Embed` and `PosEmbed`, run on a tokenized sentence:

```
text='The cat sat on the mat'
ref.to_str_tokens(text)=['<|endoftext|>', 'The', ' cat', ' sat', ' on', ' the', ' mat']
tokens.shape=torch.Size([1, 7])

    Embed in: (1, 7)          # token ids, not activations
         W_E: (50257, 768)    # one row per vocab entry
         out: (1, 7, 768)
 PosEmbed in: (1, 7)          # values ignored, only the shape is read
       W_pos: (1024, 768)     # one row per position, up to n_ctx
      sliced: (7, 768)        # W_pos[:seq_len]
         out: (1, 7, 768)     # same table repeated for each sequence
```
