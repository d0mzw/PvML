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
shape trace, gated on `Config.debug`. Run them with `-m`, not by file path,
which breaks the package-relative imports:

```
python -m pvml.config
python -m pvml.modules.normalization
python -m pvml.modules.embedding
python -m pvml.modules.attention
python -m pvml.modules.mlp
python -m pvml.modules.block
python -m pvml.modules.unembedding
```

Each one loads real GPT-2 weights through `pvml.reference.gpt2` and checks its output
against the reference, so the first run downloads ~500MB from the HuggingFace
Hub into `~/.cache/huggingface`.

Each module is documented in [`docs/modules/`](docs/modules/): what the layer
does, its parameter shapes, and the details that are easy to get wrong.

## Models

The trained weights live on the HuggingFace Hub, one folder per run:

**https://huggingface.co/d0mzw/pvml-tinystories**

| Run | n_ctx | Steps | Loss | Accuracy |
| --- | --- | --- | --- | --- |
| [`tinystories-d128-l6-h4-ctx512-40k`](https://huggingface.co/d0mzw/pvml-tinystories/tree/main/tinystories-d128-l6-h4-ctx512-40k) | 512 | 40,000 | 1.732 | 0.573 |
| [`tinystories-d128-l6-h4-ctx128-20k`](https://huggingface.co/d0mzw/pvml-tinystories/tree/main/tinystories-d128-l6-h4-ctx128-20k) | 128 | 20,000 | 2.080 | 0.517 |
| [`tinystories-d128-l6-h4-ctx128-5k`](https://huggingface.co/d0mzw/pvml-tinystories/tree/main/tinystories-d128-l6-h4-ctx128-5k) | 128 | 5,000 | 2.369 | 0.476 |
| [`tinystories-d256-l8-h8-ctx128-5k`](https://huggingface.co/d0mzw/pvml-tinystories/tree/main/tinystories-d256-l8-h8-ctx128-5k) | 128 | 5,000 | 2.497 | 0.461 |
| [`tinystories-d32-l4-h16-ctx128-20k`](https://huggingface.co/d0mzw/pvml-tinystories/tree/main/tinystories-d32-l4-h16-ctx128-20k) | 128 | 20,000 | 2.782 | 0.421 |
| [`tinystories-d32-l4-h16-ctx128-5k`](https://huggingface.co/d0mzw/pvml-tinystories/tree/main/tinystories-d32-l4-h16-ctx128-5k) | 128 | 5,000 | 2.996 | 0.395 |

Each folder holds `model.safetensors`, the `config.json` it was trained with,
and a `summary.json`. Weights are kept out of this repo by `.gitignore`; the
metrics and configs are tracked here under `runs/`.

```
hf download d0mzw/pvml-tinystories --include 'tinystories-d128-l6-h4-ctx512-40k/*' --local-dir runs
python experiments/sample.py runs/tinystories-d128-l6-h4-ctx512-40k
```

### Example output

`python -m pvml.modules.embedding`, with GPT-2's own `W_E` and `W_pos` loaded into
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
