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

