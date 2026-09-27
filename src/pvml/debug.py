"""One-line shape tracing, gated on Config.debug.

Every line looks like:

    [b0.attn]     scores: (1, 12, 7, 7)   # (batch, n_heads, query_pos, key_pos)

The tag says which module instance produced it. A module sets `self.tag` in
__init__; a parent overwrites it with the child's position in the tree, so the
same class prints [attn] when run alone and [b3.attn] inside a transformer.
"""

_SEEN = set()

TAG_WIDTH = 12
LABEL_WIDTH = 20
SHAPE_WIDTH = 16


def trace(module, label, value, note="", once=False):
    """Print one shape line, if the module's config has debug on.

    once=True prints a given (class, label) pair the first time only. Use it
    for weights, whose shapes never change: in a 12-block model they would
    otherwise repeat twelve times per forward pass.
    """
    if not getattr(module.cfg, "debug", False):
        return

    tag = getattr(module, "tag", None) or type(module).__name__.lower()
    if once:
        # keyed on the CLASS, not the tag: every block's W_Q has the same
        # shape, so printing it once covers all twelve
        key = (type(module).__name__, label)
        if key in _SEEN:
            return
        _SEEN.add(key)
    shape = tuple(value.shape) if hasattr(value, "shape") else value

    line = (f"{f'[{tag}]':<{TAG_WIDTH}}"
            f"{label:>{LABEL_WIDTH}}: "
            f"{str(shape):<{SHAPE_WIDTH}}"
            f"{note}")
    print(line.rstrip())


def shorten(path):
    """blocks.3.ln1 -> b3.ln1, so the tag column stays narrow."""
    return path.replace("blocks.", "b")


def tag_tree(model):
    """Give every submodule a tag naming its position in the tree.

    Called once by the parent after it has built its children. PyTorch already
    tracks the paths, so nothing here is hand-maintained: rename an attribute
    and the tag follows. The paths match TransformerLens's hook names, so a
    trace line and a run_with_cache key describe the same place.
    """
    for path, module in model.named_modules():
        if path:
            module.tag = shorten(path)
