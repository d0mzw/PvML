from transformer_lens import HookedTransformer


def load_reference_gpt2() -> HookedTransformer:
    """GPT-2 small with weight processing OFF, so the raw weights match ours.

    fold_ln / center_unembed / center_writing_weights all default to True and
    algebraically rearrange the weights. Left on, nothing we load will match.
    """
    return HookedTransformer.from_pretrained(
        "gpt2-small",
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,
    )
