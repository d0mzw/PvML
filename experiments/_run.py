"""Shared machinery for the experiment files.

Each experiment is a file holding a MODEL and an ARGS, named after the run
directory it writes. This is everything those files have in common.
"""

import torch as t

from pvml.data.tinystories import tinystories_loaders
from pvml.device import get_device
from pvml.modules.transformer import Transformer
from pvml.training.trainer import Trainer


def make_sampler(cfg):
    """Generate from our own weights, using transformer_lens for the loop.

    Our parameter names match theirs, so the weights load straight across and
    the logits come out identical. Sampling properly is its own exercise; this
    borrows a tested loop so the samples say something about the model.
    """
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    hooked = HookedTransformer(
        HookedTransformerConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads, d_head=cfg.d_head,
            d_mlp=cfg.d_mlp, n_layers=cfg.n_layers, n_ctx=cfg.n_ctx,
            d_vocab=cfg.d_vocab, act_fn="gelu_new", normalization_type="LN",
            tokenizer_name="gpt2",
        )
    )

    def sample(model, prompt: str) -> str:
        hooked.load_state_dict(model.state_dict(), strict=False)
        hooked.eval()
        return hooked.generate(
            prompt, max_new_tokens=50, temperature=0.7, top_p=0.95, verbose=False
        )

    return sample


def run(cfg, args, samples: bool = True) -> None:
    t.manual_seed(args.seed)
    device = get_device()
    model = Transformer(cfg).to(device)

    total = sum(p.numel() for p in model.parameters())
    vocab = sum(
        p.numel() for n, p in model.named_parameters()
        if n.endswith(("W_E", "W_U", "W_pos", "b_U"))
    )
    print(f"run        : {args.name}")
    print(f"device     : {device}")
    print(f"parameters : {total:,}  ({total - vocab:,} outside the vocab tables)")
    print(f"steps      : {args.epochs * args.max_steps_per_epoch:,}")

    train_loader, test_loader = tinystories_loaders(
        cfg, batch_size=args.batch_size, seed=args.seed
    )
    Trainer(
        args, model, train_loader, test_loader,
        sample_fn=make_sampler(cfg) if samples else None,
    ).train()
