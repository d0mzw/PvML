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
    """Generate from our own weights with our own loop.

    Rebuilt once per run and reused every epoch. The Sampler holds no state of
    its own beyond the model and tokenizer, so the same one works as the
    weights change underneath it.
    """
    from transformers import GPT2TokenizerFast

    from pvml.sampling.args import SamplingArgs
    from pvml.sampling.sampler import Sampler

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    args = SamplingArgs(max_new_tokens=50, temperature=0.7, top_p=0.95)

    def sample(model, prompt: str) -> str:
        return Sampler(model, tokenizer).sample(prompt, args)

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
