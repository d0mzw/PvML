"""Train the small transformer on TinyStories.

    python experiments/train_tinystories.py --epochs 10 --steps 500

Metrics land in runs/<name>/metrics.jsonl.
"""

import argparse

import torch as t

from pvml.data.tinystories import tinystories_loaders
from pvml.device import get_device
from pvml.modules.transformer import Transformer
from pvml.training.args import TrainingArgs
from pvml.training.trainer import Trainer, small_config


def make_sampler(cfg):
    """Generate text from our own weights, using transformer_lens for the loop.

    Our parameter names match theirs, so the weights load straight across and
    the logits come out identical. Sampling properly (top-k, top-p, frequency
    penalty, KV caching) is its own exercise and its own post; this borrows a
    tested implementation so the samples say something about the model rather
    than about a generation loop written an hour ago.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500, help="steps per epoch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--no-samples", action="store_true", help="skip text generation")
    cli = parser.parse_args()

    t.manual_seed(cli.seed)
    device = get_device()

    cfg = small_config()
    model = Transformer(cfg).to(device)

    args = TrainingArgs(
        batch_size=cli.batch_size,
        epochs=cli.epochs,
        max_steps_per_epoch=cli.steps,
        lr=cli.lr,
        run_name=cli.run_name,
        eval_prompt=cli.prompt,
    )

    print(f"device     : {device}")
    print(f"parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"steps      : {args.epochs * args.max_steps_per_epoch:,}")

    train_loader, test_loader = tinystories_loaders(
        cfg, batch_size=args.batch_size, seed=cli.seed
    )

    Trainer(
        args,
        model,
        train_loader,
        test_loader,
        sample_fn=None if cli.no_samples else make_sampler(cfg),
    ).train()


if __name__ == "__main__":
    main()
