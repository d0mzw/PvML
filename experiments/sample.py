"""Prompt a trained model interactively.

    python experiments/sample.py runs/tinystories-5k

Type a prompt to generate from it. Commands start with a colon:

    :greedy           always take the top token
    :temp 0.7         sample with temperature (0 is greedy)
    :topk 40          keep the 40 best tokens, 0 to disable
    :topp 0.95        keep the smallest set summing to 0.95, 0 to disable
    :len 60           tokens to generate
    :settings         show the current sampler
    :help
    :q
"""

import argparse
import json
from pathlib import Path

import torch as t

try:
    import readline  # gives the prompt history and line editing
except ImportError:
    readline = None

from pvml.config import Config
from pvml.device import get_device
from pvml.modules.transformer import Transformer


def load_weights(run_dir: Path) -> dict:
    """Prefer safetensors, which is what the Hub serves, and fall back to the
    model.pt the trainer writes locally."""
    safe = run_dir / "model.safetensors"
    if safe.exists():
        from safetensors.torch import load_file

        return load_file(safe, device=str(get_device()))
    return t.load(run_dir / "model.pt", map_location=get_device())


def load_run(run_dir: Path):
    """Rebuild the model a run directory describes, weights included."""
    saved = json.loads((run_dir / "config.json").read_text())
    cfg = Config(**saved["model"])

    model = Transformer(cfg).to(get_device())
    model.load_state_dict(load_weights(run_dir))
    model.eval()
    return cfg, model


def make_hooked(cfg, model):
    """transformer_lens owns the generation loop; our weights load straight in."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    hooked = HookedTransformer(
        HookedTransformerConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads, d_head=cfg.d_head,
            d_mlp=cfg.d_mlp, n_layers=cfg.n_layers, n_ctx=cfg.n_ctx,
            d_vocab=cfg.d_vocab, act_fn="gelu_new", normalization_type="LN",
            tokenizer_name="gpt2",
        )
    )
    hooked.load_state_dict(model.state_dict(), strict=False)
    hooked.eval()
    return hooked


class Sampler:
    """The knobs, and how they read back."""

    def __init__(self):
        self.greedy = False
        self.temperature = 0.7
        self.top_k = 0
        self.top_p = 0.95
        self.max_new_tokens = 50

    def describe(self) -> str:
        if self.greedy:
            return f"greedy, {self.max_new_tokens} tokens"
        parts = [f"temp {self.temperature}"]
        if self.top_k:
            parts.append(f"top_k {self.top_k}")
        if self.top_p:
            parts.append(f"top_p {self.top_p}")
        parts.append(f"{self.max_new_tokens} tokens")
        return ", ".join(parts)

    def kwargs(self) -> dict:
        if self.greedy:
            return {"do_sample": False, "max_new_tokens": self.max_new_tokens}
        # transformer_lens wants None to disable these, not 0, and it checks
        # top_k before top_p, so only one of them is ever in play
        return {
            "do_sample": True,
            "temperature": self.temperature,
            "top_k": self.top_k or None,
            "top_p": self.top_p or None,
            "max_new_tokens": self.max_new_tokens,
        }


def handle_command(line: str, sampler: Sampler) -> bool:
    """Returns False to quit."""
    cmd, _, rest = line[1:].partition(" ")
    rest = rest.strip()

    if cmd in {"q", "quit", "exit"}:
        return False
    if cmd in {"h", "help"}:
        print(__doc__)
    elif cmd == "greedy":
        sampler.greedy = True
        print(f"  {sampler.describe()}")
    elif cmd in {"temp", "temperature"}:
        sampler.temperature = float(rest)
        sampler.greedy = sampler.temperature == 0
        print(f"  {sampler.describe()}")
    elif cmd == "topk":
        sampler.top_k = int(rest)
        sampler.top_p = 0  # they are alternatives, top_k would win anyway
        sampler.greedy = False
        print(f"  {sampler.describe()}")
    elif cmd == "topp":
        sampler.top_p = float(rest)
        sampler.top_k = 0
        sampler.greedy = False
        print(f"  {sampler.describe()}")
    elif cmd == "len":
        sampler.max_new_tokens = int(rest)
        print(f"  {sampler.describe()}")
    elif cmd in {"settings", "set"}:
        print(f"  {sampler.describe()}")
    else:
        print(f"  unknown command: :{cmd}   (:help)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    cli = parser.parse_args()

    cfg, model = load_run(cli.run_dir)
    hooked = make_hooked(cfg, model)
    sampler = Sampler()

    print(f"\n{cli.run_dir}")
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{cfg.n_layers} layers, d_model {cfg.d_model}")
    print(f"  {sampler.describe()}")
    print("  a prompt generates, :help for commands, :q to quit\n")

    while True:
        try:
            line = input("pvml> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith(":"):
            if not handle_command(line, sampler):
                break
            continue

        with t.inference_mode():
            print(f"\n{hooked.generate(line, verbose=False, **sampler.kwargs())}\n")


if __name__ == "__main__":
    main()
