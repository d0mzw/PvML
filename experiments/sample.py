"""Prompt a trained model interactively.

    python experiments/sample.py runs/tinystories-d128-l6-h4-ctx512-40k

Type a prompt to generate from it. Commands start with a colon:

    :greedy           always take the top token
    :temp 0.7         sample with temperature (0 is greedy)
    :topk 40          keep the 40 best tokens, 0 to disable
    :topp 0.95        keep the smallest set summing to 0.95, 0 to disable
    :freq 1.0         penalise tokens already generated, 0 to disable
    :len 60           tokens to generate
    :seed 0           fix the draws, blank to unfix
    :settings         show the current sampler
    :help
    :q
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch as t
from transformers import GPT2TokenizerFast

try:
    import readline  # gives the prompt history and line editing
except ImportError:
    readline = None

from pvml.config import Config
from pvml.device import get_device
from pvml.modules.transformer import Transformer
from pvml.sampling.args import SamplingArgs
from pvml.sampling.sampler import Sampler


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


def handle_command(line: str, args: SamplingArgs) -> tuple[SamplingArgs, bool]:
    """Returns the new args and False to quit.

    Every change goes through dataclasses.replace rather than mutating the
    object, so SamplingArgs.__post_init__ revalidates. Setting top_k while
    top_p is live would otherwise pass unchecked.
    """
    cmd, _, rest = line[1:].partition(" ")
    rest = rest.strip()

    if cmd in {"q", "quit", "exit"}:
        return args, False
    try:
        if cmd in {"h", "help"}:
            print(__doc__)
        elif cmd == "greedy":
            args = replace(args, temperature=0.0)
        elif cmd in {"temp", "temperature"}:
            args = replace(args, temperature=float(rest))
        elif cmd == "topk":
            args = replace(args, top_k=int(rest), top_p=0.0)
        elif cmd == "topp":
            args = replace(args, top_p=float(rest), top_k=0)
        elif cmd == "freq":
            args = replace(args, frequency_penalty=float(rest))
        elif cmd == "len":
            args = replace(args, max_new_tokens=int(rest))
        elif cmd == "seed":
            args = replace(args, seed=int(rest) if rest else None)
        elif cmd in {"settings", "set"}:
            pass
        else:
            print(f"  unknown command: :{cmd}   (:help)")
            return args, True
    except (ValueError, AssertionError) as e:
        print(f"  {e}")
        return args, True

    if cmd not in {"h", "help"}:
        print(f"  {args.describe()}" + (f", seed {args.seed}" if args.seed is not None else ""))
    return args, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    cli = parser.parse_args()

    cfg, model = load_run(cli.run_dir)
    sampler = Sampler(model, GPT2TokenizerFast.from_pretrained("gpt2"))
    args = SamplingArgs(max_new_tokens=50, temperature=0.7, top_p=0.95)

    print(f"\n{cli.run_dir}")
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{cfg.n_layers} layers, d_model {cfg.d_model}, n_ctx {cfg.n_ctx}")
    print(f"  {args.describe()}")
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
            args, keep_going = handle_command(line, args)
            if not keep_going:
                break
            continue

        print(f"\n{sampler.sample(line, args)}\n")


if __name__ == "__main__":
    main()
