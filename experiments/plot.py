"""Chart the finished runs.

    python experiments/plot.py                      every run in runs/
    python experiments/plot.py runs/a runs/b        just these

Writes into plots/.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run_dir: Path) -> dict | None:
    """A run is only usable once summary.json exists, which is written at the end."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").open()]
    model = json.loads((run_dir / "config.json").read_text())["model"]
    summary = json.loads(summary_path.read_text())

    # The vocabulary tables dominate the parameter count and do not change with
    # depth or width, so the number worth plotting is what is left.
    # W_E and W_U, plus W_pos, plus b_U which is d_vocab long on its own
    vocab = (
        model["d_vocab"] * model["d_model"] * 2
        + model["n_ctx"] * model["d_model"]
        + model["d_vocab"]
    )

    loss = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    return {
        "name": run_dir.name.replace("tinystories-", ""),
        "n_ctx": model["n_ctx"],
        "loss": loss,
        "elapsed": [(r["elapsed"], r["loss"]) for r in rows
                    if "loss" in r and r.get("elapsed") is not None],
        "accuracy": [r["accuracy"] for r in rows if "accuracy" in r],
        "non_embedding": summary["parameters"] - vocab,
        "minutes": summary["seconds"] / 60,
        "final_loss": statistics.mean(l for _, l in loss[-500:]),
    }


def smooth(pairs, window=100):
    """Per-step loss is too noisy to read, so average it into a moving window."""
    xs, ys = zip(*pairs)
    out = []
    for i in range(0, len(ys), window):
        chunk = ys[i : i + window]
        out.append((xs[i + len(chunk) // 2], sum(chunk) / len(chunk)))
    return zip(*out)


def wider_windows(wide, base, out):
    """A run with a longer window, on its own axes.

    Kept apart from the comparison charts on purpose: its loss is lower partly
    because more context makes the next token easier to guess, not only because
    the model is better. The short-window run of the same shape is drawn dashed
    as a reference, not as a like-for-like rival.
    """
    for r in wide:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(*smooth(r["loss"]), label=f"{r['name']} (ctx {r['n_ctx']})")
        twin = next((b for b in base if b["non_embedding"] == r["non_embedding"]), None)
        if twin:
            ax1.plot(*smooth(twin["loss"]), "--", alpha=0.6,
                     label=f"{twin['name']} (ctx {twin['n_ctx']}, different task)")
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss (nats)")
        ax1.set_title("Loss against step")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.plot(range(1, len(r["accuracy"]) + 1), r["accuracy"],
                 marker="o", markersize=3, label=r["name"])
        if twin:
            ax2.plot(range(1, len(twin["accuracy"]) + 1), twin["accuracy"],
                     "--", marker="o", markersize=3, alpha=0.6, label=twin["name"])
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("next-token accuracy")
        ax2.set_title("Accuracy per epoch")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        fig.suptitle(f"n_ctx {r['n_ctx']}: longer window, not a like-for-like comparison")
        fig.savefig(out / f"ctx{r['n_ctx']}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    cli = parser.parse_args()

    dirs = cli.runs or sorted(Path("runs").glob("*"))
    runs = [r for r in (load(d) for d in dirs if d.is_dir()) if r]
    if not runs:
        print("no finished runs found")
        return
    runs.sort(key=lambda r: r["non_embedding"])

    out = Path("plots")
    out.mkdir(exist_ok=True)

    # Predicting a token from 511 tokens of context is an easier problem than
    # from 127, so a longer-window run is not on the same axis as a short one.
    # The comparison charts keep the window most runs share; the others are
    # plotted on their own by wider_windows() below.
    BASE = 128
    wide = [r for r in runs if r["n_ctx"] != BASE]
    runs = [r for r in runs if r["n_ctx"] == BASE] or runs

    # 1. who plateaued and who is still descending
    plt.figure(figsize=(8, 5))
    for r in runs:
        plt.plot(*smooth(r["loss"]), label=r["name"])
    plt.xlabel("step")
    plt.ylabel("loss (nats)")
    plt.title("Loss against step")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.savefig(out / "loss-by-step.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 2. the same runs priced in minutes, which is what you actually spend
    plt.figure(figsize=(8, 5))
    for r in runs:
        if r["elapsed"]:
            xs, ys = smooth(r["elapsed"])
            plt.plot([x / 60 for x in xs], ys, label=r["name"])
    plt.xlabel("minutes")
    plt.ylabel("loss (nats)")
    plt.title("Loss against wall clock")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.savefig(out / "loss-by-time.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 3. capacity against result
    plt.figure(figsize=(7, 5))
    for r in runs:
        plt.scatter(r["non_embedding"], r["final_loss"])
        plt.annotate(r["name"], (r["non_embedding"], r["final_loss"]),
                     fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("non-embedding parameters")
    plt.ylabel("final loss (nats)")
    plt.title("Final loss against capacity")
    plt.grid(alpha=0.3)
    plt.savefig(out / "loss-by-capacity.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 4. accuracy, which reads more easily than nats
    plt.figure(figsize=(8, 5))
    for r in runs:
        plt.plot(range(1, len(r["accuracy"]) + 1), r["accuracy"],
                 marker="o", markersize=3, label=r["name"])
    plt.xlabel("epoch")
    plt.ylabel("next-token accuracy")
    plt.title("Accuracy per epoch")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.savefig(out / "accuracy.png", dpi=140, bbox_inches="tight")
    plt.close()

    if wide:
        wider_windows(wide, runs, out)

    print(f"{len(runs) + len(wide)} runs plotted into {out}/")
    for r in runs + wide:
        print(f"  {r['name']:<30} {r['non_embedding']:>9,} non-emb  "
              f"loss {r['final_loss']:.3f}  {r['minutes']:.1f} min")


if __name__ == "__main__":
    main()
