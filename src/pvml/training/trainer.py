import json
import time
from dataclasses import asdict
from pathlib import Path

import torch as t
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pvml.training.args import TrainingArgs
from pvml.training.losses import get_log_probs


class Trainer:
    def __init__(
        self,
        args: TrainingArgs,
        model: t.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        sample_fn=None,
        extra: dict | None = None,
    ):
        """
        The loaders are arguments rather than globals, so this trains on any
        dataset. sample_fn is an optional hook called after each epoch with
        (model, prompt); it exists so text sampling does not require replacing
        the training loop.
        """
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sample_fn = sample_fn

        # from the model, not a module-level global
        self.device = next(model.parameters()).device
        self.optimizer = t.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.step = 0
        self.started = None

        self.run_dir = Path(args.run_dir) / args.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        (self.run_dir / "config.json").write_text(
            json.dumps(
                {"args": asdict(args), "model": asdict(model.cfg), **(extra or {})},
                indent=2,
            )
        )

    def save(self, name: str = "model.pt") -> Path:
        """Weights only. The config sits beside them, so a run is self-contained."""
        path = self.run_dir / name
        t.save(self.model.state_dict(), path)
        return path

    def log(self, **metrics) -> None:
        """One JSON object per line. Plain file, no account, nothing leaves the box."""
        elapsed = None if self.started is None else round(time.perf_counter() - self.started, 2)
        with self.metrics_path.open("a") as f:
            f.write(json.dumps({"step": self.step, "elapsed": elapsed, **metrics}) + "\n")

    def training_step(self, batch) -> t.Tensor:
        """One gradient update. Returns the loss and logs nothing.

        Logging lives in train(), so this can be called on its own to check a
        single batch without a run directory or a live logger.
        """
        tokens = batch["tokens"].to(self.device)
        loss = -get_log_probs(self.model(tokens), tokens).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """Next-token accuracy on the held-out set: how often argmax is right."""
        self.model.eval()
        correct = total = 0

        for batch in tqdm(self.test_loader, desc="evaluating", leave=False):
            tokens = batch["tokens"].to(self.device)
            predictions = self.model(tokens)[:, :-1].argmax(dim=-1)
            correct += (predictions == tokens[:, 1:]).sum().item()
            total += tokens.size(0) * (tokens.size(1) - 1)

        self.model.train()
        return correct / total

    def train(self) -> None:
        self.started = time.perf_counter()
        accuracy = float("nan")
        progress = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                self.log(loss=loss.item(), epoch=epoch)
                progress.update()
                progress.set_description(
                    f"epoch {epoch + 1}  loss {loss:.3f}  accuracy {accuracy:.3f}"
                )
                if i + 1 >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()
            self.log(accuracy=accuracy, epoch=epoch)

            if self.sample_fn is not None:
                sample = self.sample_fn(self.model, self.args.eval_prompt)
                self.log(sample=sample, epoch=epoch)
                print(f"\nepoch {epoch + 1}: {sample!r}")

            # overwritten each epoch, so an interrupted run still leaves weights
            self.save()

        progress.close()

        seconds = time.perf_counter() - self.started
        summary = {
            "device": t.cuda.get_device_name(0) if self.device.type == "cuda" else str(self.device),
            "steps": self.step,
            "seconds": round(seconds, 1),
            "steps_per_second": round(self.step / seconds, 2),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "final_accuracy": accuracy,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print(f"\n{summary['steps']} steps in {seconds / 60:.1f} min "
              f"({summary['steps_per_second']} steps/s)")
        print(f"metrics: {self.metrics_path}")
        print(f"weights: {self.run_dir / 'model.pt'}")
