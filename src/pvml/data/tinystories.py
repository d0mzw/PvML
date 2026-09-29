"""TinyStories, tokenized and chunked into fixed-length training examples."""

import datasets
from torch.utils.data import DataLoader
from transformer_lens.utilities import tokenize_and_concatenate
from transformers import GPT2TokenizerFast

from pvml.config import Config


def tinystories_loaders(
    cfg: Config,
    batch_size: int = 32,
    test_size: int = 1000,
    seed: int | None = None,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader).

    Returned rather than left as a module global, so the trainer takes its
    data as an argument and works with any dataset.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")

    # Not one example per story. The whole corpus is glued into a single token
    # stream and chopped into n_ctx chunks, so every row is exactly the same
    # length, no padding is needed, and a chunk can span a story boundary.
    tokenized = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=cfg.n_ctx,
        column_name="text",
        add_bos_token=True,
        num_proc=8,
    )

    # Shuffles before splitting, so the held-out chunks are not all from the
    # tail of the corpus. Pass seed= to keep the same split across runs.
    split = tokenized.train_test_split(test_size=test_size, seed=seed)

    # ARENA leaves num_workers off because interrupting a worker-backed loader
    # kills a Jupyter kernel. That does not apply to a script, but the data is
    # already tokenized and memory-mapped, so loading is not the bottleneck
    # and extra processes buy little. Exposed as an argument either way.
    train_loader = DataLoader(
        split["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        split["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


if __name__ == "__main__":
    cfg = Config(n_ctx=128)
    train_loader, test_loader = tinystories_loaders(cfg, batch_size=4, seed=0)

    print(f"train chunks : {len(train_loader.dataset):,}")
    print(f"test chunks  : {len(test_loader.dataset):,}")
    print(f"chunk length : {cfg.n_ctx} tokens")

    batch = next(iter(train_loader))
    tokens = batch["tokens"]
    print(f"batch        : {tuple(tokens.shape)}  keys={list(batch.keys())}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print("\nfirst chunk, decoded back to text:\n")
    print(tokenizer.decode(tokens[0]))
