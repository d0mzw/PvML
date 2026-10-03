"""Ways to turn one vector of logits into one token id.

Every function here takes the logits for a single position, with no model, no
tokenizer and no batch axis, so each one can be checked on a hand-written
vector. They are variations on one idea: reshape the distribution, then draw
from it. Only `greedy_search` makes no draw at all.
"""

import torch as t
from jaxtyping import Float, Int
from torch import Tensor


def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
    """The most likely token. No randomness, so the same prompt always continues
    the same way."""
    return logits.argmax().item()


def apply_temperature(
    logits: Float[Tensor, "d_vocab"], temperature: float
) -> Float[Tensor, "d_vocab"]:
    """Divide before the softmax. Below 1 sharpens the distribution, above 1
    flattens it.

    Temperature 0 is not handled here: it would divide by zero. The dispatcher
    routes it to greedy_search instead, which is the limit this approaches.
    """
    return logits / temperature


def apply_frequency_penalty(
    input_ids: Int[Tensor, "seq_len"],
    logits: Float[Tensor, "d_vocab"],
    freq_penalty: float,
) -> Float[Tensor, "d_vocab"]:
    """Subtract a cost proportional to how often each token already appeared.

    Acts on the logits, so it discourages repetition before any draw is made
    rather than rejecting one afterwards.
    """
    # minlength is what lines the counts up with the logits. Without it bincount
    # stops at the largest id present and the subtraction fails to broadcast,
    # or worse, silently penalises the wrong tokens.
    counts = t.bincount(input_ids, minlength=logits.size(0))
    return logits - freq_penalty * counts


def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
    """Draw from the full distribution, all 50257 tokens in play."""
    return t.distributions.categorical.Categorical(logits=logits).sample().item()


def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
    """Keep the k most likely tokens, renormalise, draw from those."""
    top_logits, top_ids = logits.topk(k)

    # Categorical renormalises whatever it is given, so passing the k surviving
    # logits is already the renormalisation. The index that comes back points
    # into the top-k list, not into the vocabulary, so it has to be mapped back.
    sampled = t.distributions.categorical.Categorical(logits=top_logits).sample()
    return top_ids[sampled].item()


def sample_top_p(
    logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1
) -> int:
    """Keep the smallest set of tokens whose probabilities sum to top_p.

    Unlike top-k the size of that set varies with the position. Where the model
    is confident a couple of tokens reach the threshold; where it is unsure it
    may take hundreds.
    """
    sorted_logits, sorted_ids = logits.sort(descending=True, stable=True)
    cumul_probs = sorted_logits.softmax(-1).cumsum(-1)

    # searchsorted finds the first index whose cumulative probability reaches
    # top_p. The +1 turns that index into a count, so the token that crosses the
    # threshold is kept rather than dropped.
    n_keep = t.searchsorted(cumul_probs, top_p, side="left").item() + 1
    n_keep = max(n_keep, min_tokens_to_keep)

    keep_ids = sorted_ids[:n_keep]
    sampled = t.distributions.categorical.Categorical(logits=logits[keep_ids]).sample()
    return keep_ids[sampled].item()


if __name__ == "__main__":
    # A hand-written distribution, so every number below can be checked by eye.
    logits = t.tensor([3.0, 2.0, 1.0, 0.0, -1.0])
    probs = logits.softmax(-1)
    print("logits ", logits.tolist())
    print("probs  ", [f"{p:.3f}" for p in probs.tolist()])
    print(f"cumulative {[f'{c:.3f}' for c in probs.cumsum(-1).tolist()]}\n")

    print(f"greedy_search              -> {greedy_search(logits)}   # the argmax")
    print(f"apply_temperature(0.5)     -> {apply_temperature(logits, 0.5).tolist()}")
    print(f"apply_temperature(2.0)     -> {apply_temperature(logits, 2.0).tolist()}")

    seen = t.tensor([0, 0, 0, 1])
    print(f"\nfrequency_penalty, token 0 seen 3x, token 1 once, penalty 1.0")
    print(f"  {apply_frequency_penalty(seen, logits, 1.0).tolist()}")

    # The draws are random, so count them instead of printing one.
    t.manual_seed(0)
    for name, fn in [
        ("sample_basic", lambda: sample_basic(logits)),
        ("sample_top_k(2)", lambda: sample_top_k(logits, 2)),
        ("sample_top_p(0.7)", lambda: sample_top_p(logits, 0.7)),
    ]:
        counts = t.bincount(t.tensor([fn() for _ in range(10_000)]), minlength=5)
        print(f"\n{name} over 10,000 draws")
        print(f"  {[f'{c / 10_000:.3f}' for c in counts.tolist()]}")
