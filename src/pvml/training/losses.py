import math

import torch as t
from jaxtyping import Float, Int
from torch import Tensor


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"],
    tokens: Int[Tensor, "batch posn"],
) -> Float[Tensor, "batch posn-1"]:
    """Log-probability the model gave to each token that actually came next.

    One value per prediction, not a scalar: the trainer takes -mean() of it,
    but keeping it per position lets you see which positions are hard.
    """
    log_probs = logits.log_softmax(dim=-1)

    # The offset is the whole supervised signal. Predictions from positions
    # 0..n-2 are scored against tokens 1..n-1, so position i is graded on the
    # token that followed it. Swap the slices and it still runs, still returns
    # a number, and trains towards nonsense.
    #
    # gather picks ONE entry per position out of d_vocab: the score the model
    # assigned to the true next token, ignoring the other 50256.
    return (
        log_probs[:, :-1]
        .gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )


if __name__ == "__main__":
    from pvml.config import Config
    from pvml.reference.gpt2 import load_reference_gpt2
    from pvml.modules.transformer import Transformer

    ref = load_reference_gpt2()
    device = next(ref.parameters()).device
    text = "The cat sat on the mat"
    tokens = ref.to_tokens(text)

    cfg = Config(debug=False)
    model = Transformer(cfg).to(device)
    model.load_state_dict(ref.state_dict(), strict=False)

    logits = model(tokens)
    log_probs = get_log_probs(logits, tokens)

    print(f"logits    {tuple(logits.shape)}")
    print(f"log_probs {tuple(log_probs.shape)}   # one per prediction, not per token")
    print()
    print(f"cross entropy, trained model : {-log_probs.mean():.3f} nats")
    print(f"cross entropy, uniform guess : {math.log(cfg.d_vocab):.3f} nats")
    print(f"mean probability of the true next token: {log_probs.exp().mean():.3f}")

    print("\nper prediction")
    labels = [s.replace("<|endoftext|>", "<BOS>") for s in ref.to_str_tokens(text)]
    for i, lp in enumerate(log_probs[0].tolist()):
        print(f"  {labels[i]:>8} -> {labels[i + 1]:<8} logprob {lp:7.3f}   p {math.exp(lp):.4f}")
