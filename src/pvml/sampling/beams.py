"""Beam search: keep the k best partial sequences instead of drawing one token.

The other strategies collapse a distribution to one id and move on. Beam search
carries k candidates forward, so the batch axis of `tokens` is the beams rather
than independent sequences, and it is deterministic: the same prompt gives the
same answer every time.
"""

from dataclasses import dataclass

import einops
import torch as t
from jaxtyping import Float, Int
from torch import Tensor


@dataclass
class Beams:
    """k partial sequences and the summed log-probability of each."""

    model: t.nn.Module
    tokenizer: object
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    def __getitem__(self, idx) -> "Beams":
        """Slice along the beam axis, which is what filter needs."""
        return Beams(self.model, self.tokenizer, self.logprob_sums[idx], self.tokens[idx])

    @property
    def logprobs_and_completions(self) -> list[tuple[float, str]]:
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens, skip_special_tokens=True))
            for logprob_sum, tokens in zip(self.logprob_sums, self.tokens)
        ]

    def generate(self, k: int, no_repeat_ngram_size: int | None = None) -> "Beams":
        """Expand every beam into its k best continuations, giving batch * k beams."""
        n_ctx = self.model.cfg.n_ctx
        logprobs = self.model(self.tokens[:, -n_ctx:])[:, -1, :].log_softmax(-1)
        topk_logprobs, topk_ids = self.get_topk_non_repeating(logprobs, no_repeat_ngram_size, k)

        # Scores are sums of log-probs, so extending a beam is an addition. Working
        # in log space is what makes that true; in probability space it would be a
        # product and would underflow within a dozen tokens.
        new_sums = einops.repeat(self.logprob_sums, "b -> b k", k=k) + topk_logprobs
        new_tokens = t.concat(
            [einops.repeat(self.tokens, "b s -> b k s", k=k), topk_ids.unsqueeze(-1)], dim=-1
        )

        # Flatten (batch, k) back into one beam axis so the next round treats all
        # batch * k candidates equally. Without this the beams stay grouped by
        # parent and a strong parent's weak children beat a weak parent's strong ones.
        return Beams(self.model, self.tokenizer, new_sums.flatten(), new_tokens.flatten(0, 1))

    def filter(self, k: int) -> tuple["Beams", "Beams"]:
        """Split the k best beams into those still running and those that ended."""
        top = self.logprob_sums.topk(k=k, dim=0).indices.tolist()
        terminated = set(
            t.nonzero(self.tokens[:, -1] == self.tokenizer.eos_token_id).flatten().tolist()
        )
        return self[[i for i in top if i not in terminated]], self[
            [i for i in top if i in terminated]
        ]

    def get_topk_non_repeating(
        self,
        logprobs: Float[Tensor, "batch d_vocab"],
        no_repeat_ngram_size: int | None,
        k: int,
    ) -> tuple[Float[Tensor, "batch k"], Int[Tensor, "batch k"]]:
        """topk, with any token that would repeat an n-gram ruled out first.

        Beam search is the setting where repetition bites hardest. Picking the
        highest-scoring continuation every time is exactly the policy that walks
        into a loop, and unlike the sampling strategies there is no randomness to
        break it.
        """
        batch, seq_len = self.tokens.shape

        if no_repeat_ngram_size is not None and seq_len > no_repeat_ngram_size - 1:
            # The n-1 tokens already on the end. Any earlier n-gram starting with
            # the same n-1 tokens tells us which token would complete a repeat.
            prefix = self.tokens[:, seq_len - (no_repeat_ngram_size - 1) :]
            for i in range(seq_len - (no_repeat_ngram_size - 1)):
                ngrams = self.tokens[:, i : i + no_repeat_ngram_size]
                repeated = (ngrams[:, :-1] == prefix).all(-1)
                last = ngrams[:, [-1]]
                logprobs[range(batch), last] = t.where(
                    repeated, -1.0e10, logprobs[range(batch), last]
                )

        return logprobs.topk(k=k, dim=-1)

    def print(self, title: str = "Best completions", max_chars: int = 80) -> None:
        for logprob_sum, text in self.logprobs_and_completions:
            text = text.replace("\n", "\\n")
            print(f"  {logprob_sum:>8.3f}  {text[:max_chars]!r}")


@t.inference_mode()
def beam_search(
    model: t.nn.Module,
    tokenizer,
    prompt: str,
    num_return_sequences: int,
    num_beams: int,
    max_new_tokens: int,
    no_repeat_ngram_size: int | None = None,
    prepend_bos: bool = True,
) -> list[tuple[float, str]]:
    """Alternate generate and filter until enough beams terminate or the cap is hit."""
    assert num_return_sequences <= num_beams, "cannot return more beams than are kept"

    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if prepend_bos:
        bos = t.tensor([[tokenizer.bos_token_id]], device=device)
        tokens = t.cat([bos, tokens], dim=-1)

    was_training = model.training
    model.eval()
    finished: list[tuple[float, str]] = []
    # One beam to start, scoring 0. The first generate() fans it out to k.
    beams = Beams(model, tokenizer, t.tensor([0.0], device=device), tokens)

    try:
        for _ in range(max_new_tokens):
            beams = beams.generate(num_beams, no_repeat_ngram_size)
            beams, terminated = beams.filter(num_beams)
            finished.extend(terminated.logprobs_and_completions)
            if len(finished) >= num_return_sequences:
                return finished[:num_return_sequences]
    finally:
        model.train(was_training)

    # Nothing hit end-of-text, so fall back to the best beams still running.
    finished.extend(beams.logprobs_and_completions)
    return finished[:num_return_sequences]


if __name__ == "__main__":
    from pvml.config import Config
    from pvml.modules.transformer import Transformer
    from pvml.reference.gpt2 import load_reference_gpt2

    ref = load_reference_gpt2()
    device = next(ref.parameters()).device
    model = Transformer(Config(debug=False)).to(device)
    model.load_state_dict(ref.state_dict(), strict=False)

    prompt = "When I was"
    print(f"prompt {prompt!r}\n")
    for n in (1, 3):
        print(f"num_beams={n}")
        for logprob, text in beam_search(model, ref.tokenizer, prompt, n, n, max_new_tokens=12):
            print(f"  {logprob:>8.3f}  {text!r}")
        print()

    print("no_repeat_ngram_size=3, 25 tokens")
    for logprob, text in beam_search(
        model, ref.tokenizer, prompt, 2, 4, max_new_tokens=25, no_repeat_ngram_size=3
    ):
        print(f"  {logprob:>8.3f}  {text!r}")
