"""The generation loop: run the model, pick one token, append, repeat."""

import torch as t
from jaxtyping import Float, Int
from torch import Tensor

from pvml.sampling.args import SamplingArgs
from pvml.sampling.strategies import (
    apply_frequency_penalty,
    apply_temperature,
    greedy_search,
    sample_basic,
    sample_top_k,
    sample_top_p,
)


class Sampler:
    """Wraps a trained model and its tokenizer.

    The model is never modified. Everything that varies between calls lives in
    SamplingArgs, so one Sampler serves every setting.
    """

    def __init__(self, model: t.nn.Module, tokenizer, prepend_bos: bool = True):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos
        # From the model rather than a global, the same way Trainer does it.
        self.device = next(model.parameters()).device

    @staticmethod
    def next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        args: SamplingArgs,
    ) -> int:
        """Pick one token id. The order matters.

        The penalty goes first because it is a correction to the raw scores.
        Temperature comes next because top_k and top_p read the distribution it
        produces. Greedy short-circuits before temperature, since dividing by
        zero is how temperature 0 would otherwise end.
        """
        if args.frequency_penalty != 0.0:
            logits = apply_frequency_penalty(input_ids, logits, args.frequency_penalty)
        if args.temperature == 0:
            return greedy_search(logits)
        if args.temperature != 1.0:
            logits = apply_temperature(logits, args.temperature)
        if args.top_k > 0:
            return sample_top_k(logits, args.top_k)
        if args.top_p > 0.0:
            return sample_top_p(logits, args.top_p)
        return sample_basic(logits)

    @t.inference_mode()
    def sample(
        self, prompt: str, args: SamplingArgs | None = None, verbose: bool = False
    ) -> str:
        """Generate from a prompt, stopping at max_new_tokens or end-of-text."""
        args = args or SamplingArgs()
        if args.seed is not None:
            t.manual_seed(args.seed)

        was_training = self.model.training
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)[0]

        # Not cosmetic. tokenize_and_concatenate built every training chunk with
        # add_bos_token=True, so the model only ever saw <|endoftext|> at
        # position 0. Start with a real token there and it is off distribution:
        # the TinyStories runs degenerate into a run of commas. GPT-2 itself
        # shrugs this off, which is why a parity check against it will not catch
        # the mistake.
        if self.prepend_bos:
            bos = t.tensor([self.tokenizer.bos_token_id], device=self.device)
            input_ids = t.cat([bos, input_ids], dim=-1)

        try:
            for _ in range(args.max_new_tokens):
                # None makes the batch axis the model wants. The slice is there
                # in case the input tokens run past n_ctx, which they will since
                # the sequence grows every step. [0, -1] drops the batch again
                # and keeps the last position, the only one generation needs.
                logits = self.model(input_ids[None, -self.cfg.n_ctx :])[0, -1]

                next_id = self.next_token(input_ids, logits, args)
                input_ids = t.cat(
                    [input_ids, t.tensor([next_id], device=self.device)], dim=-1
                )

                if verbose:
                    print(self.tokenizer.decode(input_ids), end="\r")
                if next_id == self.tokenizer.eos_token_id:
                    break
        finally:
            self.model.train(was_training)

        return self.tokenizer.decode(input_ids, skip_special_tokens=True)


if __name__ == "__main__":
    from pvml.config import Config
    from pvml.modules.transformer import Transformer
    from pvml.reference.gpt2 import load_reference_gpt2

    ref = load_reference_gpt2()
    device = next(ref.parameters()).device

    model = Transformer(Config(debug=False)).to(device)
    model.load_state_dict(ref.state_dict(), strict=False)
    sampler = Sampler(model, ref.tokenizer)

    # Greedy is deterministic, so this is a real parity check and not a vibe.
    prompt = "Jingle bells, jingle bells, jingle all the way"
    greedy = SamplingArgs(max_new_tokens=8, temperature=0.0)

    ours = sampler.sample(prompt, greedy)
    theirs = ref.generate(prompt, max_new_tokens=8, do_sample=False, verbose=False)
    expected = "Jingle bells, jingle bells, jingle all the way down to the top of the mountain."

    print(f"prompt    {prompt!r}")
    print(f"ours      {ours!r}")
    print(f"reference {theirs!r}")
    print(f"expected  {expected!r}")
    print(f"\nmatches transformer_lens : {ours == theirs}")
    print(f"matches ARENA's expected : {ours == expected}")

    print("\nthe knobs, same prompt and seed")
    for args in [
        SamplingArgs(max_new_tokens=20, temperature=0.0),
        SamplingArgs(max_new_tokens=20, temperature=0.7, seed=0),
        SamplingArgs(max_new_tokens=20, temperature=1.0, top_k=10, seed=0),
        SamplingArgs(max_new_tokens=20, temperature=1.0, top_p=0.95, seed=0),
        SamplingArgs(max_new_tokens=20, temperature=1.0, frequency_penalty=2.0, seed=0),
    ]:
        print(f"\n  {args.describe()}")
        print(f"  {sampler.sample(prompt, args)!r}")
