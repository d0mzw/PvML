import einops
import torch as t
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from pvml.config import Config
from pvml.debug import trace


class Unembed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tag = "unembed"

        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)

        # GPT-2 has no unembedding bias. It exists so the shapes line up and is
        # frozen at zero, so it never appears in a gradient update.
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)), requires_grad=False)

    def forward(
        self,
        normalized_resid_final: Float[Tensor, "batch posn d_model"],
    ) -> Float[Tensor, "batch posn d_vocab"]:
        """One score per vocabulary entry, per position."""
        trace(self, "normalized_resid_final", normalized_resid_final, "# (batch, posn, d_model)")
        trace(self, "W_U", self.W_U, "# (d_model, d_vocab), the mirror of W_E", once=True)

        logits = (
            einops.einsum(
                normalized_resid_final,
                self.W_U,
                "batch posn d_model, d_model d_vocab -> batch posn d_vocab",
            )
            + self.b_U
        )

        trace(self, "logits", logits, "# one score per vocab entry")

        return logits


if __name__ == "__main__":
    from pvml.loading import load_reference_gpt2

    cfg = Config(debug=True)
    unembed = Unembed(cfg)

    for name, param in unembed.named_parameters():
        print(f"{name:>4}: {tuple(param.shape)}  requires_grad={param.requires_grad}")
    print(f"total: {sum(p.numel() for p in unembed.parameters()):,} parameters\n")

    ref = load_reference_gpt2()
    device = next(ref.parameters()).device
    text = "The cat sat on the mat"
    tokens = ref.to_tokens(text)
    _, cache = ref.run_with_cache(tokens)

    # Run ln_final rather than reading ln_final.hook_normalized, which fires
    # BEFORE the layer applies its w and b.
    resid = ref.ln_final(cache["blocks.11.hook_resid_post"])

    unembed = Unembed(cfg).to(device)
    unembed.load_state_dict(ref.unembed.state_dict())

    logits = unembed(resid)
    theirs = ref(tokens)
    print(f"\nmax abs diff vs gpt-2: {(logits - theirs).abs().max().item():.3e}")

    # What the model actually predicts next, at every position
    print("\nnext-token predictions")
    labels = [s.replace("<|endoftext|>", "<BOS>") for s in ref.to_str_tokens(text)]
    top = logits[0].argmax(dim=-1)
    for label, guess in zip(labels, ref.to_str_tokens(top)):
        print(f"  {label:>8}  ->  {guess!r}")
