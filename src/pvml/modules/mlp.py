import einops
import torch as t
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from transformer_lens.utilities import gelu_new

from pvml.config import Config
from pvml.debug import trace


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tag = "mlp"

        # 2-D, no head axis: the MLP has no notion of heads. It reads and
        # writes the residual stream at full width, one position at a time.
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))

        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self,
        normalized_resid_mid: Float[Tensor, "batch posn d_model"],
    ) -> Float[Tensor, "batch posn d_model"]:
        """
        Expand to d_mlp, apply the nonlinearity, project back.

        Expects the residual stream after attention has been added, normalised
        by the block's ln2.
        """
        trace(self, "normalized_resid_mid", normalized_resid_mid, "# (batch, posn, d_model)")
        trace(self, "W_in", self.W_in, "# (d_model, d_mlp), contracts d_model", once=True)

        # d_model contracts on the way up. No heads and no position mixing:
        # every position goes through the same matrix, independently.
        pre = (
            einops.einsum(
                normalized_resid_mid,
                self.W_in,
                "batch position d_model, d_model d_mlp -> batch position d_mlp",
            )
            + self.b_in
        )

        trace(self, "pre", pre, "# 4x wider than the residual stream")

        # The only nonlinearity in the block. Everything else is linear, so
        # without this the whole model would collapse to one matmul.
        post = gelu_new(pre)

        trace(self, "post", post, "# gelu_new, same shape")
        trace(self, "W_out", self.W_out, "# (d_mlp, d_model), contracts d_mlp", once=True)

        # d_mlp contracts on the way down, back to the residual stream's width.
        mlp_out = (
            einops.einsum(
                post,
                self.W_out,
                "batch position d_mlp, d_mlp d_model -> batch position d_model",
            )
            + self.b_out
        )

        trace(self, "mlp_out", mlp_out, "# back to (batch, posn, d_model)")

        return mlp_out


if __name__ == "__main__":
    from pvml.loading import load_reference_gpt2

    cfg = Config(debug=True)
    mlp = MLP(cfg)

    for name, param in mlp.named_parameters():
        print(f"{name:>6}: {tuple(param.shape)}")
    print(f" total: {sum(p.numel() for p in mlp.parameters()):,} parameters\n")

    ref = load_reference_gpt2()
    device = next(ref.parameters()).device
    tokens = ref.to_tokens("The cat sat on the mat")
    _, cache = ref.run_with_cache(tokens)

    # The real input: the residual stream after attention, through ln2. Run the
    # layer rather than reading ln2.hook_normalized, which fires BEFORE ln2
    # applies its w and b.
    resid = ref.blocks[0].ln2(cache["blocks.0.hook_resid_mid"])

    mlp = MLP(cfg).to(device)
    mlp.load_state_dict(ref.blocks[0].mlp.state_dict())

    ours = mlp(resid)
    theirs = cache["blocks.0.hook_mlp_out"]
    print(f"\nmax abs diff vs gpt-2: {(ours - theirs).abs().max().item():.3e}")
