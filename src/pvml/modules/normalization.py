import torch as t
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from pvml.config import Config
from pvml.debug import trace


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tag = "ln"
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        trace(self, "residual", residual, "# as it arrives")

        # keepdim=True leaves the reduced axis as a size-1 slot so it broadcasts
        # back against the original. Broadcasting aligns from the RIGHT:
        #     (2, 10, 768) - (2, 10, 1)  ->  1 stretches to 768   ok
        #     (2, 10, 768) - (2, 10)     ->  compares 768 vs 10   error
        #
        # unbiased=False divides by N, not N-1. We're standardising these 768
        # numbers, not estimating a population from a sample. Also what GPT-2
        # was trained with, so parity depends on it.
        
        residual_mean = residual.mean(dim=-1, keepdim=True)
        trace(self, "residual_mean", residual_mean)

        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()
        trace(self, "residual_std", residual_std)

        # both ops broadcast the size-1 slot back out to full width:
        #     (2, 10, 768) - (2, 10, 1)  ->  (2, 10, 768)
        #     (2, 10, 768) / (2, 10, 1)  ->  (2, 10, 768)
        # so every position gets standardised by its own mean and std.
        residual = (residual - residual_mean) / residual_std
        trace(self, "residual", residual, "# reassigned: (residual - mean) / std")

        # w and b are 1-D (768,). Broadcasting pads missing LEADING axes with
        # 1s, then stretches them:
        #     (2, 10, 768) * (768,) -> (1, 1, 768) -> (2, 10, 768)
        # so the same learned scale and shift is applied at every position.
        out = residual * self.w + self.b
        trace(self, "w, b", self.w, "# padded to (1, 1, d_model), stretched to out", once=True)
        trace(self, "out", out, "# * w + b")
        return out


if __name__ == "__main__":
    # d_model is the width of the residual stream: the vector representing
    # one token at one position. x is 2 sequences x 10 positions x 768.
    cfg = Config(debug=True)
    x = t.randn(2, 10, cfg.d_model)
    LayerNorm(cfg)(x)  # debug=True prints the shape trace
