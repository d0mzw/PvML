import torch as t
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from pvml.config import Config


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        if self.cfg.debug:
            name = type(self).__name__
            print(f"{name + ' in:':>13} {tuple(residual.shape)}")

        # keepdim=True leaves the reduced axis as a size-1 slot so it broadcasts
        # back against the original. Broadcasting aligns from the RIGHT:
        #     (2, 10, 768) - (2, 10, 1)  ->  1 stretches to 768   ok
        #     (2, 10, 768) - (2, 10)     ->  compares 768 vs 10   error
        #
        # unbiased=False divides by N, not N-1. We're standardising these 768
        # numbers, not estimating a population from a sample. Also what GPT-2
        # was trained with, so parity depends on it.
        
        residual_mean = residual.mean(dim=-1, keepdim=True)
        if self.cfg.debug:
            print(f"        mean: {tuple(residual_mean.shape)}")

        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps).sqrt()
        if self.cfg.debug:
            print(f"         std: {tuple(residual_std.shape)}")

        # both ops broadcast the size-1 slot back out to full width:
        #     (2, 10, 768) - (2, 10, 1)  ->  (2, 10, 768)
        #     (2, 10, 768) / (2, 10, 1)  ->  (2, 10, 768)
        # so every position gets standardised by its own mean and std.
        residual = (residual - residual_mean) / residual_std
        if self.cfg.debug:
            print(f"  normalised: {str(tuple(residual.shape)):<16}# (batch, posn, d_model) - (batch, posn, 1)")

        # w and b are 1-D (768,). Broadcasting pads missing LEADING axes with
        # 1s, then stretches them:
        #     (2, 10, 768) * (768,) -> (1, 1, 768) -> (2, 10, 768)
        # so the same learned scale and shift is applied at every position.
        out = residual * self.w + self.b
        if self.cfg.debug:
            print(f"        w, b: {str(tuple(self.w.shape)):<16}# padded to (1, 1, d_model), stretched to out")
            print(f"         out: {tuple(out.shape)}")
        return out


if __name__ == "__main__":
    # d_model is the width of the residual stream: the vector representing
    # one token at one position. x is 2 sequences x 10 positions x 768.
    cfg = Config()
    x = t.randn(2, 10, cfg.d_model)
    LayerNorm(cfg)(x)  # debug=True prints the shape trace
