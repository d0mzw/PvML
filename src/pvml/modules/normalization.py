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
            print(f"LayerNorm in: {tuple(residual.shape)}")

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
            print(f"  normalised: {tuple(residual.shape)}   <- (batch, posn, d_model) - (batch, posn, 1)")

        # w and b are 1-D (768,). Broadcasting pads missing LEADING axes with
        # 1s, then stretches them:
        #     (2, 10, 768) * (768,) -> (1, 1, 768) -> (2, 10, 768)
        # so the same learned scale and shift is applied at every position.
        out = residual * self.w + self.b
        if self.cfg.debug:
            print(f"        w, b: {tuple(self.w.shape)}   <- padded to (1, 1, d_model), then stretched to (batch, posn, d_model)")
            print(f"         out: {tuple(out.shape)}")
        return out


if __name__ == "__main__":
    t.set_printoptions(precision=2, sci_mode=False)

    # 1. GPT-2 sized — too wide to read, so watch the shapes.
    #    d_model is the width of the residual stream: the vector representing
    #    one token at one position. x is 2 sequences x 10 positions x 768.
    cfg = Config()
    x = t.randn(2, 10, cfg.d_model)
    LayerNorm(cfg)(x)  # debug=True prints the shape trace

    # 2. Small enough to read — watch the numbers instead.
    #    The (1, 2, 1) mean is subtracted from all 4 components of its
    #    position, and both rows land identically: scale is divided out.
    cfg_small = Config(d_model=4, debug=False)
    x_small = t.tensor([[[1.0, 2.0, 3.0, 4.0],
                         [10.0, 20.0, 30.0, 40.0]]])
    mean = x_small.mean(dim=-1, keepdim=True)
    out_small = LayerNorm(cfg_small)(x_small).detach()

    print()
    print(f"x     {tuple(x_small.shape)}\n{x_small}")
    print(f"mean  {tuple(mean.shape)}\n{mean}")
    print(f"out   {tuple(out_small.shape)}\n{out_small}")
