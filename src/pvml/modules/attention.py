import einops
import torch as t
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from pvml.config import Config


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # One projection per head, stacked on a leading n_heads axis: all 12
        # heads run as a single batched matmul rather than 12 separate modules.
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))

        # W_O is reversed: Q/K/V project the residual stream DOWN into each
        # head's d_head space, W_O projects the result back UP into d_model.
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))

        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))

        # No head axis: added once after the heads are summed back together.
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))

        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)

        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32))

    def forward(
        self,
        normalized_resid_pre: Float[Tensor, "batch posn d_model"],
    ) -> Float[Tensor, "batch posn d_model"]:
        """
        Attention over the residual stream. Expects input already normalised by
        the block's ln1 — attention never normalises its own input.
        """

        # d_model appears in both inputs but not the output, so it is summed
        # over: each position's 768-vector is dotted against each head's
        # (d_model, d_head) matrix, giving d_head numbers per head.
        #     (batch, posn, 768) x (12, 768, 64) -> (batch, posn, 12, 64)
        # b_Q is (12, 64), broadcast to (1, 1, 12, 64): same bias at every
        # position, different per head.
        q = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_Q
        )

        # Same operation, different learned matrices. All three read the SAME
        # residual stream — three views of one vector.
        k = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_V
        )

        if self.cfg.debug:
            name = type(self).__name__
            print(f"{name + ' in:':>13} {str(tuple(normalized_resid_pre.shape)):<16}# (batch, posn, d_model)")
            print(f"       W_QKV: {str(tuple(self.W_Q.shape)):<16}# (n_heads, d_model, d_head)")
            print(f"       b_QKV: {str(tuple(self.b_Q.shape)):<16}# (n_heads, d_head), broadcast over posn")
            print(f"     q, k, v: {str(tuple(q.shape)):<16}# (batch, posn, n_heads, d_head)")

        # batch and nheads appear in both inputs AND the output, so they are
        # batched over, not summed: an independent (posn_Q, posn_K) grid per
        # (sequence, head). Only d_head is contracted. The two position axes
        # get different names so they survive as separate dimensions.
        #     (1, 7, 12, 64) x (1, 7, 12, 64) -> (1, 12, 7, 7)
        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
        )

        # Scale before masking. Dot products of 64-dim vectors have variance
        # ~d_head, so raw scores would saturate softmax to near one-hot and
        # kill the gradients.
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head**0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        if self.cfg.debug:
            print(f"     pattern: {str(tuple(attn_pattern.shape)):<16}# rows sum to 1 over visible keys")

        return attn_pattern

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """

        # A (query_pos, key_pos) square, built fresh each call because the
        # sequence length varies. Created on the input's device: unlike a
        # registered buffer, a tensor made inside forward doesn't follow .to().
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)

        # triu keeps the upper triangle. diagonal=1 starts one above the main
        # diagonal, so the diagonal itself survives as False — a position may
        # attend to itself, just not to anything after it:
        #     [[0, 1, 1, 1],      True  = key_pos > query_pos = the future
        #      [0, 0, 1, 1],      False = visible
        #      [0, 0, 0, 1],
        #      [0, 0, 0, 0]]
        # 2-D, so it broadcasts over batch and n_heads: causality depends only
        # on position, identically for every sequence and every head.
        mask = t.triu(all_ones, diagonal=1).bool()

        if self.cfg.debug:
            print(f"      scores: {str(tuple(attn_scores.shape)):<16}# (batch, n_heads, query_pos, key_pos)")
            print(f"    triangle: {str(tuple(mask.shape)):<16}# True above the diagonal = cannot attend")

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


if __name__ == "__main__":
    cfg = Config()
    attn = Attention(cfg)

    for name, param in attn.named_parameters():
        print(f"{name:>4}: {tuple(param.shape)}")
    total = sum(p.numel() for p in attn.parameters())
    print(f"total: {total:,} parameters\n")

    # A residual stream for 1 sequence of 7 positions, as ln1 would hand over.
    resid = t.randn(1, 7, cfg.d_model)
    attn(resid)

    print()

    # 1 sequence, 1 head, 4 positions — small enough to see the triangle.
    scores = t.randn(1, 1, 4, 4)
    masked = attn.apply_causal_mask(scores)

    print("\nmasked scores (query_pos down, key_pos across):")
    print(masked[0, 0])

    print("\nafter softmax — each row is a probability distribution:")
    print(masked[0, 0].softmax(dim=-1))
