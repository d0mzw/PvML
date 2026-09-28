import einops
import torch as t
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from pvml.config import Config
from pvml.debug import trace


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tag = "attn"

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

        trace(self, "normalized_resid_pre", normalized_resid_pre, "# (batch, posn, d_model)")
        trace(self, "W_Q", self.W_Q, "# (n_heads, d_model, d_head)", once=True)
        trace(self, "b_Q", self.b_Q, "# (n_heads, d_head), broadcast over posn", once=True)
        trace(self, "q", q, "# (batch, posn, n_heads, d_head)")
        trace(self, "W_K", self.W_K, once=True)
        trace(self, "b_K", self.b_K, once=True)
        trace(self, "k", k)
        trace(self, "W_V", self.W_V, once=True)
        trace(self, "b_V", self.b_V, once=True)
        trace(self, "v", v)

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

        trace(self, "attn_pattern", attn_pattern, "# rows sum to 1 over visible keys")

        # posn_K is contracted, so this sums over KEY positions: each query
        # position takes a weighted average of every value vector it can see.
        # This is the only step where information crosses between positions.
        #     (1, 7, 12, 64) x (1, 12, 7, 7) -> (1, 7, 12, 64)
        z = einops.einsum(
            v,
            attn_pattern,
            "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head",
        )

        trace(self, "z", z, "# weighted average of v, per query position")

        # Two names contracted at once: nheads and d_head both vanish, so each
        # head's 64 dims are projected up to d_model and the 12 results are
        # SUMMED, not concatenated. Equivalently, glue (nheads d_head) into one
        # axis of 768 and it is a single (posn, 768) @ (768, d_model) matmul.
        # That additivity is why one head's contribution can be isolated.
        attn_out = (
            einops.einsum(
                z,
                self.W_O,
                "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
            )
            + self.b_O
        )

        trace(self, "W_O", self.W_O, "# (n_heads, d_head, d_model), projects back up", once=True)
        trace(self, "attn_out", attn_out, "# back to the residual stream's width")

        return attn_out

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

        trace(self, "attn_scores", attn_scores, "# (batch, n_heads, query_pos, key_pos)")
        trace(self, "mask", mask, "# True above the diagonal = cannot attend")

        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


if __name__ == "__main__":
    # transformer_lens is only needed for this check, so import it here rather
    # than at module level.
    from pvml.loading import load_reference_gpt2

    cfg = Config(debug=True)
    attn = Attention(cfg)

    for name, param in attn.named_parameters():
        print(f"{name:>4}: {tuple(param.shape)}")
    print(f"total: {sum(p.numel() for p in attn.parameters()):,} parameters\n")

    ref = load_reference_gpt2()
    device = next(ref.parameters()).device
    text = "The cat sat on the mat"
    tokens = ref.to_tokens(text)
    _, cache = ref.run_with_cache(tokens)

    # The real input to block 0's attention: the residual stream after ln1.
    # Run the layer rather than reading ln1.hook_normalized, which fires
    # BEFORE ln1 applies its w and b.
    resid = ref.blocks[0].ln1(cache["blocks.0.hook_resid_pre"])

    attn = Attention(cfg).to(device)
    # strict=False: transformer_lens carries an extra `mask` buffer that we
    # build on the fly instead.
    attn.load_state_dict(ref.blocks[0].attn.state_dict(), strict=False)

    attn_out = attn(resid)

    # Head 0's pattern, on real weights. The zeros above the diagonal are the
    # causal mask; each row sums to 1 across the positions it can see.
    # forward already traced itself, and this recomputes part of it, so quieten
    # the trace rather than print the same shapes twice.
    cfg.debug = False
    pattern = attn.apply_causal_mask(
        einops.einsum(
            einops.einsum(resid, attn.W_Q, "b p m, h m d -> b p h d") + attn.b_Q,
            einops.einsum(resid, attn.W_K, "b p m, h m d -> b p h d") + attn.b_K,
            "b q h d, b k h d -> b h q k",
        )
        / cfg.d_head**0.5
    ).softmax(-1)[0, 0]

    labels = [s.replace("<|endoftext|>", "<BOS>") for s in ref.to_str_tokens(text)]
    print("\nhead 0 attention pattern")
    print(f"{'query / key':>14}" + "".join(f"{l:>8}" for l in labels))
    for i, label in enumerate(labels):
        row = "".join(f"{pattern[i][j]:>8.2f}" for j in range(len(labels)))
        print(f"{label:>14}{row}")
