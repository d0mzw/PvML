import torch as t
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from pvml.config import Config
from pvml.debug import tag_tree, trace
from pvml.modules.attention import Attention
from pvml.modules.mlp import MLP
from pvml.modules.normalization import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tag = "block"

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

        # two LayerNorms in one block, so name the children by position.
        # A Transformer calls tag_tree again later and overwrites these with
        # b0.ln1, b0.attn, and so on.
        tag_tree(self)

    def forward(
        self,
        resid_pre: Float[Tensor, "batch posn d_model"],
    ) -> Float[Tensor, "batch posn d_model"]:
        """
        Attention and the MLP each read the stream and add a delta back.

        Neither replaces it. That additivity is what keeps every component's
        contribution separable, and it is why the residual stream is called a
        stream rather than a pipeline.
        """
        trace(self, "resid_pre", resid_pre, "# (batch, posn, d_model)")

        # ln1 normalises what attention READS. The value added back is the
        # unnormalised attention output, so the stream itself never passes
        # through a LayerNorm.
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre

        trace(self, "resid_mid", resid_mid, "# resid_pre + attn_out")

        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid

        trace(self, "resid_post", resid_post, "# resid_mid + mlp_out")

        return resid_post


if __name__ == "__main__":
    from pvml.models.loading import load_reference_gpt2

    cfg = Config(debug=True)
    block = TransformerBlock(cfg)
    print(f"block parameters: {sum(p.numel() for p in block.parameters()):,}\n")

    ref = load_reference_gpt2()
    device = next(ref.parameters()).device
    tokens = ref.to_tokens("The cat sat on the mat")
    _, cache = ref.run_with_cache(tokens)

    # No ln reconstruction needed here: the block does its own normalising,
    # so the input is the raw residual stream entering block 0.
    resid_pre = cache["blocks.0.hook_resid_pre"]

    block = TransformerBlock(cfg).to(device)
    # strict=False for transformer_lens's precomputed attn.mask buffer
    block.load_state_dict(ref.blocks[0].state_dict(), strict=False)

    resid_post = block(resid_pre)
    theirs = cache["blocks.0.hook_resid_post"]
    print(f"\nmax abs diff vs gpt-2: {(resid_post - theirs).abs().max().item():.3e}")
