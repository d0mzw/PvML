import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from pvml.config import Config
from pvml.debug import tag_tree, trace
from pvml.modules.block import TransformerBlock
from pvml.modules.embedding import Embed, PosEmbed
from pvml.modules.normalization import LayerNorm
from pvml.modules.unembedding import Unembed


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tag = "model"

        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        # ModuleList gives the children paths like blocks.3.attn, which is both
        # what tag_tree shortens to b3.attn and what transformer_lens calls its
        # hooks. One vocabulary for the trace and the reference activations.
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

        # re-stamps every child, overwriting the names each block gave its own
        tag_tree(self)

    def forward(
        self,
        tokens: Int[Tensor, "batch posn"],
    ) -> Float[Tensor, "batch posn d_vocab"]:
        """Tokens in, one score per vocabulary entry per position out."""
        trace(self, "tokens", tokens, "# (batch, posn), token ids")

        residual = self.embed(tokens) + self.pos_embed(tokens)

        trace(self, "residual", residual, "# embed + pos_embed")

        # The shape never changes, which is the whole reason blocks stack.
        for block in self.blocks:
            residual = block(residual)

        trace(self, "residual", residual, f"# after {self.cfg.n_layers} blocks")

        # Nothing normalises the stream itself along the way, so it needs one
        # final LayerNorm before the unembedding can read it.
        logits = self.unembed(self.ln_final(residual))

        trace(self, "logits", logits, "# (batch, posn, d_vocab)")

        return logits


if __name__ == "__main__":
    import torch as t

    from pvml.loading import load_reference_gpt2

    ref = load_reference_gpt2()
    device = next(ref.parameters()).device
    text = "The cat sat on the mat"
    tokens = ref.to_tokens(text)

    # Quiet for the parity check: a traced pass is ~220 lines.
    cfg = Config(debug=False)
    model = Transformer(cfg).to(device)
    report = model.load_state_dict(ref.state_dict(), strict=False)
    print(f"parameters      : {sum(p.numel() for p in model.parameters()):,}")
    print(f"missing keys    : {report.missing_keys}")
    print(f"unexpected keys : {len(report.unexpected_keys)} "
          f"({sorted({k.split('.')[-1] for k in report.unexpected_keys})})")

    ours = model(tokens)
    theirs = ref(tokens)
    print(f"\nlogits {tuple(ours.shape)}")
    print(f"max abs diff vs gpt-2: {(ours - theirs).abs().max().item():.3e}")
    print(f"same argmax everywhere: {t.equal(ours.argmax(-1), theirs.argmax(-1))}")

    print("\nnext-token predictions")
    labels = [s.replace("<|endoftext|>", "<BOS>") for s in ref.to_str_tokens(text)]
    for label, guess in zip(labels, ref.to_str_tokens(ours[0].argmax(dim=-1))):
        print(f"  {label:>8}  ->  {guess!r}")
