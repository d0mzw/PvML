import einops
import torch as t
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from pvml.config import Config


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        # Indexing a 2-D table with a 2-D index tensor gives 3-D: each token id
        # is replaced by its whole row, so the index shape gains a d_model axis.
        #     (d_vocab, d_model)[(batch, posn)] -> (batch, posn, d_model)
        out = self.W_E[tokens]

        if self.cfg.debug:
            name = type(self).__name__
            print(f"{name + ' in:':>13} {str(tuple(tokens.shape)):<16}# token ids, not activations")
            print(f"         W_E: {str(tuple(self.W_E.shape)):<16}# one row per vocab entry")
            print(f"         out: {tuple(out.shape)}")

        return out


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        # Only the SHAPE of tokens is used — position embeddings don't care
        # which token sits where. W_pos always has n_ctx rows, so slice to the
        # sequence length, then repeat the same table for every sequence.
        batch, seq_len = tokens.shape
        sliced = self.W_pos[:seq_len]
        out = einops.repeat(sliced, "seq d_model -> batch seq d_model", batch=batch)

        if self.cfg.debug:
            name = type(self).__name__
            print(f"{name + ' in:':>13} {str(tuple(tokens.shape)):<16}# values ignored, only the shape is read")
            print(f"       W_pos: {str(tuple(self.W_pos.shape)):<16}# one row per position, up to n_ctx")
            print(f"      sliced: {str(tuple(sliced.shape)):<16}# W_pos[:seq_len]")
            print(f"         out: {str(tuple(out.shape)):<16}# same table repeated for each sequence")

        return out


if __name__ == "__main__":
    # transformer_lens is only needed for this check, so import it here rather
    # than at module level — importing pvml.modules.embedding stays cheap.
    from pvml.models.loading import load_reference_gpt2

    ref = load_reference_gpt2()
    text = "The cat sat on the mat"
    tokens = ref.to_tokens(text)

    # GPT-2's BPE: note the leading <|endoftext|> (BOS) and the leading
    # spaces that are part of the tokens themselves.
    print(f"{text=}")
    print(f"{ref.to_str_tokens(text)=}")
    print(f"{tokens.shape=}\n")

    cfg = Config()

    # to_tokens puts the tokens on the reference model's device, so put ours
    # there too rather than assuming CPU.
    device = next(ref.parameters()).device

    embed = Embed(cfg).to(device)
    embed.load_state_dict(ref.embed.state_dict())
    embed(tokens)

    pos_embed = PosEmbed(cfg).to(device)
    pos_embed.load_state_dict(ref.pos_embed.state_dict())
    pos_embed(tokens)

