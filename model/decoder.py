import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.multi_head_cross_attention import MultiHeadCrossAttention
from model.feed_forward import FeedForward


class DecoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        expansion=4
    ):

        super().__init__()

        self.self_attention = MultiHeadAttention(
            embed_dim,
            num_heads
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attention = MultiHeadCrossAttention(
            embed_dim,
            num_heads
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = FeedForward(
            embed_dim,
            expansion
        )

        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x,
        encoder_output,
        src_mask=None,
        tgt_mask=None
    ):

        attention = self.self_attention(
            x,
            tgt_mask
        )

        x = self.norm1(
            x + attention
        )

        cross = self.cross_attention(
            x,
            encoder_output,
            src_mask
        )

        x = self.norm2(
            x + cross
        )

        forward = self.ffn(x)

        return self.norm3(
            x + forward
        )