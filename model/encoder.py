import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward


class EncoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        expansion=4
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            embed_dim,
            num_heads
        )

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.ffn = FeedForward(
            embed_dim,
            expansion
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

    def forward(
    self,
    x,
    mask=None
):

        attention = self.attention(x, mask)

        x = self.norm1(
            x + attention
        )

        forward = self.ffn(x)

        out = self.norm2(
            x + forward
        )

        return out