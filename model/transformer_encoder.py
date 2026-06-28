import torch.nn as nn

from model.embeddings import TokenEmbedding
from model.positional_encoding import PositionalEncoding
from model.encoder import EncoderBlock


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        expansion=4,
        max_len=5000
    ):
        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )

        self.position = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=max_len
        )

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    expansion=expansion
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
    self,
    x,
    mask=None
):

        x = self.embedding(x)

        x = self.position(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x