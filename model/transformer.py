import torch.nn as nn

from model.transformer_encoder import TransformerEncoder
from model.decoder import DecoderBlock
from model.embeddings import TokenEmbedding
from model.positional_encoding import PositionalEncoding


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        expansion=4,
        max_len=5000
    ):

        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            expansion=expansion,
            max_len=max_len
        )

        self.decoder_embedding = TokenEmbedding(
            tgt_vocab_size,
            embed_dim
        )

        self.decoder_position = PositionalEncoding(
            embed_dim,
            max_len
        )

        self.decoder_layers = nn.ModuleList([
            DecoderBlock(
                embed_dim,
                num_heads,
                expansion
            )
            for _ in range(num_decoder_layers)
        ])

        self.fc_out = nn.Linear(
            embed_dim,
            tgt_vocab_size
        )

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None
    ):

        encoder_output = self.encoder(
            src,
            src_mask
        )

        x = self.decoder_embedding(tgt)

        x = self.decoder_position(x)

        for layer in self.decoder_layers:

            x = layer(
                x,
                encoder_output,
                src_mask,
                tgt_mask
            )

        return self.fc_out(x)