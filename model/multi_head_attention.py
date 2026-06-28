import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):

    def __init__(self,
                embed_dim,
                num_heads,
                dropout=0.1
            ):

        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
    self,
    encoder,
    decoder,
    mask=None
     ):

        batch_size = decoder.shape[0]
        seq_len = decoder.shape[1]

        Q = self.query(decoder)
        K = self.key(encoder)
        V = self.value(encoder)

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )

        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:

         scores = scores.masked_fill(
            mask == 0,
            float("-inf")
         )

        attention = torch.softmax(
            scores,
            dim=-1
        )
        attention = self.dropout(attention)

        out = torch.matmul(
            attention,
            V
        )

        out = out.transpose(1,2)

        out = out.contiguous().view(
            batch_size,
            seq_len,
            self.embed_dim
        )

        out = self.fc_out(out)

        return out