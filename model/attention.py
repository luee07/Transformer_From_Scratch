import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, embed_dim):

        super().__init__()

        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)

        self.key = nn.Linear(embed_dim, embed_dim)

        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.embed_dim
        )

        if mask is not None:

         scores = scores.masked_fill(
        mask == 0,
        float("-inf")
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention,
            V
        )

        return output