import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        embed_dim,
        expansion=4,
        dropout=0.1
    ):

        super().__init__()

        self.linear1 = nn.Linear(
            embed_dim,
            expansion * embed_dim
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(
            expansion * embed_dim,
            embed_dim
        )

    def forward(self, x):

        x = self.linear1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.linear2(x)

        return x