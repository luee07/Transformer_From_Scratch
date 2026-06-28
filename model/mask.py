import torch


def generate_square_subsequent_mask(size):

    mask = torch.triu(
        torch.ones(size, size),
        diagonal=1
    )

    mask = mask.masked_fill(
        mask == 1,
        float("-inf")
    )

    return mask


def create_padding_mask(sequence, pad_idx=0):

    return (sequence != pad_idx).unsqueeze(1).unsqueeze(2)