import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):

    src = [item[0] for item in batch]

    tgt = [item[1] for item in batch]

    src = pad_sequence(
        src,
        batch_first=True,
        padding_value=0
    )

    tgt = pad_sequence(
        tgt,
        batch_first=True,
        padding_value=0
    )

    return src, tgt