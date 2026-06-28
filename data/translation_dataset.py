import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):

    def __init__(
        self,
        src_sentences,
        tgt_sentences,
        src_vocab,
        tgt_vocab,
        tokenizer
    ):

        self.src = src_sentences
        self.tgt = tgt_sentences

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.src)

    def __getitem__(
        self,
        idx
    ):

        src = self.src_vocab.numericalize(
            self.src[idx],
            self.tokenizer
        )

        tgt = self.tgt_vocab.numericalize(
            self.tgt[idx],
            self.tokenizer
        )

        src = [1] + src + [2]

        tgt = [1] + tgt + [2]

        return (
            torch.tensor(src),
            torch.tensor(tgt)
        )