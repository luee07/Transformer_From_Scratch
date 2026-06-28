from collections import Counter


class Vocabulary:

    def __init__(
        self,
        min_freq=1
    ):

        self.min_freq = min_freq

        self.stoi = {
            "<PAD>":0,
            "<SOS>":1,
            "<EOS>":2,
            "<UNK>":3
        }

        self.itos = {
            0:"<PAD>",
            1:"<SOS>",
            2:"<EOS>",
            3:"<UNK>"
        }

    def build(self, sentences, tokenizer):

        counter = Counter()

        for sentence in sentences:

            counter.update(
                tokenizer(sentence)
            )

        idx = 4

        for word, freq in counter.items():

            if freq >= self.min_freq:

                self.stoi[word] = idx

                self.itos[idx] = word

                idx += 1

    def numericalize(
        self,
        sentence,
        tokenizer
    ):

        tokens = tokenizer(sentence)

        ids = [
            self.stoi.get(
                token,
                self.stoi["<UNK>"]
            )
            for token in tokens
        ]

        return ids