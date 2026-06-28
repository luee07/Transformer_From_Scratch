import torch

from data.tokenizer import tokenize


def greedy_decode(
    model,
    sentence,
    src_vocab,
    tgt_vocab,
    device,
    max_length=50
):

    model.eval()

    tokens = tokenize(sentence)

    src = [1]

    src.extend(
        src_vocab.numericalize(
            sentence,
            tokenize
        )
    )

    src.append(2)

    src = torch.tensor(src).unsqueeze(0).to(device)

    generated = [1]

    for _ in range(max_length):

        tgt = torch.tensor(generated).unsqueeze(0).to(device)

        with torch.no_grad():

            output = model(
                src,
                tgt
            )

        next_token = output.argmax(-1)[0, -1].item()

        generated.append(next_token)

        if next_token == 2:
            break

    words = []

    for token in generated:

        if token in [0, 1, 2]:
            continue

        words.append(
            tgt_vocab.itos.get(
                token,
                "<UNK>"
            )
        )

    return " ".join(words)