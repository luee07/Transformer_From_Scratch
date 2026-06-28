from nltk.translate.bleu_score import corpus_bleu


def compute_bleu(
    references,
    predictions
):

    refs = [[r.split()] for r in references]

    preds = [p.split() for p in predictions]

    return corpus_bleu(
        refs,
        preds
    )