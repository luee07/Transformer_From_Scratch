import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from model.transformer import Transformer
from model.mask import (
    create_padding_mask,
    generate_square_subsequent_mask
)

from data.tokenizer import tokenize
from data.vocabulary import Vocabulary
from data.translation_dataset import TranslationDataset
from data.collate import collate_fn

from config import *

from utils import save_checkpoint


source_sentences = [

    "i love deep learning",
    "transformers are powerful",
    "attention is all you need",
    "machine learning is fun",
    "hello world",
    "good morning",
    "how are you",
    "i am a student",
    "artificial intelligence is amazing",
    "pytorch is easy to learn",
    "this is my transformer model",
    "we are learning natural language processing",
    "the cat is sleeping",
    "the dog is running",
    "open the door",
    "close the window",
    "i like programming",
    "python is a great language",
    "today is a beautiful day",
    "have a nice day"

]

target_sentences = [

    "j aime l apprentissage profond",
    "les transformateurs sont puissants",
    "l attention est tout ce dont vous avez besoin",
    "l apprentissage automatique est amusant",
    "bonjour le monde",
    "bonjour",
    "comment allez vous",
    "je suis un etudiant",
    "l intelligence artificielle est incroyable",
    "pytorch est facile a apprendre",
    "c est mon modele transformer",
    "nous apprenons le traitement du langage naturel",
    "le chat dort",
    "le chien court",
    "ouvrez la porte",
    "fermez la fenetre",
    "j aime programmer",
    "python est un excellent langage",
    "aujourd hui est une belle journee",
    "bonne journee"

]


src_vocab = Vocabulary()
tgt_vocab = Vocabulary()

src_vocab.build(
    source_sentences,
    tokenize
)

tgt_vocab.build(
    target_sentences,
    tokenize
)


dataset = TranslationDataset(
    source_sentences,
    target_sentences,
    src_vocab,
    tgt_vocab,
    tokenize
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


model = Transformer(
    src_vocab_size=len(src_vocab.stoi),
    tgt_vocab_size=len(tgt_vocab.stoi),
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    expansion=EXPANSION,
    max_len=MAX_LENGTH
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    ignore_index=0,
    label_smoothing=LABEL_SMOOTHING
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)


def initialize_weights(model):

    for parameter in model.parameters():

        if parameter.dim() > 1:

            nn.init.xavier_uniform_(parameter)


initialize_weights(model)


def train_epoch():

    model.train()

    epoch_loss = 0

    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for src, tgt in progress_bar:

        src = src.to(DEVICE)

        tgt = tgt.to(DEVICE)

        optimizer.zero_grad()

        tgt_input = tgt[:, :-1]

        tgt_output = tgt[:, 1:]

        src_mask = create_padding_mask(
            src
        ).to(DEVICE)

        tgt_mask = generate_square_subsequent_mask(
            tgt_input.size(1)
        ).to(DEVICE)

        output = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask
        )

        output = output.reshape(
            -1,
            output.shape[-1]
        )

        tgt_output = tgt_output.reshape(-1)

        loss = criterion(
            output,
            tgt_output
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CLIP
        )

        optimizer.step()

        epoch_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    scheduler.step()

    return epoch_loss / len(loader)

def evaluate():

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for src, tgt in loader:

            src = src.to(DEVICE)

            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:, :-1]

            tgt_output = tgt[:, 1:]

            src_mask = create_padding_mask(
                src
            ).to(DEVICE)

            tgt_mask = generate_square_subsequent_mask(
                tgt_input.size(1)
            ).to(DEVICE)

            output = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask
            )

            output = output.reshape(
                -1,
                output.shape[-1]
            )

            tgt_output = tgt_output.reshape(-1)

            loss = criterion(
                output,
                tgt_output
            )

            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def main():

    print("=" * 60)
    print("Transformer From Scratch Training")
    print("=" * 60)

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):

        train_loss = train_epoch()

        val_loss = evaluate()

        print()

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}]"
        )

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Validation Loss : {val_loss:.4f}"
        )

        print()

        if val_loss < best_loss:

            best_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                epoch,
                CHECKPOINT_PATH
            )

            print("Best model saved.")

        print("-" * 60)

    print()

    print("=" * 60)
    print("Training Finished Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()