import torch
import torch.nn as nn


def initialize_weights(model):

    for module in model.modules():

        if isinstance(module, nn.Linear):

            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:

                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0,
                std=0.02
            )


def save_checkpoint(model, optimizer, epoch, path):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict()

    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):

    checkpoint = torch.load(path)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    print("Checkpoint Loaded")

    return checkpoint["epoch"]

