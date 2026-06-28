import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SRC_VOCAB_SIZE = 5000
TGT_VOCAB_SIZE = 5000

EMBED_DIM = 256

NUM_HEADS = 8

NUM_ENCODER_LAYERS = 4

NUM_DECODER_LAYERS = 4

EXPANSION = 4

MAX_LENGTH = 200

BATCH_SIZE = 32

LEARNING_RATE = 3e-4

NUM_EPOCHS = 20

LABEL_SMOOTHING = 0.1

CLIP = 1.0

CHECKPOINT_PATH = "checkpoints/model.pt"