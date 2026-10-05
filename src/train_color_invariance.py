import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from color_pair_torch_dataset import ColorPairSareeDataset
from model import SareeEmbeddingModel
from loss import ContrastiveLoss


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PAIRS_FILE = "results/train_color_pairs.csv"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "saree_color_invariant_model.pth"
)

BATCH_SIZE = 8

EPOCHS = 5

LEARNING_RATE = 0.0001

EMBEDDING_DIM = 256

MARGIN = 1.0


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = ColorPairSareeDataset(
    pairs_csv=PAIRS_FILE,
    train=True
)

print(
    "Number of color-invariance training pairs:",
    len(dataset)
)


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = SareeEmbeddingModel(
    embedding_dim=EMBEDDING_DIM
)


# --------------------------------------------------
# Freeze pretrained backbone
# --------------------------------------------------

for parameter in model.backbone.parameters():

    parameter.requires_grad = False


# Train only embedding head
for parameter in model.embedding.parameters():

    parameter.requires_grad = True


model = model.to(DEVICE)


# --------------------------------------------------
# Loss
# --------------------------------------------------

criterion = ContrastiveLoss(
    margin=MARGIN
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = optim.AdamW(
    model.embedding.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_idx, (
        image1,
        image2,
        labels
    ) in enumerate(dataloader):

        image1 = image1.to(DEVICE)

        image2 = image2.to(DEVICE)

        labels = labels.to(DEVICE)


        # -------------------------------
        # Clear gradients
        # -------------------------------

        optimizer.zero_grad()


        # -------------------------------
        # Generate embeddings
        # -------------------------------

        embedding1 = model(image1)

        embedding2 = model(image2)


        # -------------------------------
        # Calculate loss
        # -------------------------------

        loss = criterion(
            embedding1,
            embedding2,
            labels
        )


        # -------------------------------
        # Backpropagation
        # -------------------------------

        loss.backward()

        optimizer.step()


        running_loss += loss.item()


        # -------------------------------
        # Progress
        # -------------------------------

        if (batch_idx + 1) % 50 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )


    # -------------------------------
    # Epoch loss
    # -------------------------------

    epoch_loss = (
        running_loss / len(dataloader)
    )

    print()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Average Loss: {epoch_loss:.4f}"
    )

    print()


# --------------------------------------------------
# Save model
# --------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


torch.save(
    model.state_dict(),
    MODEL_PATH
)


print("=" * 60)

print("COLOR-INVARIANCE TRAINING COMPLETED")

print("=" * 60)

print()

print("Model saved to:")

print(MODEL_PATH)