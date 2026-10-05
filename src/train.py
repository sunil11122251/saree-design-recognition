import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pair_torch_dataset import PairSareeDataset
from model import SareeEmbeddingModel
from loss import ContrastiveLoss


# ==================================================
# Configuration
# ==================================================

PAIRS_FILE = "results/train_pairs.csv"
MODEL_DIR = "models"

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.0001

EMBEDDING_DIM = 256
MARGIN = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ==================================================
# Create dataset
# ==================================================

dataset = PairSareeDataset(
    pairs_csv=PAIRS_FILE,
    train=True
)

print("Number of training pairs:", len(dataset))


# ==================================================
# Create DataLoader
# ==================================================

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ==================================================
# Create model
# ==================================================

model = SareeEmbeddingModel(
    embedding_dim=EMBEDDING_DIM
)


# --------------------------------------------------
# Freeze ResNet backbone for initial training
# --------------------------------------------------

for parameter in model.backbone.parameters():
    parameter.requires_grad = False


# Keep embedding layers trainable
for parameter in model.embedding.parameters():
    parameter.requires_grad = True


model = model.to(DEVICE)


# ==================================================
# Loss function
# ==================================================

criterion = ContrastiveLoss(
    margin=MARGIN
)


# ==================================================
# Optimizer
# ==================================================

optimizer = optim.AdamW(
    model.embedding.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ==================================================
# Training
# ==================================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_idx, (image1, image2, labels) in enumerate(dataloader):

        # Move data to device
        image1 = image1.to(DEVICE)
        image2 = image2.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Generate embeddings
        embedding1 = model(image1)
        embedding2 = model(image2)

        # Calculate contrastive loss
        loss = criterion(
            embedding1,
            embedding2,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Print progress
        if (batch_idx + 1) % 20 == 0:

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )

    # Average epoch loss
    epoch_loss = running_loss / len(dataloader)

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}] "
        f"Average Loss: {epoch_loss:.4f}\n"
    )


# ==================================================
# Save model
# ==================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model_path = os.path.join(
    MODEL_DIR,
    "saree_embedding_model.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

print("Training completed.")
print("Model saved to:")
print(model_path)