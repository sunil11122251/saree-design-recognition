import torch

from model import SareeEmbeddingModel


# Create model
model = SareeEmbeddingModel(
    embedding_dim=256
)


# Create a fake batch of 2 images
images = torch.randn(
    2,
    3,
    224,
    224
)


# Generate embeddings
embeddings = model(images)


print("Input shape:")
print(images.shape)

print("\nEmbedding shape:")
print(embeddings.shape)

print("\nEmbedding norm:")
print(torch.norm(embeddings, dim=1))