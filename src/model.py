import torch
import torch.nn as nn
import torchvision.models as models


class SareeEmbeddingModel(nn.Module):

    def __init__(self, embedding_dim=256):

        super().__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Get the number of features produced by ResNet18
        feature_dim = self.backbone.fc.in_features

        # Remove the original ImageNet classification layer
        self.backbone.fc = nn.Identity()

        # Convert ResNet features into a design embedding
        self.embedding = nn.Sequential(

            nn.Linear(
                feature_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.2
            ),

            nn.Linear(
                512,
                embedding_dim
            )
        )


    def forward(self, x):

        # Extract visual features
        features = self.backbone(x)

        # Generate design embedding
        embedding = self.embedding(features)

        # Normalize the embedding
        embedding = nn.functional.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding