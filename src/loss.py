import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):

        super().__init__()

        self.margin = margin

    def forward(self, embedding1, embedding2, label):

        # Calculate Euclidean distance
        distance = torch.nn.functional.pairwise_distance(
            embedding1,
            embedding2
        )

        # Positive pairs:
        # Minimize the distance between embeddings

        positive_loss = label * torch.pow(
            distance,
            2
        )

        # Negative pairs:
        # Push embeddings apart until they reach the margin

        negative_loss = (1 - label) * torch.pow(
            torch.clamp(
                self.margin - distance,
                min=0.0
            ),
            2
        )

        # Average loss
        loss = torch.mean(
            positive_loss + negative_loss
        )

        return loss