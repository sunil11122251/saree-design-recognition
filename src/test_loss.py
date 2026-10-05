import torch

from loss import ContrastiveLoss


loss_function = ContrastiveLoss(
    margin=1.0
)


# Example embeddings
embedding1 = torch.tensor([
    [1.0, 0.0, 0.0]
])

embedding2 = torch.tensor([
    [0.9, 0.1, 0.0]
])


# Same design
label = torch.tensor([
    1.0
])


loss = loss_function(
    embedding1,
    embedding2,
    label
)


print("Contrastive loss:")
print(loss)