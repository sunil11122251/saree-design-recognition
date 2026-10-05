import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model import SareeEmbeddingModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/saree_color_invariant_model.pth"

TEST_DIR = "data/color_variants/test"

EMBEDDING_DIM = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Image transformation
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Device:", DEVICE)

model = SareeEmbeddingModel(
    embedding_dim=EMBEDDING_DIM
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)

model.eval()


# --------------------------------------------------
# Get embedding
# --------------------------------------------------

def get_embedding(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        embedding = model(image)

    return embedding.squeeze(0).cpu()


# --------------------------------------------------
# Find test groups
# --------------------------------------------------

gallery = []

import os

for class_name in os.listdir(TEST_DIR):

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    if not os.path.isdir(class_dir):
        continue

    for group_id in os.listdir(class_dir):

        group_dir = os.path.join(
            class_dir,
            group_id
        )

        if not os.path.isdir(group_dir):
            continue

        original_path = os.path.join(
            group_dir,
            "original.jpg"
        )

        variant_paths = []

        for number in [1, 2, 3]:

            path = os.path.join(
                group_dir,
                f"color_variant_{number}.jpg"
            )

            if os.path.exists(path):

                variant_paths.append(path)

        if (
            os.path.exists(original_path)
            and len(variant_paths) == 3
        ):

            gallery.append({
                "group_id": group_id,
                "class": class_name,
                "original": original_path,
                "variants": variant_paths
            })


print()
print("=" * 70)
print("GALLERY INFORMATION")
print("=" * 70)

print(
    "Number of design groups:",
    len(gallery)
)


# --------------------------------------------------
# Create gallery embeddings
# --------------------------------------------------

print()
print("Creating gallery embeddings...")

gallery_embeddings = []

for item in gallery:

    embedding = get_embedding(
        item["original"]
    )

    gallery_embeddings.append(
        embedding
    )

gallery_embeddings = torch.stack(
    gallery_embeddings
)

print(
    "Gallery embedding shape:",
    gallery_embeddings.shape
)


# --------------------------------------------------
# Retrieval evaluation
# --------------------------------------------------

recall_at_1 = 0

recall_at_5 = 0

total_queries = 0

correct_distances = []


for index, item in enumerate(gallery):

    correct_group = item["group_id"]

    for query_path in item["variants"]:

        query_embedding = get_embedding(
            query_path
        )

        # ------------------------------------------
        # Calculate distances
        # ------------------------------------------

        distances = torch.nn.functional.pairwise_distance(
            query_embedding.unsqueeze(0),
            gallery_embeddings
        )

        # ------------------------------------------
        # Rank gallery
        # ------------------------------------------

        ranked_indices = torch.argsort(
            distances
        ).tolist()

        ranked_groups = [
            gallery[i]["group_id"]
            for i in ranked_indices
        ]

        # ------------------------------------------
        # Rank of correct design
        # ------------------------------------------

        correct_rank = (
            ranked_groups.index(correct_group) + 1
        )

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        if correct_rank == 1:

            recall_at_1 += 1

        if correct_rank <= 5:

            recall_at_5 += 1

        correct_distances.append(
            distances[index].item()
        )

        total_queries += 1


# --------------------------------------------------
# Final metrics
# --------------------------------------------------

recall1 = (
    recall_at_1 / total_queries
)

recall5 = (
    recall_at_5 / total_queries
)

average_correct_distance = np.mean(
    correct_distances
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("=" * 70)
print("COLOR-INVARIANT RETRIEVAL RESULTS")
print("=" * 70)

print()

print(
    "Number of queries:",
    total_queries
)

print(
    "Gallery size:",
    len(gallery)
)

print()

print(
    f"Recall@1: "
    f"{recall1 * 100:.2f}%"
)

print(
    f"Recall@5: "
    f"{recall5 * 100:.2f}%"
)

print(
    f"Average correct-design distance: "
    f"{average_correct_distance:.4f}"
)

print()
print("=" * 70)
print("RETRIEVAL EVALUATION COMPLETED")
print("=" * 70)