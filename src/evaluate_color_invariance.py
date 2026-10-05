import torch
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from color_pair_torch_dataset import ColorPairSareeDataset
from model import SareeEmbeddingModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/saree_color_invariant_model.pth"

EMBEDDING_DIM = 256

BATCH_SIZE = 8

THRESHOLD = 0.5


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# --------------------------------------------------
# Load model
# --------------------------------------------------

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
# Evaluation function
# --------------------------------------------------

def evaluate_split(pairs_file, split_name):

    print()
    print("=" * 70)
    print(f"Evaluating {split_name}")
    print("=" * 70)

    dataset = ColorPairSareeDataset(
        pairs_csv=pairs_file,
        train=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    distances = []

    labels = []

    with torch.no_grad():

        for image1, image2, batch_labels in dataloader:

            image1 = image1.to(DEVICE)

            image2 = image2.to(DEVICE)

            embedding1 = model(image1)

            embedding2 = model(image2)

            batch_distances = torch.nn.functional.pairwise_distance(
                embedding1,
                embedding2
            )

            distances.extend(
                batch_distances.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )


    distances = np.array(distances)

    labels = np.array(labels)


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    predictions = (
        distances < THRESHOLD
    ).astype(int)


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )


    # Smaller distance = more likely positive
    similarity_score = -distances

    roc_auc = roc_auc_score(
        labels,
        similarity_score
    )


    # --------------------------------------------------
    # Distance statistics
    # --------------------------------------------------

    positive_distances = distances[
        labels == 1
    ]

    negative_distances = distances[
        labels == 0
    ]


    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions
    )


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()

    print("Number of pairs:", len(labels))

    print(
        "Positive pairs:",
        np.sum(labels == 1)
    )

    print(
        "Negative pairs:",
        np.sum(labels == 0)
    )

    print()

    print(
        f"Average positive distance: "
        f"{positive_distances.mean():.4f}"
    )

    print(
        f"Average negative distance: "
        f"{negative_distances.mean():.4f}"
    )

    print()

    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        f"Accuracy:  {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall:    {recall * 100:.2f}%"
    )

    print(
        f"F1-score:  {f1 * 100:.2f}%"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print()

    print("Confusion Matrix:")

    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "positive_distance": positive_distances.mean(),
        "negative_distance": negative_distances.mean()
    }


# --------------------------------------------------
# Validation
# --------------------------------------------------

validation_results = evaluate_split(
    "results/valid_color_pairs.csv",
    "Validation"
)


# --------------------------------------------------
# Test
# --------------------------------------------------

test_results = evaluate_split(
    "results/test_color_pairs.csv",
    "Test"
)


# --------------------------------------------------
# Final comparison
# --------------------------------------------------

print()
print("=" * 70)
print("FINAL COLOR-INVARIANCE EVALUATION")
print("=" * 70)

print()

print("Validation:")
print(
    f"Accuracy: {validation_results['accuracy'] * 100:.2f}%"
)

print(
    f"F1-score: {validation_results['f1'] * 100:.2f}%"
)

print(
    f"ROC-AUC: {validation_results['roc_auc']:.4f}"
)

print()

print("Test:")
print(
    f"Accuracy: {test_results['accuracy'] * 100:.2f}%"
)

print(
    f"F1-score: {test_results['f1'] * 100:.2f}%"
)

print(
    f"ROC-AUC: {test_results['roc_auc']:.4f}"
)