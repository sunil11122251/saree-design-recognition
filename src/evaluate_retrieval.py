import torch
from torch.utils.data import DataLoader

from pair_torch_dataset import PairSareeDataset
from model import SareeEmbeddingModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


MODEL_FILE = "models/saree_embedding_model.pth"

BATCH_SIZE = 16
EMBEDDING_DIM = 256

# Distance below this value = same
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate_model(pairs_file, dataset_name):

    print("\n" + "=" * 60)
    print(f"Evaluating on {dataset_name}")
    print("=" * 60)

    # -----------------------------------------
    # LOAD DATASET
    # -----------------------------------------

    dataset = PairSareeDataset(
        pairs_csv=pairs_file,
        train=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------
    # LOAD MODEL
    # -----------------------------------------

    model = SareeEmbeddingModel(
        embedding_dim=EMBEDDING_DIM
    )

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    all_distances = []
    all_labels = []

    # -----------------------------------------
    # GENERATE EMBEDDINGS
    # -----------------------------------------

    with torch.no_grad():

        for image1, image2, labels in dataloader:

            image1 = image1.to(DEVICE)
            image2 = image2.to(DEVICE)

            embedding1 = model(image1)
            embedding2 = model(image2)

            distances = torch.nn.functional.pairwise_distance(
                embedding1,
                embedding2
            )

            all_distances.extend(
                distances.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    # -----------------------------------------
    # CONVERT DISTANCE TO SIMILARITY SCORE
    # -----------------------------------------

    # Smaller distance = more similar.
    #
    # For ROC-AUC, use negative distance because
    # higher score should mean "more likely same".

    similarity_scores = [
        -distance
        for distance in all_distances
    ]

    # -----------------------------------------
    # PREDICTIONS
    # -----------------------------------------

    predictions = [
        1 if distance < THRESHOLD else 0
        for distance in all_distances
    ]

    # -----------------------------------------
    # METRICS
    # -----------------------------------------

    accuracy = accuracy_score(
        all_labels,
        predictions
    )

    precision = precision_score(
        all_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        all_labels,
        similarity_scores
    )

    cm = confusion_matrix(
        all_labels,
        predictions
    )

    # -----------------------------------------
    # DISTANCE STATISTICS
    # -----------------------------------------

    positive_distances = [
        distance
        for distance, label in zip(
            all_distances,
            all_labels
        )
        if label == 1
    ]

    negative_distances = [
        distance
        for distance, label in zip(
            all_distances,
            all_labels
        )
        if label == 0
    ]

    positive_average = (
        sum(positive_distances)
        / len(positive_distances)
    )

    negative_average = (
        sum(negative_distances)
        / len(negative_distances)
    )

    # -----------------------------------------
    # PRINT RESULTS
    # -----------------------------------------

    print("\nNumber of positive pairs:", len(positive_distances))
    print("Number of negative pairs:", len(negative_distances))

    print("\nAverage positive distance:")
    print(f"{positive_average:.4f}")

    print("\nAverage negative distance:")
    print(f"{negative_average:.4f}")

    print("\nThreshold:")
    print(THRESHOLD)

    print("\nAccuracy:")
    print(f"{accuracy * 100:.2f}%")

    print("\nPrecision:")
    print(f"{precision * 100:.2f}%")

    print("\nRecall:")
    print(f"{recall * 100:.2f}%")

    print("\nF1-score:")
    print(f"{f1 * 100:.2f}%")

    print("\nROC-AUC:")
    print(f"{roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "positive_distance": positive_average,
        "negative_distance": negative_average,
        "confusion_matrix": cm
    }


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("Device:", DEVICE)

    print("\nLoading trained model:")
    print(MODEL_FILE)

    # -----------------------------------------
    # VALIDATION
    # -----------------------------------------

    validation_results = evaluate_model(
        "results/valid_pairs.csv",
        "VALIDATION SET"
    )

    # -----------------------------------------
    # TEST
    # -----------------------------------------

    test_results = evaluate_model(
        "results/test_pairs.csv",
        "TEST SET"
    )

    # -----------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)

    print("\n                 Validation     Test")
    print("-" * 45)

    print(
        f"Accuracy         "
        f"{validation_results['accuracy'] * 100:>8.2f}%"
        f"     "
        f"{test_results['accuracy'] * 100:>8.2f}%"
    )

    print(
        f"Precision        "
        f"{validation_results['precision'] * 100:>8.2f}%"
        f"     "
        f"{test_results['precision'] * 100:>8.2f}%"
    )

    print(
        f"Recall           "
        f"{validation_results['recall'] * 100:>8.2f}%"
        f"     "
        f"{test_results['recall'] * 100:>8.2f}%"
    )

    print(
        f"F1-score         "
        f"{validation_results['f1'] * 100:>8.2f}%"
        f"     "
        f"{test_results['f1'] * 100:>8.2f}%"
    )

    print(
        f"ROC-AUC          "
        f"{validation_results['roc_auc']:>8.4f}"
        f"     "
        f"{test_results['roc_auc']:>8.4f}"
    )

    print("\nEvaluation completed.")