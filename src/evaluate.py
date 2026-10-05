import torch
from torch.utils.data import DataLoader

from pair_torch_dataset import PairSareeDataset
from model import SareeEmbeddingModel


MODEL_FILE = "models/saree_embedding_model.pth"

BATCH_SIZE = 16
EMBEDDING_DIM = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate_model(pairs_file, dataset_name):

    print("\n" + "=" * 50)
    print(f"Evaluating on {dataset_name}")
    print("=" * 50)

    # Dataset
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

    # Model
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

    positive_distances = []
    negative_distances = []

    # -----------------------------------------
    # EVALUATION
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

            for distance, label in zip(
                distances.cpu(),
                labels
            ):

                if label.item() == 1:
                    positive_distances.append(
                        distance.item()
                    )
                else:
                    negative_distances.append(
                        distance.item()
                    )

    # -----------------------------------------
    # DISTANCE STATISTICS
    # -----------------------------------------

    positive_average = (
        sum(positive_distances)
        / len(positive_distances)
    )

    negative_average = (
        sum(negative_distances)
        / len(negative_distances)
    )

    print("\nPositive pairs:", len(positive_distances))
    print("Negative pairs:", len(negative_distances))

    print("\nAverage positive distance:")
    print(f"{positive_average:.4f}")

    print("\nAverage negative distance:")
    print(f"{negative_average:.4f}")

    # -----------------------------------------
    # THRESHOLD
    # -----------------------------------------

    threshold = 0.5

    correct = 0
    total = 0

    # Positive pairs
    for distance in positive_distances:

        prediction = 1 if distance < threshold else 0

        if prediction == 1:
            correct += 1

        total += 1

    # Negative pairs
    for distance in negative_distances:

        prediction = 1 if distance < threshold else 0

        if prediction == 0:
            correct += 1

        total += 1

    accuracy = correct / total

    print("\nThreshold:", threshold)

    print("Correct predictions:", correct)
    print("Total predictions:", total)

    print("\nVerification accuracy:")
    print(f"{accuracy * 100:.2f}%")

    return accuracy


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("Device:", DEVICE)

    print("\nLoading trained model:")
    print(MODEL_FILE)

    valid_accuracy = evaluate_model(
        "results/valid_pairs.csv",
        "VALIDATION SET"
    )

    test_accuracy = evaluate_model(
        "results/test_pairs.csv",
        "TEST SET"
    )

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)

    print(
        f"Validation accuracy: "
        f"{valid_accuracy * 100:.2f}%"
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )