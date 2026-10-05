from pair_torch_dataset import PairSareeDataset


# ==========================================
# TEST VALIDATION DATASET
# ==========================================

valid_dataset = PairSareeDataset(
    pairs_csv="results/valid_pairs.csv",
    train=False
)

print("Number of validation pairs:", len(valid_dataset))

image1, image2, label = valid_dataset[0]

print("\nFirst validation pair:")
print("Image 1 shape:", image1.shape)
print("Image 2 shape:", image2.shape)
print("Label:", label)


# ==========================================
# TEST TEST DATASET
# ==========================================

test_dataset = PairSareeDataset(
    pairs_csv="results/test_pairs.csv",
    train=False
)

print("\nNumber of test pairs:", len(test_dataset))

image1, image2, label = test_dataset[0]

print("\nFirst test pair:")
print("Image 1 shape:", image1.shape)
print("Image 2 shape:", image2.shape)
print("Label:", label)