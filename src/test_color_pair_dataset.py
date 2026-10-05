from color_pair_torch_dataset import ColorPairSareeDataset


# ------------------------------------------
# Validation dataset
# ------------------------------------------

valid_dataset = ColorPairSareeDataset(
    pairs_csv="results/valid_color_pairs.csv",
    train=False
)

print("Number of validation pairs:")
print(len(valid_dataset))


image1, image2, label = valid_dataset[0]

print()
print("First validation pair:")
print("Image 1 shape:", image1.shape)
print("Image 2 shape:", image2.shape)
print("Label:", label)


# ------------------------------------------
# Test dataset
# ------------------------------------------

test_dataset = ColorPairSareeDataset(
    pairs_csv="results/test_color_pairs.csv",
    train=False
)

print()
print("Number of test pairs:")
print(len(test_dataset))


image1, image2, label = test_dataset[0]

print()
print("First test pair:")
print("Image 1 shape:", image1.shape)
print("Image 2 shape:", image2.shape)
print("Label:", label)