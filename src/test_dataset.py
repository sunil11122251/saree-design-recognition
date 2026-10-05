from dataset import SareeDataset


dataset = SareeDataset(
    csv_file="results/metadata.csv",
    split="train"
)

print("Number of images:", len(dataset))

print("Classes:")
print(dataset.class_to_idx)

sample = dataset[0]

print("\nSample:")
print("Image shape:", sample["image"].shape)
print("Label:", sample["label"])
print("Group ID:", sample["group_id"])
print("Path:", sample["path"])