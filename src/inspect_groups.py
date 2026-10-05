import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load metadata
df = pd.read_csv("results/metadata.csv")

# Keep only Kaggle training images
df = df[df["source"] == "kaggle"].copy()

# Extract original source-image ID
df["group_id"] = df["filename"].str.split(".rf.").str[0]

# Find groups having 3 images
group_counts = df["group_id"].value_counts()
groups = group_counts[group_counts == 3].index

# Select first 3 groups
selected_groups = list(groups[:3])

print("Selected groups:")
for group in selected_groups:
    print(group)

# Display images
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for row, group in enumerate(selected_groups):

    group_images = df[df["group_id"] == group]

    for col, (_, item) in enumerate(group_images.iterrows()):

        image_path = item["path"]

        image = Image.open(image_path).convert("RGB")

        axes[row, col].imshow(image)
        axes[row, col].set_title(
            f"{item['class']}\nVersion {col + 1}"
        )
        axes[row, col].axis("off")

plt.tight_layout()

# Save result
os.makedirs("results", exist_ok=True)
plt.savefig("results/group_inspection.jpg", dpi=200)

print("\nSaved:")
print("results/group_inspection.jpg")