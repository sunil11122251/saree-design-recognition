from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# 1. Select one saree image
# --------------------------------------------------

image_path = r"data\indian_saree_patterns\train\Pichwai"

# Automatically take the first image from the folder
files = [
    f for f in os.listdir(image_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_file = os.path.join(image_path, files[0])

image = Image.open(image_file).convert("RGB")

print("Testing image:")
print(image_file)


# --------------------------------------------------
# 2. Define color transformations
# --------------------------------------------------

augmentations = [
    transforms.Compose([
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.4,
            hue=0.05
        )
    ]),

    transforms.Compose([
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.7,
            hue=0.10
        )
    ]),

    transforms.Compose([
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.2,
            saturation=1.0,
            hue=0.15
        )
    ]),

    transforms.Compose([
        transforms.RandomGrayscale(p=1.0)
    ])
]


# --------------------------------------------------
# 3. Create visualization
# --------------------------------------------------

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# Original
axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis("off")

# Augmented versions
for i, transform in enumerate(augmentations):

    augmented_image = transform(image)

    axes[i + 1].imshow(augmented_image)
    axes[i + 1].set_title(f"Augmented {i + 1}")
    axes[i + 1].axis("off")


plt.tight_layout()

# --------------------------------------------------
# 4. Save result
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

output_path = "results/color_augmentation_test.jpg"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

print("\nSaved:")
print(output_path)
