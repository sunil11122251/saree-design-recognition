import os
from PIL import Image

# Dataset paths
kaggle_path = r"C:\Users\Hi\saree-design-recognition\data\indian_saree_patterns"
deeplure_path = r"C:\Users\Hi\saree-design-recognition\data\sarees_dataset"

def inspect_folder(folder_path):
    total_images = 0
    extensions = {}
    sizes = {}

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                total_images += 1

                extension = os.path.splitext(file)[1].lower()
                extensions[extension] = extensions.get(extension, 0) + 1

                image_path = os.path.join(root, file)

                try:
                    with Image.open(image_path) as img:
                        sizes[img.size] = sizes.get(img.size, 0) + 1
                except Exception:
                    print("Could not read:", image_path)

    return total_images, extensions, sizes


print("\n========== KAGGLE DATASET ==========")

total, extensions, sizes = inspect_folder(kaggle_path)

print("Total images:", total)
print("File types:", extensions)

print("\nImage sizes:")
for size, count in sizes.items():
    print(size, ":", count)


print("\n========== DEEPLURE DATASET ==========")

total, extensions, sizes = inspect_folder(deeplure_path)

print("Total images:", total)
print("File types:", extensions)

print("\nImage sizes:")
for size, count in sizes.items():
    print(size, ":", count)