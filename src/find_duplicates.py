import os
from PIL import Image
import numpy as np

dataset_path = r"C:\Users\Hi\saree-design-recognition\data\sarees_dataset\handloom_sarees"


def average_hash(image, hash_size=16):
    image = image.convert("L")
    image = image.resize((hash_size, hash_size))

    pixels = np.array(image)
    average = pixels.mean()

    return pixels > average


def hamming_distance(hash1, hash2):
    return np.count_nonzero(hash1 != hash2)


# Get all image files
image_files = []

for file in os.listdir(dataset_path):
    if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        image_files.append(file)

image_files.sort()

print("Total images:", len(image_files))

# Calculate hashes
hashes = {}

for file in image_files:
    path = os.path.join(dataset_path, file)

    try:
        with Image.open(path) as image:
            hashes[file] = average_hash(image)

    except Exception as e:
        print("Could not read:", file, e)


# Find very similar images
threshold = 5

similar_pairs = []

for i in range(len(image_files)):

    file1 = image_files[i]

    if file1 not in hashes:
        continue

    for j in range(i + 1, len(image_files)):

        file2 = image_files[j]

        if file2 not in hashes:
            continue

        distance = hamming_distance(
            hashes[file1],
            hashes[file2]
        )

        if distance <= threshold:
            similar_pairs.append(
                (file1, file2, distance)
            )


print("\nSimilar image pairs:")
print("====================")

if len(similar_pairs) == 0:

    print("No highly similar pairs found.")

else:

    for file1, file2, distance in similar_pairs:

        print(
            f"{file1} <--> {file2} | "
            f"distance = {distance}"
        )

print("\nTotal similar pairs:", len(similar_pairs))
