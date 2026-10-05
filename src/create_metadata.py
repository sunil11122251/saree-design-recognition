import os
import csv

# Project paths
kaggle_path = r"data\indian_saree_patterns"
deeplure_path = r"data\sarees_dataset\handloom_sarees"

output_path = r"results\metadata.csv"


# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)


rows = []


# =========================================================
# 1. KAGGLE DATASET
# =========================================================

splits = ["train", "valid", "test"]

for split in splits:

    split_path = os.path.join(kaggle_path, split)

    if not os.path.exists(split_path):
        print("Folder not found:", split_path)
        continue

    for class_name in os.listdir(split_path):

        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):

            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):

                relative_path = os.path.join(
                    "data",
                    "indian_saree_patterns",
                    split,
                    class_name,
                    filename
                )

                rows.append([
                    filename,
                    "kaggle",
                    split,
                    class_name,
                    relative_path
                ])


# =========================================================
# 2. DEEPLURE DATASET
# =========================================================

if os.path.exists(deeplure_path):

    for filename in os.listdir(deeplure_path):

        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):

            relative_path = os.path.join(
                "data",
                "sarees_dataset",
                "handloom_sarees",
                filename
            )

            rows.append([
                filename,
                "deeplure",
                "unlabeled",
                "unlabeled",
                relative_path
            ])


# =========================================================
# 3. SAVE CSV
# =========================================================

with open(output_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow([
        "filename",
        "source",
        "split",
        "class",
        "path"
    ])

    writer.writerows(rows)


# =========================================================
# 4. SUMMARY
# =========================================================

print("\nMetadata created successfully.")
print("Output:", output_path)
print("Total records:", len(rows))

print("\nRecords by source:")

kaggle_count = sum(1 for row in rows if row[1] == "kaggle")
deeplure_count = sum(1 for row in rows if row[1] == "deeplure")

print("Kaggle:", kaggle_count)
print("DeepLure:", deeplure_count)
