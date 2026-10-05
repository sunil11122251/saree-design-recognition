import pandas as pd
import random
from itertools import combinations
import os


class PairDatasetBuilder:

    def __init__(self, csv_file, split="train", seed=42):

        self.df = pd.read_csv(csv_file)

        # Use only Kaggle dataset
        self.df = self.df[
            (self.df["source"] == "kaggle") &
            (self.df["split"] == split)
        ].copy()

        self.seed = seed
        random.seed(seed)

        # Create a unique ID for every image
        self.df["image_id"] = (
            self.df["class"]
            + "__"
            + self.df["filename"]
        )

    def create_train_pairs(self):

        # -----------------------------------------
        # TRAIN PAIRS
        # -----------------------------------------

        # For training we use the existing source-image
        # groups created from the filename prefix.

        self.df["group_id"] = (
            self.df["class"]
            + "__"
            + self.df["filename"].str.split(".rf.").str[0]
        )

        groups = {}

        for group_id, group in self.df.groupby("group_id"):
            groups[group_id] = group.to_dict("records")

        group_ids = list(groups.keys())

        # Positive pairs
        positive_pairs = []

        for group_id in group_ids:

            images = groups[group_id]

            for image1, image2 in combinations(images, 2):

                positive_pairs.append({
                    "image1": image1["path"],
                    "image2": image2["path"],
                    "group1": group_id,
                    "group2": group_id,
                    "class1": image1["class"],
                    "class2": image2["class"],
                    "label": 1
                })

        # Negative pairs
        negative_pairs = []

        group_combinations = list(
            combinations(group_ids, 2)
        )

        random.shuffle(group_combinations)

        for group1_id, group2_id in group_combinations:

            group1 = groups[group1_id][0]
            group2 = groups[group2_id][0]

            negative_pairs.append({
                "image1": group1["path"],
                "image2": group2["path"],
                "group1": group1_id,
                "group2": group2_id,
                "class1": group1["class"],
                "class2": group2["class"],
                "label": 0
            })

            if len(negative_pairs) >= len(positive_pairs):
                break

        pairs = positive_pairs + negative_pairs

        random.shuffle(pairs)

        return pd.DataFrame(pairs)

    def create_eval_pairs(self):

        # -----------------------------------------
        # VALIDATION / TEST PAIRS
        # -----------------------------------------

        images = self.df.to_dict("records")

        positive_pairs = []

        # Each image is paired with itself.
        # During evaluation we will apply different
        # color transformations to the two copies.
        for image in images:

            positive_pairs.append({
                "image1": image["path"],
                "image2": image["path"],
                "group1": image["image_id"],
                "group2": image["image_id"],
                "class1": image["class"],
                "class2": image["class"],
                "label": 1
            })

        # Create negative pairs
        negative_pairs = []

        image_combinations = list(
            combinations(images, 2)
        )

        random.shuffle(image_combinations)

        for image1, image2 in image_combinations:

            # Different image = negative pair
            negative_pairs.append({
                "image1": image1["path"],
                "image2": image2["path"],
                "group1": image1["image_id"],
                "group2": image2["image_id"],
                "class1": image1["class"],
                "class2": image2["class"],
                "label": 0
            })

            if len(negative_pairs) >= len(positive_pairs):
                break

        pairs = positive_pairs + negative_pairs

        random.shuffle(pairs)

        return pd.DataFrame(pairs)


# =================================================
# MAIN
# =================================================

if __name__ == "__main__":

    for split in ["train", "valid", "test"]:

        print("\n" + "=" * 50)
        print(f"Creating {split} pairs...")
        print("=" * 50)

        builder = PairDatasetBuilder(
            csv_file="results/metadata.csv",
            split=split,
            seed=42
        )

        # Train uses actual grouped variants
        if split == "train":
            pairs = builder.create_train_pairs()

        # Validation/Test use image + color-transformed
        # version of the same image
        else:
            pairs = builder.create_eval_pairs()

        positive_count = (pairs["label"] == 1).sum()
        negative_count = (pairs["label"] == 0).sum()

        print("Total pairs:", len(pairs))
        print("Positive pairs:", positive_count)
        print("Negative pairs:", negative_count)

        # Check positive pairs
        positive = pairs[pairs["label"] == 1]

        invalid_positive = positive[
            positive["class1"] != positive["class2"]
        ]

        print(
            "Invalid positive pairs:",
            len(invalid_positive)
        )

        # Save
        os.makedirs("results", exist_ok=True)

        output_path = f"results/{split}_pairs.csv"

        pairs.to_csv(
            output_path,
            index=False
        )

        print("Saved:", output_path)