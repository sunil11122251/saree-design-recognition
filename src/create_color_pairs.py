import os
import random
from itertools import combinations

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

METADATA_FILE = "results/metadata.csv"

COLOR_VARIANTS_DIR = "data/color_variants"

OUTPUT_DIR = "results"

SEED = 42

random.seed(SEED)


# --------------------------------------------------
# Get groups from metadata
# --------------------------------------------------

def get_groups(df, split):

    split_df = df[
        (df["source"] == "kaggle") &
        (df["split"] == split)
    ].copy()

    # Same source image group
    split_df["group_id"] = (
        split_df["class"]
        + "__"
        + split_df["filename"].str.split(".rf.").str[0]
    )

    groups = {}

    for group_id, group in split_df.groupby("group_id"):

        first_row = group.iloc[0]

        class_name = first_row["class"]

        group_dir = os.path.join(
            COLOR_VARIANTS_DIR,
            split,
            class_name,
            group_id
        )

        if not os.path.exists(group_dir):
            continue

        image_paths = []

        for name in [
            "original.jpg",
            "color_variant_1.jpg",
            "color_variant_2.jpg",
            "color_variant_3.jpg"
        ]:

            path = os.path.join(group_dir, name)

            if os.path.exists(path):
                image_paths.append(path)

        # Require all four images
        if len(image_paths) == 4:

            groups[group_id] = {
                "group_id": group_id,
                "class": class_name,
                "images": image_paths
            }

    return groups


# --------------------------------------------------
# Create positive pairs
# --------------------------------------------------

def create_positive_pairs(groups):

    positive_pairs = []

    for group_id, group in groups.items():

        images = group["images"]

        # Every combination of the four versions
        for image1, image2 in combinations(images, 2):

            positive_pairs.append({

                "image1": image1,
                "image2": image2,

                "group1": group_id,
                "group2": group_id,

                "class1": group["class"],
                "class2": group["class"],

                "label": 1
            })

    return positive_pairs


# --------------------------------------------------
# Create negative pairs
# --------------------------------------------------

def create_negative_pairs(groups, target_count):

    group_ids = list(groups.keys())

    combinations_of_groups = list(
        combinations(group_ids, 2)
    )

    random.shuffle(combinations_of_groups)

    negative_pairs = []

    for group1_id, group2_id in combinations_of_groups:

        group1 = groups[group1_id]
        group2 = groups[group2_id]

        # Make sure they are genuinely different groups
        if group1_id == group2_id:
            continue

        # Randomly select one version from each group
        image1 = random.choice(group1["images"])
        image2 = random.choice(group2["images"])

        negative_pairs.append({

            "image1": image1,
            "image2": image2,

            "group1": group1_id,
            "group2": group2_id,

            "class1": group1["class"],
            "class2": group2["class"],

            "label": 0
        })

        if len(negative_pairs) >= target_count:
            break

    return negative_pairs


# --------------------------------------------------
# Create pairs for one split
# --------------------------------------------------

def create_pairs_for_split(df, split):

    print()
    print("=" * 70)
    print(f"Creating color-invariance pairs: {split}")
    print("=" * 70)

    groups = get_groups(df, split)

    print("Number of design groups:", len(groups))

    # ----------------------------------------------
    # Positive pairs
    # ----------------------------------------------

    positive_pairs = create_positive_pairs(groups)

    print("Positive pairs:", len(positive_pairs))

    # ----------------------------------------------
    # Negative pairs
    # ----------------------------------------------

    negative_pairs = create_negative_pairs(
        groups,
        target_count=len(positive_pairs)
    )

    print("Negative pairs:", len(negative_pairs))

    # ----------------------------------------------
    # Combine
    # ----------------------------------------------

    pairs = positive_pairs + negative_pairs

    random.shuffle(pairs)

    pairs_df = pd.DataFrame(pairs)

    # ----------------------------------------------
    # Validation checks
    # ----------------------------------------------

    invalid_positive = pairs_df[
        (pairs_df["label"] == 1) &
        (pairs_df["group1"] != pairs_df["group2"])
    ]

    invalid_negative = pairs_df[
        (pairs_df["label"] == 0) &
        (pairs_df["group1"] == pairs_df["group2"])
    ]

    print("Invalid positive pairs:", len(invalid_positive))
    print("Invalid negative pairs:", len(invalid_negative))

    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{split}_color_pairs.csv"
    )

    pairs_df.to_csv(
        output_path,
        index=False
    )

    print("Total pairs:", len(pairs_df))
    print("Saved:", output_path)

    return pairs_df


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading metadata...")

    df = pd.read_csv(METADATA_FILE)

    print("Total metadata records:", len(df))

    for split in [
        "train",
        "valid",
        "test"
    ]:

        create_pairs_for_split(
            df,
            split
        )

    print()
    print("=" * 70)
    print("COLOR-INVARIANCE PAIR GENERATION COMPLETED")
    print("=" * 70)