import os
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance


# --------------------------------------------------
# Configuration
# --------------------------------------------------

METADATA_FILE = "results/metadata.csv"
OUTPUT_DIR = "data/color_variants"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# --------------------------------------------------
# Color transformation
# --------------------------------------------------

def create_color_variant(image, variant_number, seed):
    """
    Create a strong synthetic color/palette variation
    while preserving the underlying image structure.
    """

    rng = random.Random(seed)

    image = image.convert("RGB")

    # Convert RGB -> HSV
    hsv = np.array(image.convert("HSV")).astype(np.uint8)

    # PIL HSV:
    # H = 0-255
    # S = 0-255
    # V = 0-255

    hue_shift = {
        1: 35,
        2: 75,
        3: 115
    }[variant_number]

    # ----------------------------------------------
    # 1. Shift hue
    # ----------------------------------------------

    hsv[:, :, 0] = (
        hsv[:, :, 0].astype(np.int16) + hue_shift
    ) % 256

    # ----------------------------------------------
    # 2. Change saturation
    # ----------------------------------------------

    saturation_factor = {
        1: 1.15,
        2: 0.75,
        3: 1.30
    }[variant_number]

    saturation = hsv[:, :, 1].astype(np.float32)
    saturation = saturation * saturation_factor
    saturation = np.clip(saturation, 0, 255)

    hsv[:, :, 1] = saturation.astype(np.uint8)

    # ----------------------------------------------
    # 3. Change brightness/value
    # ----------------------------------------------

    brightness_factor = {
        1: 1.05,
        2: 0.90,
        3: 1.10
    }[variant_number]

    value = hsv[:, :, 2].astype(np.float32)
    value = value * brightness_factor
    value = np.clip(value, 0, 255)

    hsv[:, :, 2] = value.astype(np.uint8)

    # Convert back to RGB
    result = Image.fromarray(hsv, mode="HSV").convert("RGB")

    # ----------------------------------------------
    # 4. Small contrast variation
    # ----------------------------------------------

    contrast_factor = {
        1: 1.05,
        2: 0.95,
        3: 1.10
    }[variant_number]

    result = ImageEnhance.Contrast(result).enhance(
        contrast_factor
    )

    return result


# --------------------------------------------------
# Find unique source images
# --------------------------------------------------

def get_unique_images(df):
    """
    The Kaggle dataset contains three generated versions
    of each source image.

    The filename before '.rf.' identifies the source image.
    We keep only one source image from each group.
    """

    df = df.copy()

    df["group_id"] = (
        df["class"]
        + "__"
        + df["filename"].str.split(".rf.").str[0]
    )

    unique_df = (
        df
        .sort_values("filename")
        .groupby("group_id", as_index=False)
        .first()
    )

    return unique_df


# --------------------------------------------------
# Generate variants for one split
# --------------------------------------------------

def process_split(df, split):

    split_df = df[df["split"] == split].copy()

    # Only use Kaggle dataset
    split_df = split_df[
        split_df["source"] == "kaggle"
    ].copy()

    unique_df = get_unique_images(split_df)

    print()
    print("=" * 60)
    print(f"Processing split: {split}")
    print("=" * 60)

    print("Original images/groups:", len(unique_df))

    generated_count = 0

    for _, row in unique_df.iterrows():

        source_path = row["path"]
        class_name = row["class"]
        group_id = row["group_id"]

        if not os.path.exists(source_path):
            print("WARNING - image not found:")
            print(source_path)
            continue

        image = Image.open(source_path).convert("RGB")

        # Output directory
        group_dir = os.path.join(
            OUTPUT_DIR,
            split,
            class_name,
            group_id
        )

        os.makedirs(group_dir, exist_ok=True)

        # ------------------------------------------
        # Save original
        # ------------------------------------------

        original_path = os.path.join(
            group_dir,
            "original.jpg"
        )

        image.save(
            original_path,
            quality=95
        )

        # ------------------------------------------
        # Generate 3 color variants
        # ------------------------------------------

        for variant_number in [1, 2, 3]:

            variant = create_color_variant(
                image=image,
                variant_number=variant_number,
                seed=SEED + variant_number
            )

            variant_path = os.path.join(
                group_dir,
                f"color_variant_{variant_number}.jpg"
            )

            variant.save(
                variant_path,
                quality=95
            )

            generated_count += 1

    print("Generated color variants:", generated_count)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading metadata...")

    df = pd.read_csv(METADATA_FILE)

    print("Total metadata records:", len(df))

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each split separately
    for split in ["train", "valid", "test"]:
        process_split(df, split)

    print()
    print("=" * 60)
    print("COLOR VARIANT GENERATION COMPLETED")
    print("=" * 60)

    print()
    print("Output directory:")
    print(OUTPUT_DIR)