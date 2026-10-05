import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SareeDataset(Dataset):

    def __init__(self, csv_file, split="train"):
        
        # Load metadata
        self.df = pd.read_csv(csv_file)

        # Use only Kaggle images for now
        self.df = self.df[
            (self.df["source"] == "kaggle") &
            (self.df["split"] == split)
        ].reset_index(drop=True)

        # Extract group ID
        self.df["group_id"] = (
            self.df["filename"]
            .str.split(".rf.")
            .str[0]
        )

        # Pattern classes
        classes = sorted(self.df["class"].unique())

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(classes)
        }

        self.idx_to_class = {
            idx: class_name
            for class_name, idx in self.class_to_idx.items()
        }

        # Training transformations
        if split == "train":

            self.transform = transforms.Compose([

                transforms.Resize((224, 224)),

                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.6,
                    hue=0.08
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        else:

            # Validation / test
            self.transform = transforms.Compose([

                transforms.Resize((224, 224)),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


    def __len__(self):

        return len(self.df)


    def __getitem__(self, index):

        row = self.df.iloc[index]

        # Image path
        image_path = row["path"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        image = self.transform(image)

        # Class label
        label = self.class_to_idx[row["class"]]

        # Return information
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "group_id": row["group_id"],
            "path": image_path
        }