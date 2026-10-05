import pandas as pd

from PIL import Image

import torch

from torch.utils.data import Dataset

import torchvision.transforms as transforms


class ColorPairSareeDataset(Dataset):

    def __init__(self, pairs_csv, train=True):

        self.df = pd.read_csv(pairs_csv)

        self.train = train

        # ------------------------------------------
        # Basic preprocessing
        # ------------------------------------------

        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ------------------------------------------
        # Training transformation
        #
        # We already generated strong color
        # variations before creating the pairs.
        #
        # Therefore we do NOT apply another strong
        # ColorJitter here.
        # ------------------------------------------

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
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

        # ------------------------------------------
        # Load images
        # ------------------------------------------

        image1 = Image.open(
            row["image1"]
        ).convert("RGB")

        image2 = Image.open(
            row["image2"]
        ).convert("RGB")

        # ------------------------------------------
        # Label
        # ------------------------------------------

        label = torch.tensor(
            row["label"],
            dtype=torch.float32
        )

        # ------------------------------------------
        # Transform
        # ------------------------------------------

        if self.train:

            image1 = self.train_transform(image1)

            image2 = self.train_transform(image2)

        else:

            image1 = self.base_transform(image1)

            image2 = self.base_transform(image2)

        return image1, image2, label