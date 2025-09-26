import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class ISIC2019Dataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        # Image directory and CSV file
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # List of target classes
        self.label_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

        # Remove rows with unknown labels
        if "UNK" in self.data.columns:
            self.data = self.data[self.data[self.label_cols].sum(axis=1) > 0].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row["image"] + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)

        # Load image in RGB mode
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert label to tensor
        label = torch.tensor(row[self.label_cols].values.astype("float32"))
        return image, label
